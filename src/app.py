"""
CRUCIBLE-01 — The Forge (NautilusTrader evidence engine)
v1.0

Melts strategies down under real historical heat so the GECKO network can see
what they are actually made of:
  load     (historical candles from Hyperliquid's public API)
  → forge  (NautilusTrader backtest: real fills, fees, margin, slippage-aware)
  → assay  (PF, win rate, drawdown, fees paid — computed from the raw ledger)
  → archive (every assay persisted on the /data volume, forever)

CONSTITUTION: the Crucible measures; it never trades. An assay is evidence,
not an order. v1 ships three reference templates (BTC/ETH); per-card
translations from CODEX-01 are deliberate engineering, added one by one.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.examples.strategies.ema_cross import EMACross, EMACrossConfig
from src.crucible_trend import CrucibleTrend, CrucibleTrendConfig
from src.crucible_reversion import CrucibleReversion, CrucibleReversionConfig
from src.crucible_pullback import CruciblePullback, CruciblePullbackConfig
from src.crucible_gaussian import CrucibleGaussian, CrucibleGaussianConfig
from src.crucible_rotation import run_rotation
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider

PORT = int(os.environ.get("PORT", "3031"))
HL_API = os.environ.get("HL_API", "https://api.hyperliquid.xyz/info")
STATE_DIR = os.environ.get("STATE_DIR", "/data" if os.path.isdir("/data") else "./data")
RESULTS_FILE = os.path.join(STATE_DIR, "forge-results.json")
START_BALANCE = float(os.environ.get("START_BALANCE", "100000"))
MAX_DAYS = int(os.environ.get("MAX_DAYS", "365"))

app = FastAPI(title="CRUCIBLE-01", version="1.0")
_forge_lock = threading.Lock()
_started = time.time()

# ── Instruments (v1: symbols with battle-tested definitions; fees noted) ─────
def _mk_perp(base_code, price_prec, price_inc, size_prec, size_inc):
    """Factory for additional USDT-margined perps mirroring the test-kit definitions
    (same fee schedule: maker 0.02% / taker 0.04% — conservative vs Hyperliquid).
    Added 2026-08-17 for the diversity assays directed by Dr K."""
    from nautilus_trader.model import currencies as _cur
    from nautilus_trader.model.identifiers import InstrumentId, Symbol
    from nautilus_trader.model.instruments import CryptoPerpetual
    from nautilus_trader.model.objects import Price, Quantity

    base = getattr(_cur, base_code)

    def build():
        return CryptoPerpetual(
            instrument_id=InstrumentId(Symbol(f"{base_code}USDT-PERP"), Venue("BINANCE")),
            raw_symbol=Symbol(f"{base_code}USDT"),
            base_currency=base, quote_currency=USDT, settlement_currency=USDT,
            is_inverse=False,
            price_precision=price_prec, size_precision=size_prec,
            price_increment=Price.from_str(price_inc), size_increment=Quantity.from_str(size_inc),
            max_quantity=Quantity.from_str("100000000"), min_quantity=Quantity.from_str(size_inc),
            max_notional=None, min_notional=Money(10.00, USDT),
            max_price=Price.from_str("1000000"), min_price=Price.from_str(price_inc),
            margin_init=Decimal("1.00"), margin_maint=Decimal("0.35"),
            maker_fee=Decimal("0.0002"), taker_fee=Decimal("0.0004"),
            ts_event=0, ts_init=0,
        )
    return build

INSTRUMENTS = {
    "BTC": TestInstrumentProvider.btcusdt_perp_binance,
    "ETH": TestInstrumentProvider.ethusdt_perp_binance,
    "SOL": _mk_perp("SOL", 3, "0.001", 1, "0.1"),
    "XRP": _mk_perp("XRP", 4, "0.0001", 0, "1"),
    "DOGE": _mk_perp("DOGE", 5, "0.00001", 0, "1"),
}
FEE_NOTE = ("v1 fee model: Binance-perp instrument definitions (maker 0.02%/taker 0.04%) — "
            "slightly conservative vs Hyperliquid taker 0.035%. Slippage beyond spread not modeled in v1.")

# ── Strategy templates ───────────────────────────────────────────────────────
def _mk_ema_cross(iid, bar_type, p):
    return EMACross(EMACrossConfig(
        instrument_id=iid, bar_type=bar_type,
        trade_size=Decimal(str(p.get("trade_size", 0.1))),
        fast_ema_period=int(p.get("fast", 12)), slow_ema_period=int(p.get("slow", 48))))

def _mk_crucible_trend(iid, bar_type, p):
    return CrucibleTrend(CrucibleTrendConfig(
        instrument_id=iid, bar_type=bar_type,
        trade_size=Decimal(str(p.get("trade_size", 0.1))),
        fast_ema_period=int(p.get("fast", 12)), slow_ema_period=int(p.get("slow", 48)),
        atr_period=int(p.get("atr_period", 14)),
        atr_stop_mult=float(p.get("atr_stop_mult", 1.5)),
        target_r=float(p.get("target_r", 2.0)),
        max_hold_bars=int(p.get("max_hold_bars", 72)),
        allow_shorts=bool(p.get("allow_shorts", True))))

def _mk_crucible_reversion(iid, bar_type, p):
    return CrucibleReversion(CrucibleReversionConfig(
        instrument_id=iid, bar_type=bar_type,
        trade_size=Decimal(str(p.get("trade_size", 0.1))),
        fast_ema_period=int(p.get("fast", 12)), slow_ema_period=int(p.get("slow", 48)),
        regime_band_pct=float(p.get("regime_band_pct", 0.75)),
        rsi_period=int(p.get("rsi_period", 14)),
        rsi_buy=float(p.get("rsi_buy", 30.0)), rsi_sell=float(p.get("rsi_sell", 70.0)),
        atr_period=int(p.get("atr_period", 14)),
        atr_stop_mult=float(p.get("atr_stop_mult", 1.5)),
        target_r=float(p.get("target_r", 2.0)),
        max_hold_bars=int(p.get("max_hold_bars", 72)),
        allow_shorts=bool(p.get("allow_shorts", True))))

def _mk_crucible_pullback(iid, bar_type, p):
    return CruciblePullback(CruciblePullbackConfig(
        instrument_id=iid, bar_type=bar_type,
        trade_size=Decimal(str(p.get("trade_size", 0.1))),
        fast_ema_period=int(p.get("fast", 8)),
        mid_ema_period=int(p.get("mid", 21)),
        slow_ema_period=int(p.get("slow", 50)),
        atr_period=int(p.get("atr_period", 14)),
        atr_stop_mult=float(p.get("atr_stop_mult", 1.5)),
        target_r=float(p.get("target_r", 2.0)),
        max_hold_bars=int(p.get("max_hold_bars", 72)),
        allow_shorts=bool(p.get("allow_shorts", True)),
        min_stack_sep_pct=float(p.get("min_stack_sep_pct", 0.0))))

def _mk_crucible_gaussian(iid, bar_type, p):
    return CrucibleGaussian(CrucibleGaussianConfig(
        instrument_id=iid, bar_type=bar_type,
        trade_size=Decimal(str(p.get("trade_size", 0.1))),
        gaussian_length=int(p.get("gaussian_length", 14)),
        gaussian_poles=int(p.get("gaussian_poles", 4)),
        atr_period=int(p.get("atr_period", 14)),
        fixed_multiplier=float(p.get("fixed_multiplier", 1.5)),
        adaptive_width=bool(p.get("adaptive_width", True)),
        trend_multiplier=float(p.get("trend_multiplier", 0.8)),
        chop_multiplier=float(p.get("chop_multiplier", 2.5)),
        efficiency_length=int(p.get("efficiency_length", 10)),
        efficiency_smooth=int(p.get("efficiency_smooth", 5)),
        atr_stop_mult=float(p.get("atr_stop_mult", 1.5)),
        target_r=float(p.get("target_r", 2.0)),
        max_hold_bars=int(p.get("max_hold_bars", 72)),
        allow_shorts=bool(p.get("allow_shorts", True))))

STRATEGIES = {
    "crucible_gaussian": {"make": _mk_crucible_gaussian,
                          "desc": "CODEX card #71 (Gaussian Filter Trend [QuantAlgo]) entry architecture: Ehlers N-pole Gaussian basis, ER-adaptive ATR deadband, ratcheting trend line; entries on direction flips only, CrucibleTrend management. Params: gaussian_length, gaussian_poles, atr_period, fixed_multiplier, adaptive_width, trend_multiplier, chop_multiplier, efficiency_length, efficiency_smooth, atr_stop_mult, target_r, max_hold_bars, allow_shorts, trade_size."},
    "crucible_pullback": {"make": _mk_crucible_pullback,
                          "desc": "CODEX card #66 (Setup123) entry architecture: EMA-stack trend filter (fast>mid>slow), pullback below fast EMA then resume above it, CrucibleTrend management. Isolates entry quality vs crucible_trend. Params: fast, mid, slow, atr_period, atr_stop_mult, target_r, max_hold_bars, allow_shorts, trade_size, min_stack_sep_pct (regime gate: min fast/slow EMA separation as % of close before a stack counts; 0.0 = off)."},
    "ema_cross": {"make": _mk_ema_cross,
                  "desc": "Canonical EMA cross, market in/out. Params: fast, slow, trade_size."},
    "crucible_reversion": {"make": _mk_crucible_reversion,
                           "desc": "Regime-gated mean reversion (CODEX card #4 style): trades only ranging tape (|fastEMA-slowEMA|/close < regime_band_pct), RSI extremes entries, CrucibleTrend management. Params: fast, slow, regime_band_pct, rsi_period, rsi_buy, rsi_sell, atr_period, atr_stop_mult, target_r, max_hold_bars, allow_shorts, trade_size."},
    "crucible_trend": {"make": _mk_crucible_trend,
                       "desc": "The forge's native template — SUPREME-LEADER mechanics: EMA-cross entries (both sides), ATR stop, R-multiple target, time stop, bar-close management. Params: fast, slow, atr_period, atr_stop_mult, target_r, max_hold_bars, allow_shorts, trade_size."},
}

# ── Data: Hyperliquid public candles ─────────────────────────────────────────
INTERVAL_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
BAR_SPEC = {"15m": "15-MINUTE", "1h": "1-HOUR", "4h": "4-HOUR", "1d": "1-DAY"}

def fetch_hl_candles(coin: str, interval: str, days: int) -> pd.DataFrame:
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    step = INTERVAL_MS[interval]
    rows, cursor = [], start
    with httpx.Client(timeout=30) as client:
        while cursor < end:
            r = client.post(HL_API, json={"type": "candleSnapshot",
                                          "req": {"coin": coin, "interval": interval,
                                                  "startTime": cursor, "endTime": end}})
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            last_t = int(batch[-1]["t"])
            nxt = last_t + step
            if nxt <= cursor:
                break
            cursor = nxt
            if len(batch) < 2:
                break
    if not rows:
        raise HTTPException(502, f"no candles returned for {coin} {interval}")
    seen, uniq = set(), []
    for c in rows:
        if c["t"] not in seen:
            seen.add(c["t"])
            uniq.append(c)
    df = pd.DataFrame({
        "open": [float(c["o"]) for c in uniq], "high": [float(c["h"]) for c in uniq],
        "low": [float(c["l"]) for c in uniq], "close": [float(c["c"]) for c in uniq],
        "volume": [float(c["v"]) for c in uniq],
    }, index=pd.to_datetime([int(c["t"]) for c in uniq], unit="ms", utc=True)).sort_index()
    return df

def synthetic_candles(n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 0.008, n) + 0.0001
    close = 30000 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(50, 500, n)}, index=idx)

# ── Rotation assays (Dr K directive 2026-08-25: all-tokens end-state layer) ──
# Vectorized cross-sectional sim over raw HL closes — any listed coin works
# (incl. HYPE), no Nautilus instrument definition needed. See crucible_rotation.py.
ROTATION_UNIVERSE = ["BTC", "ETH", "SOL", "XRP", "DOGE", "HYPE"]

def run_rotation_assay(interval: str, days: int, params: dict) -> dict:
    universe = list(params.get("universe", ROTATION_UNIVERSE))
    closes = {}
    for sym in universe:
        df = fetch_hl_candles(sym, interval, days)
        closes[sym] = [(str(ts), float(c)) for ts, c in df["close"].items()]
    return run_rotation(closes, params)

# ── The forge itself ─────────────────────────────────────────────────────────
def run_backtest(symbol: str, interval: str, df: pd.DataFrame, strategy_key: str, params: dict) -> dict:
    instrument = INSTRUMENTS[symbol]()
    bar_type = BarType.from_str(f"{instrument.id}-{BAR_SPEC[interval]}-LAST-EXTERNAL")
    bars = BarDataWrangler(bar_type, instrument).process(df)

    engine = BacktestEngine(config=BacktestEngineConfig(
        trader_id=TraderId("CRUCIBLE-001"),
        logging=LoggingConfig(bypass_logging=True)))
    venue = Venue("BINANCE")
    engine.add_venue(venue=venue, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
                     base_currency=USDT, starting_balances=[Money(START_BALANCE, USDT)])
    engine.add_instrument(instrument)
    engine.add_data(bars)
    engine.add_strategy(STRATEGIES[strategy_key]["make"](instrument.id, bar_type, params))
    engine.run()

    positions = engine.trader.generate_positions_report()
    account = engine.trader.generate_account_report(venue)
    assay = _assay(positions, account, len(bars), df)
    engine.dispose()
    return assay

def _money(v) -> float:
    try:
        return float(str(v).split(" ")[0].replace(",", ""))
    except Exception:
        return 0.0

def _assay(positions: pd.DataFrame, account: pd.DataFrame, bar_count: int, df: pd.DataFrame) -> dict:
    closed = positions[positions["ts_closed"].notna()] if len(positions) else positions
    pnls = [_money(x) for x in closed["realized_pnl"]] if len(closed) else []
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win, gross_loss = sum(wins), abs(sum(losses))
    commissions = 0.0
    if len(closed):
        for row in closed["commissions"]:
            items = row if isinstance(row, (list, tuple)) else [row]
            commissions += sum(abs(_money(i)) for i in items)
    equity = [ _money(v) for v in account["total"] ] if len(account) else [START_BALANCE]
    peak, max_dd = equity[0], 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            max_dd = max(max_dd, (peak - e) / peak)
    final = equity[-1] if equity else START_BALANCE
    return {
        "bars": bar_count,
        "periodStart": str(df.index[0]), "periodEnd": str(df.index[-1]),
        "trades": len(pnls), "wins": len(wins), "losses": len(losses),
        "winRatePct": round(100 * len(wins) / len(pnls), 2) if pnls else None,
        "profitFactor": round(gross_win / gross_loss, 3) if gross_loss > 0 else None,
        "netPnl": round(sum(pnls), 2),
        "grossWin": round(gross_win, 2), "grossLoss": round(-gross_loss, 2),
        "avgWin": round(gross_win / len(wins), 2) if wins else None,
        "avgLoss": round(-gross_loss / len(losses), 2) if losses else None,
        "commissionsPaid": round(commissions, 2),
        "startBalance": START_BALANCE, "finalBalance": round(final, 2),
        "returnPct": round(100 * (final - START_BALANCE) / START_BALANCE, 3),
        "maxDrawdownPct": round(100 * max_dd, 3),
        "feeModelNote": FEE_NOTE,
    }

# ── Results archive ──────────────────────────────────────────────────────────
def _load_results() -> list:
    try:
        if os.path.exists(RESULTS_FILE):
            return json.load(open(RESULTS_FILE))
    except Exception:
        pass
    return []

def _save_result(entry: dict) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    results = _load_results()
    entry["id"] = (max((r.get("id", 0) for r in results), default=0) + 1)
    results.insert(0, entry)
    json.dump(results[:500], open(RESULTS_FILE, "w"))

# ── Boot battery (Sultan's Review 2026-08-25) ────────────────────────────────
# Standing assays that must exist in the archive. On every boot, any spec not
# yet archived is forged automatically — so a redeploy IS the invocation, and
# no POST access is required. Idempotent: matched by strategy/symbol/interval/
# days/params against the archive, so completed assays never re-run.
# The Crucible still only measures; it never trades.
BATTERY_ON = os.environ.get("BATTERY_ON", "1") != "0"
BATTERY_NOTE = "AUTO-BATTERY diversity: 8/24 trend clone test (XRP/DOGE)"
BATTERY = [
    # trade_size: XRP/DOGE are whole-coin instruments (size precision 0) — the
    # 0.1 default rounds to zero and the engine rejects the order.
    {"strategy": "crucible_trend", "symbol": "XRP", "interval": "4h", "days": 365,
     "params": {"fast": 8, "slow": 24, "trade_size": 100}},
    {"strategy": "crucible_trend", "symbol": "XRP", "interval": "4h", "days": 90,
     "params": {"fast": 8, "slow": 24, "trade_size": 100}},
    {"strategy": "crucible_trend", "symbol": "DOGE", "interval": "4h", "days": 365,
     "params": {"fast": 8, "slow": 24, "trade_size": 1000}},
    {"strategy": "crucible_trend", "symbol": "DOGE", "interval": "4h", "days": 90,
     "params": {"fast": 8, "slow": 24, "trade_size": 1000}},
    # Round 2 (Dr K-directed, 2026-08-25): DOGE lead resolution + second-sleeve
    # hunt — regime-gated reversion probes (template re-registered this commit).
    {"strategy": "crucible_trend", "symbol": "DOGE", "interval": "4h", "days": 180,
     "params": {"fast": 8, "slow": 24, "trade_size": 1000},
     "note": "DOGE 90d lead resolution"},
    {"strategy": "crucible_trend", "symbol": "DOGE", "interval": "4h", "days": 270,
     "params": {"fast": 8, "slow": 24, "trade_size": 1000},
     "note": "DOGE 90d lead resolution"},
    {"strategy": "crucible_reversion", "symbol": "BTC", "interval": "1h", "days": 90,
     "params": {}, "note": "second-sleeve hunt: re-confirm assay #36"},
    {"strategy": "crucible_reversion", "symbol": "BTC", "interval": "1h", "days": 180,
     "params": {}, "note": "second-sleeve hunt: does the 90d edge survive 180d"},
    {"strategy": "crucible_reversion", "symbol": "ETH", "interval": "1h", "days": 90,
     "params": {}, "note": "second-sleeve hunt: reversion on ETH"},
    {"strategy": "crucible_reversion", "symbol": "ETH", "interval": "1h", "days": 180,
     "params": {}, "note": "second-sleeve hunt: reversion on ETH"},
    # Round 3 (Dr K-approved 2026-08-25 evening, folded into the Aug 28 review):
    # (1) ETH reversion full-year confirmation; (2) first cross-sectional
    # rotation assays — the all-tokens end-state layer, ranked by relative
    # strength, top-N held, weekly rebalance. HYPE universe truncates the
    # aligned window to HYPE's listing history; the no-HYPE spec covers the
    # full year for comparison.
    {"strategy": "crucible_reversion", "symbol": "ETH", "interval": "1h", "days": 365,
     "params": {}, "note": "second-sleeve hunt: does ETH reversion hold at 365d"},
    {"strategy": "rotation", "symbol": "ROTATION", "interval": "1d", "days": 365,
     "params": {"universe": ROTATION_UNIVERSE, "lookback_bars": 30, "top_n": 2,
                "rebalance_bars": 7},
     "note": "rotation v1: 6-coin universe incl HYPE, 30d RS, top-2, weekly"},
    {"strategy": "rotation", "symbol": "ROTATION", "interval": "1d", "days": 365,
     "params": {"universe": ROTATION_UNIVERSE, "lookback_bars": 14, "top_n": 2,
                "rebalance_bars": 7},
     "note": "rotation v1: faster 14d RS ranking"},
    {"strategy": "rotation", "symbol": "ROTATION", "interval": "1d", "days": 365,
     "params": {"universe": ["BTC", "ETH", "SOL", "XRP", "DOGE"], "lookback_bars": 30,
                "top_n": 2, "rebalance_bars": 7},
     "note": "rotation v1: majors-only (full-year window, no HYPE truncation)"},
    # CODEX sweep round 1 (Sultan's Review 2026-09-04, Dr K-approved Sep 1):
    # card #66 Setup123 pullback entry architecture vs canonical #38, plus
    # card #63 Multi_MA golden cross via ema_cross 50/200. Entry-quality hunt.
    {"strategy": "crucible_pullback", "symbol": "BTC", "interval": "4h", "days": 365,
     "params": {}, "codexId": 66, "note": "CODEX sweep: #66 Setup123 pullback, BTC 4h full-year vs canonical #38"},
    {"strategy": "crucible_pullback", "symbol": "BTC", "interval": "4h", "days": 90,
     "params": {}, "codexId": 66, "note": "CODEX sweep: #66 pullback, BTC 4h recent-regime check"},
    {"strategy": "crucible_pullback", "symbol": "ETH", "interval": "4h", "days": 365,
     "params": {}, "codexId": 66, "note": "CODEX sweep: #66 pullback, ETH 4h — does pullback timing travel where 8/24 cross did not"},
    {"strategy": "crucible_pullback", "symbol": "BTC", "interval": "1d", "days": 365,
     "params": {}, "codexId": 66, "note": "CODEX sweep: #66 pullback, BTC 1d slow-regime probe"},
    {"strategy": "ema_cross", "symbol": "BTC", "interval": "4h", "days": 365,
     "params": {"fast": 50, "slow": 200}, "codexId": 63,
     "note": "CODEX sweep: #63 Multi_MA golden cross 50/200 on 4h (naive in/out baseline)"},
    # Regime-gated #66 (Sultan's Review 2026-09-10): ungated pullback decays
    # monotonically toward the present (BTC 1.359→0.537, ETH 1.594→0.587).
    # min_stack_sep_pct demands a REAL trend before the stack qualifies —
    # rescue question on 90d, preservation question on 365d. Ungated archived
    # assays are the controls (params {} → distinct specs, no collisions).
    {"strategy": "crucible_pullback", "symbol": "BTC", "interval": "4h", "days": 90,
     "params": {"min_stack_sep_pct": 0.5}, "codexId": 66,
     "note": "gated #66 rescue probe: BTC 90d, sep 0.5% (control: ungated 0.537)"},
    {"strategy": "crucible_pullback", "symbol": "BTC", "interval": "4h", "days": 90,
     "params": {"min_stack_sep_pct": 1.0}, "codexId": 66,
     "note": "gated #66 rescue probe: BTC 90d, sep 1.0% (dose response)"},
    {"strategy": "crucible_pullback", "symbol": "ETH", "interval": "4h", "days": 90,
     "params": {"min_stack_sep_pct": 0.5}, "codexId": 66,
     "note": "gated #66 rescue probe: ETH 90d, sep 0.5% (control: ungated 0.587)"},
    {"strategy": "crucible_pullback", "symbol": "BTC", "interval": "4h", "days": 365,
     "params": {"min_stack_sep_pct": 0.5}, "codexId": 66,
     "note": "gated #66 preservation probe: BTC 365d, sep 0.5% (control: ungated 1.359)"},
    {"strategy": "crucible_pullback", "symbol": "ETH", "interval": "4h", "days": 365,
     "params": {"min_stack_sep_pct": 0.5}, "codexId": 66,
     "note": "gated #66 preservation probe: ETH 365d, sep 0.5% (control: ungated 1.594)"},
    # CODEX template round (Dr K directive 2026-10-03: push #71/#64 through the
    # funnel, one template per review). #71 Gaussian flip entries vs canonical
    # #38 (BTC 4h 365d 1.279) and the refuted ETH clones (#33/#40 ~0.9).
    {"strategy": "crucible_gaussian", "symbol": "BTC", "interval": "4h", "days": 365,
     "params": {}, "codexId": 71, "note": "CODEX #71 Gaussian: BTC 4h full-year vs canonical #38 (1.279)"},
    {"strategy": "crucible_gaussian", "symbol": "BTC", "interval": "4h", "days": 90,
     "params": {}, "codexId": 71, "note": "CODEX #71 Gaussian: BTC 4h recent-regime check"},
    {"strategy": "crucible_gaussian", "symbol": "ETH", "interval": "4h", "days": 365,
     "params": {}, "codexId": 71, "note": "CODEX #71 Gaussian: ETH 4h — does the flip entry travel where 8/24 cross did not"},
]
_battery_state = {"pending": len(BATTERY), "ran": 0, "errors": []}

def _battery_missing() -> list:
    done = _load_results()
    missing = []
    for spec in BATTERY:
        hit = any(r.get("strategy") == spec["strategy"] and r.get("symbol") == spec["symbol"]
                  and r.get("interval") == spec["interval"] and r.get("days") == spec["days"]
                  and r.get("params") == spec["params"] for r in done)
        if not hit:
            missing.append(spec)
    return missing

def _run_battery():
    time.sleep(60)  # let the service settle; avoids hammering HL on crash loops
    for spec in _battery_missing():
        try:
            with _forge_lock:
                if spec["strategy"] == "rotation":
                    assay = run_rotation_assay(spec["interval"], spec["days"], spec["params"])
                else:
                    df = fetch_hl_candles(spec["symbol"], spec["interval"], spec["days"])
                    assay = run_backtest(spec["symbol"], spec["interval"], df,
                                         spec["strategy"], spec["params"])
                entry = {"at": datetime.now(timezone.utc).isoformat(),
                         "strategy": spec["strategy"], "symbol": spec["symbol"],
                         "interval": spec["interval"], "days": spec["days"],
                         "params": spec["params"], "note": spec.get("note", BATTERY_NOTE),
                         "codexId": spec.get("codexId"), "assay": assay}
                _save_result(entry)
            _battery_state["ran"] += 1
        except Exception as exc:  # keep going; missing specs retry on next boot
            _battery_state["errors"].append(f"{spec['symbol']} {spec['days']}d: {exc}")
        time.sleep(10)
    _battery_state["pending"] = len(_battery_missing())

@app.on_event("startup")
def _battery_startup():
    if BATTERY_ON:
        threading.Thread(target=_run_battery, daemon=True).start()

# ── API ──────────────────────────────────────────────────────────────────────
class ForgeRequest(BaseModel):
    strategy: str = Field(default="crucible_trend")
    symbol: str = Field(default="BTC")
    interval: str = Field(default="1h")
    days: int = Field(default=180, ge=7)
    params: dict = Field(default_factory=dict)
    note: str = Field(default="")
    codexId: int | None = None

@app.get("/health")
def health():
    return {"agent": "CRUCIBLE-01", "status": "LIVE", "role": "EVIDENCE-ENGINE",
            "engine": "nautilus_trader", "uptime": int((time.time() - _started) * 1000),
            "volume": os.path.isdir("/data"), "assays": len(_load_results()),
            "battery": {"on": BATTERY_ON, **_battery_state},
            "strategies": list(STRATEGIES.keys()), "symbols": list(INSTRUMENTS.keys())}

@app.get("/strategies")
def strategies():
    return {k: v["desc"] for k, v in STRATEGIES.items()}

@app.get("/results")
def results():
    return {"agent": "CRUCIBLE-01", "assays": _load_results()}

@app.post("/selftest")
def selftest():
    with _forge_lock:
        assay = run_backtest("BTC", "1h", synthetic_candles(), "ema_cross", {"fast": 12, "slow": 48})
    assay["selftest"] = True
    assay["note"] = "synthetic data — verifies the engine, not any strategy"
    return assay

@app.post("/forge")
def forge(req: ForgeRequest):
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"unknown strategy — have: {list(STRATEGIES.keys())}")
    if req.symbol not in INSTRUMENTS:
        raise HTTPException(400, f"v1 symbols: {list(INSTRUMENTS.keys())}")
    if req.interval not in INTERVAL_MS:
        raise HTTPException(400, f"intervals: {list(INTERVAL_MS.keys())}")
    days = min(req.days, MAX_DAYS)
    if not _forge_lock.acquire(blocking=False):
        raise HTTPException(429, "forge is busy — one assay at a time")
    try:
        df = fetch_hl_candles(req.symbol, req.interval, days)
        assay = run_backtest(req.symbol, req.interval, df, req.strategy, req.params)
        entry = {"at": datetime.now(timezone.utc).isoformat(),
                 "strategy": req.strategy, "symbol": req.symbol, "interval": req.interval,
                 "days": days, "params": req.params, "note": req.note,
                 "codexId": req.codexId, "assay": assay}
        _save_result(entry)
        return entry
    finally:
        _forge_lock.release()

class RotationRequest(BaseModel):
    interval: str = Field(default="1d")
    days: int = Field(default=365, ge=30)
    params: dict = Field(default_factory=dict)
    note: str = Field(default="")

@app.post("/forge_rotation")
def forge_rotation(req: RotationRequest):
    if req.interval not in INTERVAL_MS:
        raise HTTPException(400, f"intervals: {list(INTERVAL_MS.keys())}")
    days = min(req.days, MAX_DAYS)
    if not _forge_lock.acquire(blocking=False):
        raise HTTPException(429, "forge is busy — one assay at a time")
    try:
        assay = run_rotation_assay(req.interval, days, req.params)
        entry = {"at": datetime.now(timezone.utc).isoformat(),
                 "strategy": "rotation", "symbol": "ROTATION", "interval": req.interval,
                 "days": days, "params": req.params, "note": req.note,
                 "codexId": None, "assay": assay}
        _save_result(entry)
        return entry
    finally:
        _forge_lock.release()

@app.get("/")
def index():
    page = os.path.join(os.path.dirname(__file__), "..", "public", "index.html")
    return FileResponse(page)
