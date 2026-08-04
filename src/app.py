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
INSTRUMENTS = {
    "BTC": TestInstrumentProvider.btcusdt_perp_binance,
    "ETH": TestInstrumentProvider.ethusdt_perp_binance,
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

STRATEGIES = {
    "ema_cross": {"make": _mk_ema_cross,
                  "desc": "Canonical EMA cross, market in/out. Params: fast, slow, trade_size."},
    "crucible_trend": {"make": _mk_crucible_trend,
                       "desc": "The forge's native template — SUPREME-LEADER mechanics: EMA-cross entries (both sides), ATR stop, R-multiple target, time stop, bar-close management. Params: fast, slow, atr_period, atr_stop_mult, target_r, max_hold_bars, allow_shorts, trade_size."},
    "crucible_reversion": {"make": _mk_crucible_reversion,
                           "desc": "Regime-gated mean reversion (CODEX card #4 style): trades only when |fastEMA-slowEMA|/close*100 < regime_band_pct; RSI<=rsi_buy buys fear, RSI>=rsi_sell fades greed; management identical to crucible_trend. Params: fast, slow, regime_band_pct, rsi_period, rsi_buy, rsi_sell, atr_period, atr_stop_mult, target_r, max_hold_bars, allow_shorts, trade_size."},
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

def _run_forge_now(req: "ForgeRequest") -> dict:
    """Shared forge body — validates, runs, archives. Caller holds no lock."""
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"unknown strategy — have: {list(STRATEGIES.keys())}")
    if req.symbol not in INSTRUMENTS:
        raise HTTPException(400, f"v1 symbols: {list(INSTRUMENTS.keys())}")
    if req.interval not in INTERVAL_MS:
        raise HTTPException(400, f"intervals: {list(INTERVAL_MS.keys())}")
    days = min(req.days, MAX_DAYS)
    with _forge_lock:
        df = fetch_hl_candles(req.symbol, req.interval, days)
        assay = run_backtest(req.symbol, req.interval, df, req.strategy, req.params)
        entry = {"at": datetime.now(timezone.utc).isoformat(),
                 "strategy": req.strategy, "symbol": req.symbol, "interval": req.interval,
                 "days": days, "params": req.params, "note": req.note,
                 "codexId": req.codexId, "assay": assay}
        _save_result(entry)
        return entry


_queue_depth = 0
_queue_guard = threading.Lock()
MAX_QUEUE = 12


@app.get("/forge/trigger")
def forge_trigger(strategy: str = "crucible_trend", symbol: str = "BTC",
                  interval: str = "1h", days: int = 180, note: str = "",
                  codexId: int | None = None,
                  fast: int | None = None, slow: int | None = None,
                  atr_period: int | None = None, atr_stop_mult: float | None = None,
                  target_r: float | None = None, max_hold_bars: int | None = None,
                  allow_shorts: int | None = None, trade_size: float | None = None,
                  regime_band_pct: float | None = None, rsi_period: int | None = None,
                  rsi_buy: float | None = None, rsi_sell: float | None = None):
    """GET trigger for environments that cannot POST (e.g. the Sultan's Review
    fetch channel). Queues the assay on a background thread and returns
    immediately; the result lands in /results when the forge cools."""
    global _queue_depth
    params = {k: v for k, v in {
        "fast": fast, "slow": slow, "atr_period": atr_period,
        "atr_stop_mult": atr_stop_mult, "target_r": target_r,
        "max_hold_bars": max_hold_bars, "trade_size": trade_size,
        "regime_band_pct": regime_band_pct, "rsi_period": rsi_period,
        "rsi_buy": rsi_buy, "rsi_sell": rsi_sell,
    }.items() if v is not None}
    if allow_shorts is not None:
        params["allow_shorts"] = bool(allow_shorts)
    req = ForgeRequest(strategy=strategy, symbol=symbol, interval=interval,
                       days=days, params=params, note=note or "get-trigger",
                       codexId=codexId)
    # cheap validation before queueing so bad specs fail loudly at trigger time
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"unknown strategy — have: {list(STRATEGIES.keys())}")
    if req.symbol not in INSTRUMENTS:
        raise HTTPException(400, f"v1 symbols: {list(INSTRUMENTS.keys())}")
    if req.interval not in INTERVAL_MS:
        raise HTTPException(400, f"intervals: {list(INTERVAL_MS.keys())}")
    with _queue_guard:
        if _queue_depth >= MAX_QUEUE:
            raise HTTPException(429, f"forge queue full ({MAX_QUEUE})")
        _queue_depth += 1

    def _work():
        global _queue_depth
        try:
            _run_forge_now(req)
        except Exception as e:  # archive failures too — invisible errors are worse
            _save_result({"at": datetime.now(timezone.utc).isoformat(),
                          "strategy": req.strategy, "symbol": req.symbol,
                          "interval": req.interval, "days": req.days,
                          "params": req.params, "note": req.note,
                          "codexId": req.codexId,
                          "assay": None, "error": str(e)[:300]})
        finally:
            with _queue_guard:
                _queue_depth -= 1

    threading.Thread(target=_work, daemon=True).start()
    return {"queued": True, "strategy": req.strategy, "symbol": req.symbol,
            "interval": req.interval, "days": min(req.days, MAX_DAYS),
            "params": params, "queueDepth": _queue_depth}


@app.get("/forge/queue")
def forge_queue():
    return {"queueDepth": _queue_depth, "busy": _forge_lock.locked()}


@app.get("/")
def index():
    page = os.path.join(os.path.dirname(__file__), "..", "public", "index.html")
    return FileResponse(page)
