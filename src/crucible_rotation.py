"""
CrucibleRotation — cross-sectional relative-strength rotation assay
(Dr K directive 2026-08-25: the all-tokens end-state layer).

WHAT IT MEASURES: instead of asking "does symbol X trend?", it asks "does
holding the STRONGEST symbols of a universe beat holding any one of them?"
Rank the universe by lookback return (relative strength), hold the top-N
equally weighted, rebalance every REBALANCE_BARS bars. Optional absolute-
momentum filter: a symbol only earns a slot if its own lookback return is
positive — otherwise that slot sits in cash (the classic dual-momentum
guard against 2022-style tapes).

WHY NOT NAUTILUS: rotation is a portfolio-level, close-to-close weights
problem, not an order-mechanics problem. A vectorized simulator on the same
Hyperliquid candles is the honest tool: no fill mechanics to fake, fees
charged on turnover at taker rate. Metrics are labeled accordingly —
"trades" here are per-symbol holding periods, so PF is comparable in
spirit (gross winning periods / gross losing periods) but not in mechanics
to the Nautilus templates. The Crucible still only measures; it never trades.

Fee model: FEE_BPS per side on traded notional (default 4.0 bps = Binance
taker 0.04%, conservative vs Hyperliquid 0.035%), charged on |Δweight|.
"""

from __future__ import annotations


def run_rotation(closes: dict, params: dict) -> dict:
    """closes: {symbol: list[(ts_ms, close)]} — raw candle closes per symbol.
    Returns an assay dict in the archive's shape (plus rotation extras)."""
    lookback = int(params.get("lookback_bars", 30))
    top_n = int(params.get("top_n", 2))
    rebalance = int(params.get("rebalance_bars", 7))
    fee_bps = float(params.get("fee_bps", 4.0))
    abs_filter = bool(params.get("abs_momentum_filter", True))

    # ── Align on the intersection of timestamps (symbols list at different times)
    common = None
    for sym, rows in closes.items():
        ts = {t for t, _ in rows}
        common = ts if common is None else (common & ts)
    if not common:
        raise ValueError("no overlapping candles across universe")
    axis = sorted(common)
    px = {sym: {t: c for t, c in rows} for sym, rows in closes.items()}
    series = {sym: [px[sym][t] for t in axis] for sym in closes}
    syms = sorted(closes.keys())
    n_bars = len(axis)
    if n_bars <= lookback + rebalance:
        raise ValueError(f"only {n_bars} aligned bars — need > lookback+rebalance "
                         f"({lookback}+{rebalance}); shorten lookback or extend days")

    # ── Walk forward: weights set at each rebalance close, P&L accrued bar to bar
    weights = {s: 0.0 for s in syms}
    equity, peak, max_dd = 1.0, 1.0, 0.0
    fees_paid = 0.0
    curve = [1.0]
    open_period = {}   # sym -> {"weight", "entry_px", "entered_at"}
    trades = []        # closed per-symbol holding periods

    def _close_period(sym, i, reason):
        info = open_period.pop(sym, None)
        if info is None:
            return
        ret = series[sym][i] / info["entry_px"] - 1.0
        pnl_frac = info["weight"] * ret
        trades.append({"symbol": sym, "pnlFrac": pnl_frac, "ret": ret,
                       "enteredAt": info["entered_at"], "exitedAt": axis[i],
                       "reason": reason})

    for i in range(lookback, n_bars):
        # accrue returns for held symbols over this bar
        if i > lookback:
            bar_ret = sum(weights[s] * (series[s][i] / series[s][i - 1] - 1.0)
                          for s in syms if weights[s] > 0)
            equity *= (1.0 + bar_ret)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
        curve.append(equity)

        # rebalance at close of every REBALANCE_BARS-th bar (and at the start)
        if (i - lookback) % rebalance == 0 and i < n_bars - 1:
            mom = {s: series[s][i] / series[s][i - lookback] - 1.0 for s in syms}
            ranked = sorted(syms, key=lambda s: mom[s], reverse=True)
            chosen = [s for s in ranked[:top_n] if (mom[s] > 0 or not abs_filter)]
            target = {s: (1.0 / top_n if s in chosen else 0.0) for s in syms}
            turnover = sum(abs(target[s] - weights[s]) for s in syms)
            fee = turnover * (fee_bps / 10_000.0)
            fees_paid += fee * equity
            equity *= (1.0 - fee)
            for s in syms:
                if weights[s] > 0 and target[s] == 0:
                    _close_period(s, i, "ROTATED_OUT")
                if target[s] > 0 and s not in open_period:
                    open_period[s] = {"weight": target[s], "entry_px": series[s][i],
                                      "entered_at": axis[i]}
            weights = target

    for s in list(open_period):
        _close_period(s, n_bars - 1, "END_OF_DATA")

    pnls = [t["pnlFrac"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gw, gl = sum(wins), abs(sum(losses))
    return {
        "engine": "rotation-vectorized",
        "bars": n_bars, "alignedSymbols": syms,
        "periodStart": str(axis[0]), "periodEnd": str(axis[-1]),
        "trades": len(pnls), "wins": len(wins), "losses": len(losses),
        "winRatePct": round(100 * len(wins) / len(pnls), 2) if pnls else None,
        "profitFactor": round(gw / gl, 3) if gl > 0 else None,
        "netPnl": round(sum(pnls) * 100, 3),  # % of start equity, weight-scaled
        "grossWin": round(gw * 100, 3), "grossLoss": round(-gl * 100, 3),
        "returnPct": round(100 * (equity - 1.0), 3),
        "maxDrawdownPct": round(100 * max_dd, 3),
        "feesPaidPctEquity": round(100 * fees_paid, 4),
        "params": {"lookback_bars": lookback, "top_n": top_n,
                   "rebalance_bars": rebalance, "fee_bps": fee_bps,
                   "abs_momentum_filter": abs_filter},
        "holdingPeriods": trades[-40:],
        "tradeNote": ("'trades' are per-symbol holding periods between rebalances; "
                      "PF = gross winning periods / gross losing periods. "
                      "Vectorized close-to-close sim, fees on turnover — NOT a "
                      "Nautilus order-mechanics assay; compare rotation assays "
                      "with each other, single-symbol templates with each other."),
    }
