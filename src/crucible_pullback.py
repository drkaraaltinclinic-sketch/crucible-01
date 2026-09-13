"""
CruciblePullback — CODEX card #66 (Setup123) entry architecture:
EMA-stack trend filter with pullback-and-resume timing, then the exact
CrucibleTrend management (ATR stop, R-multiple target, time stop, bar-close
evaluation). Purpose: isolate ENTRY QUALITY against the canonical
crucible_trend assays (#38 family) — same management, different trigger.

Long setup:  fast > mid > slow (stacked uptrend), price pulls back to close
below the fast EMA while the stack holds, then closes back above the fast
EMA → enter long on that resume bar. Shorts mirror. One position at a time.

Regime gate (Sultan's Review 2026-09-10): min_stack_sep_pct requires the
fast/slow EMA separation to exceed a threshold (percent of close) before a
stack counts as a trend — the exact mirror of crucible_reversion's band
gate, pointed the other way. Ungated assays showed monotonic PF decay
toward the present (BTC 365d 1.359 → 90d 0.537); the gate asks whether
demanding a REAL trend rescues the recent window without killing the
full-year edge. Default 0.0 = gate off = bit-identical to the ungated
strategy, so existing archived assays remain valid comparators.
"""

from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators import AverageTrueRange, ExponentialMovingAverage
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy


class CruciblePullbackConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal = Decimal("0.1")
    fast_ema_period: int = 8
    mid_ema_period: int = 21
    slow_ema_period: int = 50
    atr_period: int = 14
    atr_stop_mult: float = 1.5
    target_r: float = 2.0
    max_hold_bars: int = 72
    allow_shorts: bool = True
    min_stack_sep_pct: float = 0.0  # 0.0 = gate off (bit-identical to ungated)


class CruciblePullback(Strategy):
    def __init__(self, config: CruciblePullbackConfig) -> None:
        super().__init__(config)
        self.fast = ExponentialMovingAverage(config.fast_ema_period)
        self.mid = ExponentialMovingAverage(config.mid_ema_period)
        self.slow = ExponentialMovingAverage(config.slow_ema_period)
        self.atr = AverageTrueRange(config.atr_period)
        self.instrument = None
        self.pulled_back = 0  # +1 pullback seen in uptrend, -1 in downtrend
        self._reset_trade()

    def _reset_trade(self) -> None:
        self.entry_px = None
        self.stop_px = None
        self.target_px = None
        self.direction = 0
        self.bars_held = 0

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"instrument not found: {self.config.instrument_id}")
            self.stop()
            return
        for ind in (self.fast, self.mid, self.slow, self.atr):
            self.register_indicator_for_bars(self.config.bar_type, ind)
        self.subscribe_bars(self.config.bar_type)

    def _enter(self, side: OrderSide, close: float, atr: float) -> None:
        qty = self.instrument.make_qty(self.config.trade_size)
        self.submit_order(self.order_factory.market(
            instrument_id=self.config.instrument_id, order_side=side, quantity=qty))
        risk = self.config.atr_stop_mult * atr
        self.direction = 1 if side == OrderSide.BUY else -1
        self.entry_px = close
        self.stop_px = close - self.direction * risk
        self.target_px = close + self.direction * risk * self.config.target_r
        self.bars_held = 0
        self.pulled_back = 0

    def _exit(self) -> None:
        self.close_all_positions(self.config.instrument_id)
        self._reset_trade()

    def on_bar(self, bar: Bar) -> None:
        if not (self.fast.initialized and self.mid.initialized
                and self.slow.initialized and self.atr.initialized):
            return
        close = float(bar.close)
        flat = self.portfolio.is_flat(self.config.instrument_id)

        if not flat and self.direction != 0:
            self.bars_held += 1
            hit_stop = (close <= self.stop_px) if self.direction > 0 else (close >= self.stop_px)
            hit_target = (close >= self.target_px) if self.direction > 0 else (close <= self.target_px)
            if hit_stop or hit_target or self.bars_held >= self.config.max_hold_bars:
                self._exit()
            return

        if flat:
            self._reset_trade()
            atr = float(self.atr.value)
            if atr <= 0:
                return
            up_stack = self.fast.value > self.mid.value > self.slow.value
            dn_stack = self.fast.value < self.mid.value < self.slow.value
            if self.config.min_stack_sep_pct > 0.0 and close > 0:
                # separation of the outer EMAs as a percent of price — the
                # reversion band gate mirrored: reversion trades when this is
                # SMALL (ranging), the pullback gate demands it be LARGE
                # (established trend) before a stack qualifies.
                sep_pct = abs(self.fast.value - self.slow.value) / close * 100.0
                if sep_pct < self.config.min_stack_sep_pct:
                    up_stack = False
                    dn_stack = False
            if up_stack:
                if close < self.fast.value:
                    self.pulled_back = 1  # pullback registered; stack intact
                elif self.pulled_back == 1 and close > self.fast.value:
                    self._enter(OrderSide.BUY, close, atr)
            elif dn_stack and self.config.allow_shorts:
                if close > self.fast.value:
                    self.pulled_back = -1
                elif self.pulled_back == -1 and close < self.fast.value:
                    self._enter(OrderSide.SELL, close, atr)
            else:
                self.pulled_back = 0  # stack broken — stale pullbacks don't carry

    def on_stop(self) -> None:
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)

    def on_reset(self) -> None:
        for ind in (self.fast, self.mid, self.slow, self.atr):
            ind.reset()
        self.pulled_back = 0
        self._reset_trade()
