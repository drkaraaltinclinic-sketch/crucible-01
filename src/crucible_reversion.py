"""
CrucibleReversion — regime-gated mean reversion (CODEX card #4 style),
the redesign track's non-trend candidate entry architecture.

REGIME GATE: trade only when the market is ranging — |fast EMA − slow EMA|
as a fraction of price must be inside REGIME_BAND_PCT. Trending tape is
untouchable for this template; that is the whole point of the gate.

ENTRY (only inside the gate): RSI ≤ rsi_buy → LONG (buy fear);
RSI ≥ rsi_sell → SHORT (fade greed, optional).

MANAGEMENT: identical to CrucibleTrend / SUPREME-LEADER live mechanics —
initial stop = ATR_STOP_MULT x ATR, target = TARGET_R x risk, time stop
after MAX_HOLD_BARS, evaluated at bar close, market orders only. The exits
are deliberately shared: assays isolate ENTRY quality, which is the
network's diagnosed disease.
"""

from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators import (
    AverageTrueRange,
    ExponentialMovingAverage,
    RelativeStrengthIndex,
)
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy


class CrucibleReversionConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal = Decimal("0.1")
    fast_ema_period: int = 12
    slow_ema_period: int = 48
    regime_band_pct: float = 0.75  # |fast-slow|/close*100 must be < this to trade
    rsi_period: int = 14
    rsi_buy: float = 30.0
    rsi_sell: float = 70.0
    atr_period: int = 14
    atr_stop_mult: float = 1.5
    target_r: float = 2.0
    max_hold_bars: int = 72
    allow_shorts: bool = True


class CrucibleReversion(Strategy):
    def __init__(self, config: CrucibleReversionConfig) -> None:
        super().__init__(config)
        self.fast = ExponentialMovingAverage(config.fast_ema_period)
        self.slow = ExponentialMovingAverage(config.slow_ema_period)
        self.rsi = RelativeStrengthIndex(config.rsi_period)
        self.atr = AverageTrueRange(config.atr_period)
        self.instrument = None
        self._reset_trade()

    def _reset_trade(self) -> None:
        self.entry_px = None
        self.stop_px = None
        self.target_px = None
        self.direction = 0  # +1 long, -1 short, 0 flat
        self.bars_held = 0

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"instrument not found: {self.config.instrument_id}")
            self.stop()
            return
        for ind in (self.fast, self.slow, self.rsi, self.atr):
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

    def _exit(self) -> None:
        self.close_all_positions(self.config.instrument_id)
        self._reset_trade()

    def on_bar(self, bar: Bar) -> None:
        if not (self.fast.initialized and self.slow.initialized
                and self.rsi.initialized and self.atr.initialized):
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
            if atr <= 0 or close <= 0:
                return
            band = abs(self.fast.value - self.slow.value) / close * 100.0
            if band >= self.config.regime_band_pct:
                return  # trending tape — the gate keeps us out
            rsi = float(self.rsi.value)
            # NautilusTrader RSI is normalized 0..1; accept 0..100-style config.
            rsi_scaled = rsi * 100.0 if rsi <= 1.0 else rsi
            if rsi_scaled <= self.config.rsi_buy:
                self._enter(OrderSide.BUY, close, atr)
            elif self.config.allow_shorts and rsi_scaled >= self.config.rsi_sell:
                self._enter(OrderSide.SELL, close, atr)

    def on_stop(self) -> None:
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)

    def on_reset(self) -> None:
        self.fast.reset()
        self.slow.reset()
        self.rsi.reset()
        self.atr.reset()
        self._reset_trade()
