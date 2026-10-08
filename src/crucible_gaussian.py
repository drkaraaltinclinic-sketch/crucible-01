"""
CrucibleGaussian — CODEX card #71 (Gaussian Filter Trend [QuantAlgo]) entry
architecture: Ehlers N-pole Gaussian filter basis with an ER-adaptive ATR
deadband and a ratcheting trend line. Entries on trend-direction FLIPS only
(bull flip -> long, bear flip -> short), then standard CrucibleTrend
management at bar close: ATR_STOP_MULT x ATR stop, TARGET_R target, time stop.
The card is indicator-only; stops/targets here are the network's canon.
"""

import math
from collections import deque
from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators import AverageTrueRange
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy


class CrucibleGaussianConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal = Decimal("0.1")
    gaussian_length: int = 14
    gaussian_poles: int = 4
    atr_period: int = 14
    fixed_multiplier: float = 1.5
    adaptive_width: bool = True
    trend_multiplier: float = 0.8
    chop_multiplier: float = 2.5
    efficiency_length: int = 10
    efficiency_smooth: int = 5
    atr_stop_mult: float = 1.5
    target_r: float = 2.0
    max_hold_bars: int = 72
    allow_shorts: bool = True


class CrucibleGaussian(Strategy):
    def __init__(self, config: CrucibleGaussianConfig) -> None:
        super().__init__(config)
        self.atr = AverageTrueRange(config.atr_period)
        beta = (1.0 - math.cos(2.0 * math.pi / config.gaussian_length)) / (
            pow(2.0, 1.0 / config.gaussian_poles) - 1.0)
        self.alpha = -beta + math.sqrt(beta * beta + 2.0 * beta)
        self.poles = [None] * config.gaussian_poles
        self.closes = deque(maxlen=config.efficiency_length + 1)
        self.er_window = deque(maxlen=config.efficiency_smooth)
        self.bar_count = 0
        self.warmup = max(3 * config.gaussian_length,
                          config.efficiency_length + config.efficiency_smooth,
                          config.atr_period) + 1
        self.trend_dir = 0   # ratchet state: +1 / -1 / 0 (unseeded)
        self.line = None     # ratcheting trend line
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
        self.register_indicator_for_bars(self.config.bar_type, self.atr)
        self.subscribe_bars(self.config.bar_type)

    def _gaussian(self, close: float) -> float:
        x = close
        for i in range(len(self.poles)):
            prev = self.poles[i] if self.poles[i] is not None else x
            self.poles[i] = self.alpha * x + (1.0 - self.alpha) * prev
            x = self.poles[i]
        return x

    def _efficiency(self) -> float:
        if len(self.closes) < self.closes.maxlen:
            return 0.0
        pts = list(self.closes)
        change = abs(pts[-1] - pts[0])
        volatility = sum(abs(pts[i] - pts[i - 1]) for i in range(1, len(pts)))
        er = (change / volatility) if volatility > 0 else 0.0
        self.er_window.append(er)
        return sum(self.er_window) / len(self.er_window)

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
        close = float(bar.close)
        self.bar_count += 1
        self.closes.append(close)
        basis = self._gaussian(close)
        er = self._efficiency()
        if not self.atr.initialized or self.bar_count < self.warmup:
            return
        atr = float(self.atr.value)
        if atr <= 0:
            return
        c = self.config
        mult = (c.chop_multiplier + (c.trend_multiplier - c.chop_multiplier) * er
                if c.adaptive_width else c.fixed_multiplier)
        width = atr * mult
        prev_dir = self.trend_dir
        if self.trend_dir == 0:  # seed the ratchet, no flip counted
            self.trend_dir = 1 if close >= basis else -1
            self.line = basis - width if self.trend_dir > 0 else basis + width
        elif self.trend_dir > 0:
            self.line = max(self.line, basis - width)
            if close < self.line:
                self.trend_dir, self.line = -1, basis + width
        else:
            self.line = min(self.line, basis + width)
            if close > self.line:
                self.trend_dir, self.line = 1, basis - width
        flipped = prev_dir != 0 and self.trend_dir != prev_dir

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
            if flipped and self.trend_dir > 0:
                self._enter(OrderSide.BUY, close, atr)
            elif flipped and self.trend_dir < 0 and c.allow_shorts:
                self._enter(OrderSide.SELL, close, atr)

    def on_stop(self) -> None:
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)

    def on_reset(self) -> None:
        self.atr.reset()
        self.poles = [None] * self.config.gaussian_poles
        self.closes.clear()
        self.er_window.clear()
        self.bar_count = 0
        self.trend_dir = 0
        self.line = None
        self._reset_trade()
