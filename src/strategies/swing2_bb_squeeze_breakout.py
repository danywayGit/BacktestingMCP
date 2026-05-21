"""
SWING2 — Bollinger Band Squeeze Breakout Strategy
==================================================
Timeframe:  4H
Direction:  Long and Short
Target R:R: 1:2.5

Entry Logic:
  1. BB width has recently been at a local minimum (squeeze)
  2. Long:  close breaks above upper BB AND MACD line > Signal line
  3. Short: close breaks below lower BB AND MACD line < Signal line

Stop Loss:  ATR(14) × atr_stop_mult (default 2.5×)
Take Profit: stop_distance × rr_ratio (default 1:2.5)
Sizing:     risk_pct % of equity per trade
"""

import numpy as np
import pandas as pd

from ..core.backtesting_engine import BaseStrategy

try:
    import talib
    def calculate_atr(high, low, close, period):
        return pd.Series(talib.ATR(pd.Series(high), pd.Series(low), pd.Series(close), timeperiod=period)).ffill().bfill().fillna(0)
except ImportError:
    import ta
    def calculate_atr(high, low, close, period):
        return ta.volatility.average_true_range(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).ffill().bfill().fillna(0)


def _bb_upper(close, length, mult):
    s = pd.Series(close)
    basis = s.rolling(length, min_periods=length).mean()
    dev = s.rolling(length, min_periods=length).std()
    return (basis + mult * dev).ffill().bfill()


def _bb_lower(close, length, mult):
    s = pd.Series(close)
    basis = s.rolling(length, min_periods=length).mean()
    dev = s.rolling(length, min_periods=length).std()
    return (basis - mult * dev).ffill().bfill()


def _bb_width(close, length, mult):
    s = pd.Series(close)
    basis = s.rolling(length, min_periods=length).mean()
    dev = s.rolling(length, min_periods=length).std()
    upper = basis + mult * dev
    lower = basis - mult * dev
    width = (upper - lower) / basis.replace(0, np.nan)
    return width.ffill().bfill().fillna(0)


def _macd_line(close, fast, slow, signal):
    try:
        import talib
        m, _, _ = talib.MACD(pd.Series(close), fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return pd.Series(m).fillna(0)
    except ImportError:
        import ta
        return ta.trend.MACD(pd.Series(close), window_fast=fast, window_slow=slow, window_sign=signal).macd().fillna(0)


def _macd_signal_line(close, fast, slow, signal):
    try:
        import talib
        _, s, _ = talib.MACD(pd.Series(close), fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return pd.Series(s).fillna(0)
    except ImportError:
        import ta
        return ta.trend.MACD(pd.Series(close), window_fast=fast, window_slow=slow, window_sign=signal).macd_signal().fillna(0)


class Swing2BBSqueezeBreakoutStrategy(BaseStrategy):
    """
    Bollinger Band squeeze breakout strategy.
    Enters on a volatility expansion after a squeeze, confirmed by MACD direction.
    """

    # === Parameters (all auto-discoverable by BacktestingMCP optimizer) ===
    bb_length     = 20
    bb_mult       = 2.0
    squeeze_bars  = 5      # look back N bars for squeeze detection
    macd_fast     = 12
    macd_slow     = 26
    macd_signal   = 9
    atr_period    = 14
    atr_stop_mult = 2.5
    rr_ratio      = 2.5
    risk_pct      = 1.0

    def init(self):
        self.upper  = self.I(_bb_upper,        self.data.Close, self.bb_length, self.bb_mult)
        self.lower  = self.I(_bb_lower,        self.data.Close, self.bb_length, self.bb_mult)
        self.bb_w   = self.I(_bb_width,        self.data.Close, self.bb_length, self.bb_mult)
        self.macd_l = self.I(_macd_line,       self.data.Close, self.macd_fast, self.macd_slow, self.macd_signal)
        self.macd_s = self.I(_macd_signal_line, self.data.Close, self.macd_fast, self.macd_slow, self.macd_signal)
        self.atr    = self.I(calculate_atr,    self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def _was_squeezed(self):
        """True if BB width hit a local minimum within the last squeeze_bars bars."""
        n = self.squeeze_bars + 1
        if len(self.bb_w) < n + 1:
            return False
        # width values: index -1 = current, -2..-N-1 = past N bars
        widths = [float(self.bb_w[-(i + 1)]) for i in range(n + 1)]
        current_min = min(widths[1:])   # min of past N bars excluding current
        return min(widths) <= current_min

    def next(self):
        if not self.should_trade():
            return

        close  = float(self.data.Close[-1])
        upper  = float(self.upper[-1])
        lower  = float(self.lower[-1])
        macd_l = float(self.macd_l[-1])
        macd_s = float(self.macd_s[-1])
        atr    = float(self.atr[-1])

        if not all(np.isfinite([close, upper, lower, macd_l, macd_s, atr])) or atr <= 0:
            return

        was_squeezed = self._was_squeezed()
        stop_dist = atr * self.atr_stop_mult
        risk_amt  = self.equity * self.risk_pct / 100.0
        qty       = risk_amt / stop_dist if stop_dist > 0 else 0

        if not self.position and was_squeezed and qty > 0:
            if close > upper and macd_l > macd_s:
                self.enter_long_position(
                    stop_loss   = close - stop_dist,
                    take_profit = close + stop_dist * self.rr_ratio,
                )
            elif close < lower and macd_l < macd_s:
                self.enter_short_position(
                    stop_loss   = close + stop_dist,
                    take_profit = close - stop_dist * self.rr_ratio,
                )
