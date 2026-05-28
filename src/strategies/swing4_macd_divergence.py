"""
SWING4 — MACD Histogram Divergence Reversal Strategy
=====================================================
Timeframe:  2H
Direction:  Long and Short
Target R:R: 1:2

Entry Logic:
  Bullish divergence:
    - Price makes a new low (below lowest low of past divergence_lookback bars)
    - MACD histogram does NOT confirm the new low (histogram above its recent lowest)
    - RSI(14) < rsi_long_max (still in weak territory)

  Bearish divergence:
    - Price makes a new high (above highest high of past divergence_lookback bars)
    - MACD histogram does NOT confirm the new high (histogram below its recent highest)
    - RSI(14) > rsi_short_min (still in strong territory)

Stop Loss:  ATR(14) × atr_stop_mult
Take Profit: stop_distance × rr_ratio (default 1:2)
Sizing:     risk_pct % of equity per trade
"""

import numpy as np
import pandas as pd

from ..core.backtesting_engine import BaseStrategy

try:
    import talib
    def calculate_rsi(close, period):
        return pd.Series(talib.RSI(pd.Series(close), timeperiod=period)).ffill().bfill().fillna(50)
    def calculate_atr(high, low, close, period):
        return pd.Series(talib.ATR(pd.Series(high), pd.Series(low), pd.Series(close), timeperiod=period)).ffill().bfill().fillna(0)
except ImportError:
    import ta
    def calculate_rsi(close, period):
        return ta.momentum.rsi(pd.Series(close), window=period).ffill().bfill().fillna(50)
    def calculate_atr(high, low, close, period):
        return ta.volatility.average_true_range(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).ffill().bfill().fillna(0)


def _macd_histogram(close, fast, slow, signal):
    try:
        import talib
        _, _, h = talib.MACD(pd.Series(close), fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return pd.Series(h).fillna(0)
    except ImportError:
        import ta
        obj = ta.trend.MACD(pd.Series(close), window_fast=fast, window_slow=slow, window_sign=signal)
        return obj.macd_diff().fillna(0)


class Swing4MACDDivergenceStrategy(BaseStrategy):
    """
    MACD histogram divergence reversal strategy.
    Enters counter-trend when price and MACD histogram diverge, confirmed by RSI.
    """

    # === Parameters ===
    macd_fast           = 12
    macd_slow           = 26
    macd_signal_period  = 9
    rsi_period          = 14
    divergence_lookback = 5     # bars to look back for price high/low
    rsi_long_max        = 45    # RSI must be below this for long entry
    rsi_short_min       = 55    # RSI must be above this for short entry
    atr_period          = 14
    atr_stop_mult       = 2.0
    rr_ratio            = 2.0
    risk_pct            = 1.0

    def init(self):
        self.histogram = self.I(_macd_histogram, self.data.Close,
                                self.macd_fast, self.macd_slow, self.macd_signal_period)
        self.rsi       = self.I(calculate_rsi, self.data.Close, self.rsi_period)
        self.atr       = self.I(calculate_atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        if not self.should_trade():
            return

        n = self.divergence_lookback
        if len(self.data.Close) < n + 2:
            return

        close    = float(self.data.Close[-1])
        rsi      = float(self.rsi[-1])
        atr      = float(self.atr[-1])
        hist_cur = float(self.histogram[-1])

        if not all(np.isfinite([close, rsi, atr, hist_cur])) or atr <= 0:
            return

        # Lookback slices (past N bars, not including current)
        lows_past  = [float(self.data.Low[-(i+2)])  for i in range(n)]
        highs_past = [float(self.data.High[-(i+2)]) for i in range(n)]
        hist_past  = [float(self.histogram[-(i+2)]) for i in range(n)]

        low_cur    = float(self.data.Low[-1])
        high_cur   = float(self.data.High[-1])

        # Bullish divergence: price new low, histogram higher low
        price_new_low   = low_cur  < min(lows_past)
        hist_higher_low = hist_cur > min(hist_past)
        bull_div        = price_new_low and hist_higher_low

        # Bearish divergence: price new high, histogram lower high
        price_new_high   = high_cur  > max(highs_past)
        hist_lower_high  = hist_cur  < max(hist_past)
        bear_div         = price_new_high and hist_lower_high

        stop_dist = atr * self.atr_stop_mult
        risk_amt  = self.equity * self.risk_pct / 100.0
        qty       = risk_amt / stop_dist if stop_dist > 0 else 0

        if not self.position and qty > 0:
            if bull_div and rsi < self.rsi_long_max:
                self.enter_long_position(
                    stop_loss   = close - stop_dist,
                    take_profit = close + stop_dist * self.rr_ratio,
                    atr_value   = atr,
                )
            elif bear_div and rsi > self.rsi_short_min:
                self.enter_short_position(
                    stop_loss   = close + stop_dist,
                    take_profit = close - stop_dist * self.rr_ratio,
                    atr_value   = atr,
                )
