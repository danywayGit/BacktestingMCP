"""
SWING6 — Multi-Timeframe EMA Stack Strategy
============================================
Timeframe:  30m entry / 4H bias (configurable)
Direction:  Long and Short
Target R:R: 1:2.5

Entry Logic:
  HTF Bias: resample OHLCV to htf_multiplier × current bar resolution
    - Bull bias: HTF close > HTF EMA(htf_ema_period)
    - Bear bias: HTF close < HTF EMA(htf_ema_period)

  LTF Entry (current timeframe):
    - Long:  bull bias AND EMA(ltf_fast) crosses above EMA(ltf_slow)
    - Short: bear bias AND EMA(ltf_fast) crosses below EMA(ltf_slow)

Stop Loss:  ATR(14) × atr_stop_mult
Take Profit: stop_distance × rr_ratio (default 1:2.5)
Sizing:     risk_pct % of equity per trade

Note: HTF is approximated by computing the EMA on a rolling window
      scaled by htf_multiplier (e.g. 8× 30m bars = 4H). This avoids
      actual timeframe resampling while preserving the directional bias.
"""

import numpy as np
import pandas as pd

from ..core.backtesting_engine import BaseStrategy

try:
    import talib
    def calculate_ema(close, period):
        return pd.Series(talib.EMA(pd.Series(close), timeperiod=period)).ffill().bfill()
    def calculate_atr(high, low, close, period):
        return pd.Series(talib.ATR(pd.Series(high), pd.Series(low), pd.Series(close), timeperiod=period)).ffill().bfill().fillna(0)
except ImportError:
    import ta
    def calculate_ema(close, period):
        return ta.trend.ema_indicator(pd.Series(close), window=period).ffill().bfill()
    def calculate_atr(high, low, close, period):
        return ta.volatility.average_true_range(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).ffill().bfill().fillna(0)


def _htf_ema(close, ltf_period, htf_multiplier):
    """
    Approximate HTF EMA by scaling the period by htf_multiplier.
    e.g. EMA(20) on 4H ≈ EMA(20 × 8) on 30m, where htf_multiplier=8
    """
    effective_period = max(2, int(ltf_period * htf_multiplier))
    return calculate_ema(close, effective_period)


class Swing6MTFEMAStackStrategy(BaseStrategy):
    """
    Multi-timeframe EMA stack strategy.
    Uses HTF EMA bias to filter LTF EMA crossover entries.
    """

    # === Parameters ===
    htf_ema_period  = 20     # EMA period on HTF
    htf_multiplier  = 8      # how many LTF bars = 1 HTF bar (e.g. 8 × 30m = 4H)
    ltf_fast        = 9
    ltf_slow        = 21
    atr_period      = 14
    atr_stop_mult   = 2.0
    rr_ratio        = 2.5
    risk_pct        = 1.0

    def init(self):
        self.htf_ema_val = self.I(_htf_ema, self.data.Close, self.htf_ema_period, self.htf_multiplier)
        self.ltf_fast_e  = self.I(calculate_ema, self.data.Close, self.ltf_fast)
        self.ltf_slow_e  = self.I(calculate_ema, self.data.Close, self.ltf_slow)
        self.atr         = self.I(calculate_atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        if not self.should_trade():
            return

        close    = float(self.data.Close[-1])
        htf_ema  = float(self.htf_ema_val[-1])
        fast_cur = float(self.ltf_fast_e[-1])
        fast_prv = float(self.ltf_fast_e[-2]) if len(self.ltf_fast_e) > 1 else fast_cur
        slow_cur = float(self.ltf_slow_e[-1])
        slow_prv = float(self.ltf_slow_e[-2]) if len(self.ltf_slow_e) > 1 else slow_cur
        atr      = float(self.atr[-1])

        if not all(np.isfinite([close, htf_ema, fast_cur, slow_cur, atr])) or atr <= 0:
            return

        bull_bias  = close > htf_ema
        bear_bias  = close < htf_ema

        bull_cross = fast_prv <= slow_prv and fast_cur > slow_cur   # EMA(fast) crossed above EMA(slow)
        bear_cross = fast_prv >= slow_prv and fast_cur < slow_cur   # EMA(fast) crossed below EMA(slow)

        stop_dist = atr * self.atr_stop_mult
        risk_amt  = self.equity * self.risk_pct / 100.0
        qty       = risk_amt / stop_dist if stop_dist > 0 else 0

        if not self.position and qty > 0:
            if bull_bias and bull_cross:
                self.enter_long_position(
                    stop_loss   = close - stop_dist,
                    take_profit = close + stop_dist * self.rr_ratio,
                    atr_value   = atr,
                )
            elif bear_bias and bear_cross:
                self.enter_short_position(
                    stop_loss   = close + stop_dist,
                    take_profit = close - stop_dist * self.rr_ratio,
                    atr_value   = atr,
                )
