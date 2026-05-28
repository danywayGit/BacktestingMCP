"""
SWING5 — Keltner Channel Breakout Strategy
==========================================
Timeframe:  1H
Direction:  Long and Short
Target R:R: 1:3

Entry Logic:
  Keltner Channel: EMA(kc_length) ± ATR(kc_length) × kc_mult

  Long:  close > upper KC AND CCI(cci_period) > cci_long_min
         (not deeply oversold — confirms bullish momentum)
  Short: close < lower KC AND CCI(cci_period) < cci_short_max
         (not deeply overbought)

Stop Loss:  EMA midline price at entry (price returning to midline = failed breakout)
Take Profit: entry ± stop_distance × rr_ratio
Sizing:     risk_pct % of equity per trade

Note: stop_distance = abs(entry_price - ema_at_entry)
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
    def calculate_cci(high, low, close, period):
        return pd.Series(talib.CCI(pd.Series(high), pd.Series(low), pd.Series(close), timeperiod=period)).ffill().bfill().fillna(0)
except ImportError:
    import ta
    def calculate_ema(close, period):
        return ta.trend.ema_indicator(pd.Series(close), window=period).ffill().bfill()
    def calculate_atr(high, low, close, period):
        return ta.volatility.average_true_range(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).ffill().bfill().fillna(0)
    def calculate_cci(high, low, close, period):
        return ta.trend.cci(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).ffill().bfill().fillna(0)


class Swing5KeltnerBreakoutStrategy(BaseStrategy):
    """
    Keltner Channel breakout strategy with CCI filter.
    Enters on close outside KC bands; stop at EMA midline, 1:3 R:R target.
    """

    # === Parameters ===
    kc_length      = 20
    kc_mult        = 2.0
    cci_period     = 20
    cci_long_min   = -100   # CCI must be above this for long (not over-extended short)
    cci_short_max  = 100    # CCI must be below this for short (not over-extended long)
    atr_stop_mult  = 2.0    # used by BaseStrategy 'atr' sl_mode
    rr_ratio       = 3.0
    risk_pct       = 1.0

    def init(self):
        self.ema  = self.I(calculate_ema, self.data.Close, self.kc_length)
        self.atr  = self.I(calculate_atr, self.data.High, self.data.Low, self.data.Close, self.kc_length)
        self.cci  = self.I(calculate_cci, self.data.High, self.data.Low, self.data.Close, self.cci_period)

    def next(self):
        if not self.should_trade():
            return

        close = float(self.data.Close[-1])
        ema   = float(self.ema[-1])
        atr   = float(self.atr[-1])
        cci   = float(self.cci[-1])

        if not all(np.isfinite([close, ema, atr, cci])) or atr <= 0:
            return

        upper_kc = ema + self.kc_mult * atr
        lower_kc = ema - self.kc_mult * atr

        # Stop distance = distance from entry to EMA midline
        long_stop_dist  = abs(close - ema) if close > ema else atr * 2   # fallback if already at EMA
        short_stop_dist = abs(close - ema) if close < ema else atr * 2

        long_stop_dist  = max(long_stop_dist, atr * 0.5)   # floor at 0.5× ATR
        short_stop_dist = max(short_stop_dist, atr * 0.5)

        risk_amt  = self.equity * self.risk_pct / 100.0

        if not self.position:
            # Long: breakout above upper KC, CCI not over-sold
            if close > upper_kc and cci > self.cci_long_min:
                qty = risk_amt / long_stop_dist if long_stop_dist > 0 else 0
                if qty > 0:
                    self.enter_long_position(
                        stop_loss   = ema,
                        take_profit = close + long_stop_dist * self.rr_ratio,
                        atr_value   = float(self.atr[-1]),
                    )
            # Short: breakdown below lower KC, CCI not over-bought
            elif close < lower_kc and cci < self.cci_short_max:
                qty = risk_amt / short_stop_dist if short_stop_dist > 0 else 0
                tp  = close - short_stop_dist * self.rr_ratio
                if qty > 0 and tp > 0:  # guard: TP must be positive
                    self.enter_short_position(
                        stop_loss   = ema,
                        take_profit = tp,
                        atr_value   = float(self.atr[-1]),
                    )
