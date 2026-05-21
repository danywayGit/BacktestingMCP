"""
SWING1 — EMA Wave + Volume Trend Following Strategy
====================================================
Timeframe:  1H
Direction:  Long and Short
Target R:R: 1:3

Entry Logic:
  Long:  Fast EMA > Slow EMA AND RSI crosses above rsi_long_threshold
         AND close > fast EMA AND volume > SMA(volume, vol_ma_period)
  Short: Fast EMA < Slow EMA AND RSI crosses below rsi_short_threshold
         AND close < fast EMA AND volume > SMA(volume, vol_ma_period)

Stop Loss:  ATR(14) × atr_stop_mult
Take Profit: stop_distance × rr_ratio  (default 1:3)
Sizing:     risk_pct % of equity per trade
"""

import numpy as np
import pandas as pd

from ..core.backtesting_engine import BaseStrategy

try:
    import talib
    def calculate_ema(close, period):
        return pd.Series(talib.EMA(pd.Series(close), timeperiod=period)).ffill().bfill()
    def calculate_rsi(close, period):
        return pd.Series(talib.RSI(pd.Series(close), timeperiod=period)).ffill().bfill().fillna(50)
    def calculate_atr(high, low, close, period):
        return pd.Series(talib.ATR(pd.Series(high), pd.Series(low), pd.Series(close), timeperiod=period)).ffill().bfill().fillna(0)
except ImportError:
    import ta
    def calculate_ema(close, period):
        return ta.trend.ema_indicator(pd.Series(close), window=period).ffill().bfill()
    def calculate_rsi(close, period):
        return ta.momentum.rsi(pd.Series(close), window=period).ffill().bfill().fillna(50)
    def calculate_atr(high, low, close, period):
        return ta.volatility.average_true_range(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).ffill().bfill().fillna(0)


def _calculate_volume_sma(volumes: pd.Series, period: int) -> pd.Series:
    """Simple moving average of volume."""
    return pd.Series(volumes).rolling(period, min_periods=1).mean().ffill().bfill()


class Swing1EmaWaveVolumeStrategy(BaseStrategy):
    """
    EMA Wave + Volume trend-following strategy.
    Enters on RSI momentum cross when EMAs are aligned and volume confirms.
    """

    # === Parameters (all are auto-discoverable by BacktestingMCP optimizer) ===
    ema_fast            = 9
    ema_slow            = 21
    rsi_period          = 14
    rsi_long_threshold  = 40    # RSI must cross ABOVE this for long entry
    rsi_short_threshold = 60    # RSI must cross BELOW this for short entry
    vol_ma_period       = 20
    atr_period          = 14
    atr_stop_mult       = 2.0
    rr_ratio            = 3.0
    risk_pct            = 1.0   # % of equity to risk per trade

    def init(self):
        self.ema_f   = self.I(calculate_ema, self.data.Close, self.ema_fast)
        self.ema_s   = self.I(calculate_ema, self.data.Close, self.ema_slow)
        self.rsi     = self.I(calculate_rsi, self.data.Close, self.rsi_period)
        self.atr     = self.I(calculate_atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.vol_sma = self.I(_calculate_volume_sma, self.data.Volume, self.vol_ma_period)

    def next(self):
        if not self.should_trade():
            return

        ema_f   = float(self.ema_f[-1])
        ema_s   = float(self.ema_s[-1])
        rsi_cur = float(self.rsi[-1])
        rsi_prv = float(self.rsi[-2]) if len(self.rsi) > 1 else rsi_cur
        atr     = float(self.atr[-1])
        vol     = float(self.data.Volume[-1])
        vol_avg = float(self.vol_sma[-1])
        close   = float(self.data.Close[-1])

        if not all(np.isfinite([ema_f, ema_s, rsi_cur, atr, vol_avg])) or atr <= 0:
            return

        high_vol = vol > vol_avg

        # RSI crossover detection (manual, since we don't have crossover() for raw values)
        rsi_crossed_above = rsi_prv < self.rsi_long_threshold  <= rsi_cur
        rsi_crossed_below = rsi_prv > self.rsi_short_threshold >= rsi_cur

        stop_dist  = atr * self.atr_stop_mult
        risk_amt   = self.equity * self.risk_pct / 100.0
        qty        = risk_amt / stop_dist if stop_dist > 0 else 0

        if not self.position:
            # Long entry
            if ema_f > ema_s and rsi_crossed_above and close > ema_f and high_vol:
                if qty > 0:
                    self.enter_long_position(
                        stop_loss   = close - stop_dist,
                        take_profit = close + stop_dist * self.rr_ratio,
                    )

            # Short entry
            elif ema_f < ema_s and rsi_crossed_below and close < ema_f and high_vol:
                if qty > 0:
                    self.enter_short_position(
                        stop_loss   = close + stop_dist,
                        take_profit = close - stop_dist * self.rr_ratio,
                    )
