"""
AGGR_PB — Aggressive Pullback to EMA20 (Engulfing Reversal) Strategy
======================================================================
Timeframe:  1H or 4H (recommended)
Direction:  Long and Short
Target R:R: 1:2

Concept:
  Trades high-quality engulfing candles that form during a pullback to EMA20,
  in the direction of the EMA200 major trend. Five conditions must align:
  trend context, candlestick pattern, pullback quality, swing structure, and
  candle size filter.

Entry Logic:
  Long:
    1. close > EMA(ema_length) AND close > EMA(ema200_length) → uptrend both scales
    2. Bullish engulfing: current green fully engulfs previous red candle
    3. Pullback quality: ≤ pullback_tolerance bars of last 3 closed below EMA20
    4. Swing structure: current OR previous bar is the swing_lookback-bar lowest low
    5. Not massive candle: true range < ATR × massive_candle_mult

  Short (inverse):
    1. close < EMA20 AND close < EMA200
    2. Bearish engulfing
    3. ≤ pullback_tolerance bars of last 3 closed above EMA20
    4. Current or previous bar is the swing_lookback-bar highest high
    5. True range < ATR × massive_candle_mult

Stop Loss:  (close - swing_low_N_bars) + ATR × stop_mult   [long]
            (swing_high_N_bars - close) + ATR × stop_mult  [short]
Take Profit: stop_distance × rr_ratio (default 1:2)
Sizing:     risk_pct % of equity per trade
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


class AggrPullbackStrategy(BaseStrategy):
    """
    Engulfing candle pullback to EMA20 strategy.
    Highest-conviction entry — all 5 filters must align.
    """

    # === Parameters ===
    ema_length           = 20
    ema200_length        = 200
    atr_period           = 14
    stop_mult            = 3.0    # ATR multiplier added to swing distance for stop
    rr_ratio             = 2.0
    pullback_tolerance   = 1      # max bars of last 3 that can close on wrong side of EMA20
    swing_lookback       = 7      # bars to define swing low/high
    massive_candle_mult  = 2.0    # true range must be < ATR × this to avoid news spikes
    risk_pct             = 1.0

    def init(self):
        self.ema20  = self.I(calculate_ema, self.data.Close, self.ema_length)
        self.ema200 = self.I(calculate_ema, self.data.Close, self.ema200_length)
        self.atr    = self.I(calculate_atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        if not self.should_trade():
            return

        n = self.swing_lookback
        pb = self.pullback_tolerance
        min_bars = max(n + 2, 205)   # need enough bars for EMA200 and swing

        if len(self.data.Close) < min_bars:
            return

        close   = float(self.data.Close[-1])
        open_   = float(self.data.Open[-1])
        high    = float(self.data.High[-1])
        low     = float(self.data.Low[-1])
        close1  = float(self.data.Close[-2])
        open1   = float(self.data.Open[-2])
        high1   = float(self.data.High[-2])

        ema20   = float(self.ema20[-1])
        ema200  = float(self.ema200[-1])
        atr     = float(self.atr[-1])

        if not all(np.isfinite([close, open_, high, low, ema20, ema200, atr])) or atr <= 0:
            return

        # --- True range (current bar) ---
        true_range = max(high - low, abs(high - close1), abs(low - close1))

        # --- Engulfing patterns ---
        bull_engulf = (close > open_ and close1 < open1 and
                       close > open1 and open_ <= close1)
        bear_engulf = (close < open_ and close1 > open1 and
                       close < open1 and open_ >= close1)

        # --- Pullback quality (last 3 bars, not including current) ---
        bars_below_ema = sum(
            1 for i in range(1, 4)
            if float(self.data.Close[-i - 1]) < float(self.ema20[-i - 1])
        )
        bars_above_ema = sum(
            1 for i in range(1, 4)
            if float(self.data.Close[-i - 1]) > float(self.ema20[-i - 1])
        )
        long_pullback_ok  = bars_below_ema  <= pb
        short_pullback_ok = bars_above_ema  <= pb

        # --- Swing structure ---
        swing_lows  = [float(self.data.Low[-(i+1)])  for i in range(n)]
        swing_highs = [float(self.data.High[-(i+1)]) for i in range(n)]
        swing_low_n  = min(swing_lows)
        swing_high_n = max(swing_highs)

        is_swing_low  = (low  == swing_low_n  or float(self.data.Low[-2])  == min([float(self.data.Low[-(i+2)])  for i in range(n)]))
        is_swing_high = (high == swing_high_n or float(self.data.High[-2]) == max([float(self.data.High[-(i+2)]) for i in range(n)]))

        # --- Massive candle filter ---
        not_massive = true_range < atr * self.massive_candle_mult

        # --- Stop distance (swing + ATR buffer) ---
        long_stop_dist  = abs(close - swing_low_n)  + atr * self.stop_mult
        short_stop_dist = abs(swing_high_n - close) + atr * self.stop_mult
        long_stop_dist  = max(long_stop_dist,  atr * 0.5)
        short_stop_dist = max(short_stop_dist, atr * 0.5)

        risk_amt = self.equity * self.risk_pct / 100.0

        if not self.position:
            # Long condition
            long_cond = (close > ema20 and close > ema200 and
                         bull_engulf and long_pullback_ok and
                         is_swing_low and not_massive)
            if long_cond:
                qty = risk_amt / long_stop_dist if long_stop_dist > 0 else 0
                if qty > 0:
                    self.enter_long_position(
                        stop_loss   = close - long_stop_dist,
                        take_profit = close + long_stop_dist * self.rr_ratio,
                    )

            # Short condition
            short_cond = (close < ema20 and close < ema200 and
                          bear_engulf and short_pullback_ok and
                          is_swing_high and not_massive)
            if short_cond:
                qty = risk_amt / short_stop_dist if short_stop_dist > 0 else 0
                if qty > 0:
                    self.enter_short_position(
                        stop_loss   = close + short_stop_dist,
                        take_profit = close - short_stop_dist * self.rr_ratio,
                    )
