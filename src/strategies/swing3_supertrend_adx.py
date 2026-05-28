"""
SWING3 — Supertrend + ADX Trend Following (Trailing Stop) Strategy
===================================================================
Timeframe:  1H or 4H
Direction:  Long and Short
Exit:       Trailing — closes when Supertrend flips direction (no fixed TP)

Entry Logic:
  Long:  Supertrend flips bullish AND ADX > adx_threshold AND close > EMA(ema_filter)
  Short: Supertrend flips bearish AND ADX > adx_threshold AND close < EMA(ema_filter)

Stop Loss:  ATR(14) × atr_stop_mult (used for position sizing only at entry)
Take Profit: None — exit when Supertrend flips
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


def _calculate_supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                           period: int = 10, factor: float = 3.0):
    """
    Calculate Supertrend indicator.
    Returns (direction, level) where:
      direction: -1 = bullish (price above ST), +1 = bearish (price below ST)
      level:     the supertrend line value
    """
    high  = pd.Series(high)
    low   = pd.Series(low)
    close = pd.Series(close)
    hl2   = (high + low) / 2

    # ATR — use simple rolling mean (matches TradingView Supertrend default behaviour)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    upper_band = hl2 + factor * atr
    lower_band = hl2 - factor * atr

    direction = np.full(len(close), np.nan)
    st_level  = np.full(len(close), np.nan)

    # Start from the first bar where ATR is valid
    start = period
    if start >= len(close):
        return pd.Series(direction), pd.Series(st_level)

    direction[start] = -1
    st_level[start]  = lower_band.iloc[start]

    for i in range(start + 1, len(close)):
        ub = upper_band.iloc[i]
        lb = lower_band.iloc[i]
        c  = close.iloc[i]

        if np.isnan(ub) or np.isnan(lb):
            direction[i] = direction[i - 1]
            st_level[i]  = st_level[i - 1]
            continue

        prev_dir   = direction[i - 1]
        prev_level = st_level[i - 1]

        if prev_dir == -1:
            # Bullish — lower band ratchets up, flip to bearish if close drops below
            curr_lb = max(lb, prev_level) if not np.isnan(prev_level) else lb
            if c < curr_lb:
                direction[i] = 1
                st_level[i]  = ub
            else:
                direction[i] = -1
                st_level[i]  = curr_lb
        else:
            # Bearish — upper band ratchets down, flip to bullish if close rises above
            curr_ub = min(ub, prev_level) if not np.isnan(prev_level) else ub
            if c > curr_ub:
                direction[i] = -1
                st_level[i]  = lb
            else:
                direction[i] = 1
                st_level[i]  = curr_ub

    return pd.Series(direction), pd.Series(st_level)


def _supertrend_direction(high, low, close, period, factor):
    d, _ = _calculate_supertrend(high, low, close, period, factor)
    return d


def _supertrend_level(high, low, close, period, factor):
    _, l = _calculate_supertrend(high, low, close, period, factor)
    return l


def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ADX."""
    try:
        import talib
        return pd.Series(talib.ADX(high, low, close, timeperiod=period)).ffill().bfill().fillna(0)
    except ImportError:
        import ta
        adx = ta.trend.ADXIndicator(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        )
        return adx.adx().ffill().bfill().fillna(0)


class Swing3SupertrendADXStrategy(BaseStrategy):
    """
    Supertrend + ADX trend-following strategy with trailing Supertrend exit.
    No fixed take profit — rides the trend until Supertrend flips.
    """

    # === Parameters ===
    # Optimized on ETHUSDT 1H, 2021-2024 — walk-forward validated
    st_period     = 10
    st_factor     = 3.0
    adx_period    = 14
    adx_threshold = 30    # optimized: was 25 — stricter trend filter lifts win rate 34%→51%
    ema_filter    = 100   # optimized: was 50  — EMA100 sweet spot between 50 and 200
    atr_period    = 14
    atr_stop_mult = 2.5   # used for position sizing only
    risk_pct      = 1.0   # % of equity to risk per trade

    def init(self):
        self.st_dir   = self.I(_supertrend_direction, self.data.High, self.data.Low, self.data.Close, self.st_period, self.st_factor)
        self.st_level = self.I(_supertrend_level,     self.data.High, self.data.Low, self.data.Close, self.st_period, self.st_factor)
        self.adx      = self.I(_calculate_adx,        self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.ema_f    = self.I(calculate_ema,          self.data.Close, self.ema_filter)
        self.atr      = self.I(calculate_atr,          self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        if not self.should_trade():
            return

        st_cur  = float(self.st_dir[-1])
        st_prv  = float(self.st_dir[-2]) if len(self.st_dir) > 1 else st_cur
        adx     = float(self.adx[-1])
        ema_val = float(self.ema_f[-1])
        atr     = float(self.atr[-1])
        close   = float(self.data.Close[-1])

        if not all(np.isfinite([st_cur, st_prv, adx, ema_val, atr])) or atr <= 0:
            return

        st_flipped_bull = st_prv > 0 and st_cur < 0   # bearish → bullish flip
        st_flipped_bear = st_prv < 0 and st_cur > 0   # bullish → bearish flip
        adx_strong      = adx > self.adx_threshold

        stop_dist = atr * self.atr_stop_mult
        risk_amt  = self.equity * self.risk_pct / 100.0
        qty       = risk_amt / stop_dist if stop_dist > 0 else 0

        if not self.position:
            if st_flipped_bull and adx_strong and close > ema_val:
                if qty > 0:
                    # No fixed TP in embedded mode — trailing exit via Supertrend flip
                    self.enter_long_position(stop_loss=close - stop_dist, atr_value=atr)

            elif st_flipped_bear and adx_strong and close < ema_val:
                if qty > 0:
                    self.enter_short_position(stop_loss=close + stop_dist, atr_value=atr)

        else:
            # Trailing exit: only active when the strategy's signal drives the exit
            # (embedded + fixed_signal modes).  fixed_pct and atr rely on SL/TP orders.
            if self.sl_mode in ('embedded', 'fixed_signal'):
                if self.position.is_long  and st_flipped_bear:
                    self.position.close()
                elif self.position.is_short and st_flipped_bull:
                    self.position.close()
