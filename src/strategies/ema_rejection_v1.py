"""
EMA_REJ_V1 — EMA200 Rejection (Failed Breakout/Breakdown) Strategy
===================================================================
Timeframe:  1H or 4H (recommended)
Direction:  Long and Short
Target R:R: 1:2

Concept:
  Trades failed recoveries/breakdowns at the EMA200.
  A "rejection" = price briefly crosses to the other side of EMA200
  then reverses back within a short window, confirming EMA200 as
  hard resistance/support.

Entry Logic:
  Short (failed recovery):
    1. HTF (htf_bars × current bar = ~9H) close < HTF EMA200 → downtrend context
    2. Price crossed ABOVE EMA200 within last rejection_lookback bars
    3. Price NOW crosses back BELOW EMA200
    4. RSI EMA cross (RSI < EMA of RSI) within last rsi_confirm_window bars

  Long (failed breakdown, inverse):
    1. HTF close > HTF EMA200 → uptrend context
    2. Price crossed BELOW EMA200 within last rejection_lookback bars
    3. Price NOW crosses back ABOVE EMA200
    4. RSI crossed above its EMA within last rsi_confirm_window bars

Stop Loss:  ATR(14) × stop_mult (default 3×)
Take Profit: stop_distance × rr_ratio (default 1:2)
Sizing:     risk_pct % of equity per trade

Note: RSI EMA is computed inline as EMA(rsi_ema_period) of RSI(rsi_period).
      HTF bias approximated by EMA(ema200_length × htf_bars) on current TF.
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


def _rsi_ema(close, rsi_period, ema_period):
    """EMA of RSI — used as RSI signal line."""
    rsi = calculate_rsi(close, rsi_period)
    return calculate_ema(rsi, ema_period)


def _htf_ema200(close, ema200_length, htf_bars):
    """Approximate HTF EMA200 by scaling the period."""
    effective = max(2, int(ema200_length * htf_bars))
    return calculate_ema(close, effective)


class EmaRejectionV1Strategy(BaseStrategy):
    """
    EMA200 rejection strategy.
    Trades failed recoveries (short) and failed breakdowns (long) at EMA200.
    """

    # === Parameters ===
    ema200_length       = 200
    htf_bars            = 9      # approximate HTF multiplier (9 × 1H ≈ 9H)
    rejection_lookback  = 10     # max bars between the initial cross and the rejection cross
    rsi_period          = 14
    rsi_ema_period      = 9      # EMA of RSI used as signal line
    rsi_confirm_window  = 3      # RSI cross must occur within this many bars
    stop_mult           = 3.0
    rr_ratio            = 2.0
    risk_pct            = 1.0

    def init(self):
        self.ema200     = self.I(calculate_ema,   self.data.Close, self.ema200_length)
        self.htf_ema    = self.I(_htf_ema200,     self.data.Close, self.ema200_length, self.htf_bars)
        self.rsi        = self.I(calculate_rsi,   self.data.Close, self.rsi_period)
        self.rsi_ema    = self.I(_rsi_ema,        self.data.Close, self.rsi_period, self.rsi_ema_period)
        self.atr        = self.I(calculate_atr,   self.data.High, self.data.Low, self.data.Close, 14)

        # State: bars since last cross above / below EMA200 and RSI EMA
        self._bars_since_cross_above = 9999
        self._bars_since_cross_below = 9999
        self._bars_since_rsi_cross_above = 9999
        self._bars_since_rsi_cross_below = 9999

    def next(self):
        if not self.should_trade():
            return

        close    = float(self.data.Close[-1])
        ema200   = float(self.ema200[-1])
        htf_ema  = float(self.htf_ema[-1])
        rsi      = float(self.rsi[-1])
        rsi_ema  = float(self.rsi_ema[-1])
        atr      = float(self.atr[-1])

        close_prv   = float(self.data.Close[-2])   if len(self.data.Close)  > 1 else close
        ema200_prv  = float(self.ema200[-2])        if len(self.ema200)      > 1 else ema200
        rsi_prv     = float(self.rsi[-2])           if len(self.rsi)         > 1 else rsi
        rsi_ema_prv = float(self.rsi_ema[-2])       if len(self.rsi_ema)     > 1 else rsi_ema

        if not all(np.isfinite([close, ema200, htf_ema, rsi, rsi_ema, atr])) or atr <= 0:
            return

        # --- Track crosses ---
        cross_above = close_prv < ema200_prv and close >= ema200    # just crossed above
        cross_below = close_prv > ema200_prv and close <= ema200    # just crossed below

        rsi_cross_above = rsi_prv < rsi_ema_prv and rsi >= rsi_ema
        rsi_cross_below = rsi_prv > rsi_ema_prv and rsi <= rsi_ema

        if cross_above:
            self._bars_since_cross_above = 0
        else:
            self._bars_since_cross_above += 1

        if cross_below:
            self._bars_since_cross_below = 0
        else:
            self._bars_since_cross_below += 1

        if rsi_cross_above:
            self._bars_since_rsi_cross_above = 0
        else:
            self._bars_since_rsi_cross_above += 1

        if rsi_cross_below:
            self._bars_since_rsi_cross_below = 0
        else:
            self._bars_since_rsi_cross_below += 1

        # --- Rejection signals ---
        # Short: HTF downtrend, price crossed above then back below EMA200
        short_ema_rejection = cross_below and self._bars_since_cross_above <= self.rejection_lookback
        short_rsi_confirm   = self._bars_since_rsi_cross_below <= self.rsi_confirm_window
        htf_downtrend       = close < htf_ema

        # Long: HTF uptrend, price crossed below then back above EMA200
        long_ema_rejection  = cross_above and self._bars_since_cross_below <= self.rejection_lookback
        long_rsi_confirm    = self._bars_since_rsi_cross_above <= self.rsi_confirm_window
        htf_uptrend         = close > htf_ema

        stop_dist = atr * self.stop_mult
        risk_amt  = self.equity * self.risk_pct / 100.0
        qty       = risk_amt / stop_dist if stop_dist > 0 else 0

        if not self.position and qty > 0:
            if htf_downtrend and short_ema_rejection and short_rsi_confirm:
                self.enter_short_position(
                    stop_loss   = close + stop_dist,
                    take_profit = close - stop_dist * self.rr_ratio,
                )
            elif htf_uptrend and long_ema_rejection and long_rsi_confirm:
                self.enter_long_position(
                    stop_loss   = close - stop_dist,
                    take_profit = close + stop_dist * self.rr_ratio,
                )
