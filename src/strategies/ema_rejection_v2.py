"""
EMA_REJ_V2 — EMA200 Rejection v2 (Confirmed Stay + RSI Threshold)
==================================================================
Timeframe:  1H or 4H (recommended)
Direction:  Long and Short
Target R:R: 1:2

Improvements over V1:
  - min_bars_below_ema / min_bars_above_ema: price must have spent at least N
    consecutive bars on the correct side of EMA200 *before* the false cross,
    proving trend context (fixes BUG-003 from original Pine Script).
  - rsi_threshold_short / rsi_threshold_long: RSI ceiling/floor at entry to
    prevent entering over-extended moves.

3-Phase Rejection Pattern:
  Short:
    1. Price below EMA200 for >= min_bars_below_ema consecutive bars
    2. Price crosses ABOVE EMA200 (false recovery)
    3. Price crosses back BELOW EMA200 within rejection_lookback bars → SHORT
  Long (inverse):
    1. Price above EMA200 for >= min_bars_above_ema consecutive bars
    2. Price crosses BELOW EMA200 (false breakdown)
    3. Price crosses back ABOVE EMA200 within rejection_lookback bars → LONG

Stop Loss:  ATR(14) × stop_mult (default 3×)
Take Profit: stop_distance × rr_ratio (default 1:2)
Sizing:     risk_pct % of equity per trade

BUG-003 Fix:
  The original _shortStayedBelow_ checked bars [1..N] at the crossunder bar —
  but close[1] was definitionally above EMA200 (crossunder precondition) → always 0.
  Fix: use persistent counters (_bars_below_ema, _bars_above_ema) that increment
  each bar price is on the correct side and reset on cross. At the moment of a
  false cross-above, snapshot _bars_before_cross_above = _bars_below_ema.
  The short entry requires _bars_before_cross_above >= min_bars_below_ema.
"""

import numpy as np
import pandas as pd

from ..core.backtesting_engine import BaseStrategy

try:
    import talib
    def _calc_ema(close, period):
        return pd.Series(talib.EMA(pd.Series(close), timeperiod=period)).ffill().bfill()
    def _calc_rsi(close, period):
        return pd.Series(talib.RSI(pd.Series(close), timeperiod=period)).ffill().bfill().fillna(50)
    def _calc_atr(high, low, close, period):
        return pd.Series(
            talib.ATR(pd.Series(high), pd.Series(low), pd.Series(close), timeperiod=period)
        ).ffill().bfill().fillna(0)
except ImportError:
    import ta
    def _calc_ema(close, period):
        return ta.trend.ema_indicator(pd.Series(close), window=period).ffill().bfill()
    def _calc_rsi(close, period):
        return ta.momentum.rsi(pd.Series(close), window=period).ffill().bfill().fillna(50)
    def _calc_atr(high, low, close, period):
        return ta.volatility.average_true_range(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).ffill().bfill().fillna(0)


def _rsi_ema(close, rsi_period, ema_period):
    """EMA of RSI — used as RSI signal line."""
    rsi = _calc_rsi(close, rsi_period)
    return _calc_ema(rsi, ema_period)


def _htf_ema200(close, ema200_length, htf_bars):
    """Approximate HTF EMA200 by scaling the period to current TF."""
    effective = max(2, int(ema200_length * htf_bars))
    return _calc_ema(close, effective)


class EmaRejectionV2Strategy(BaseStrategy):
    """
    EMA200 Rejection v2.

    Adds two quality filters over V1:
      1. Minimum confirmation bars (3-phase rejection pattern)
      2. RSI threshold filter at entry

    Key fix: persistent bar counters correctly capture how many consecutive
    bars price spent on the correct side of EMA200 before the false cross.
    """

    # === Parameters ===
    ema200_length       = 200
    htf_bars            = 9       # HTF multiplier (9 × 1H ≈ 9H)
    rejection_lookback  = 10      # max bars between false cross and rejection cross
    min_bars_below_ema  = 3       # min consecutive bars below EMA before false cross-above (short context)
    min_bars_above_ema  = 3       # min consecutive bars above EMA before false cross-below (long context)
    rsi_period          = 14
    rsi_ema_period      = 9
    rsi_confirm_window  = 3       # RSI cross must occur within N bars of price rejection
    rsi_threshold_short = 55.0    # RSI must be < this for short entry
    rsi_threshold_long  = 45.0    # RSI must be > this for long entry
    atr_stop_mult       = 3.0   # renamed from stop_mult for sl_mode framework alignment
    rr_ratio            = 2.0
    risk_pct            = 1.0

    def init(self):
        self.ema200  = self.I(_calc_ema,    self.data.Close, self.ema200_length)
        self.htf_ema = self.I(_htf_ema200,  self.data.Close, self.ema200_length, self.htf_bars)
        self.rsi     = self.I(_calc_rsi,    self.data.Close, self.rsi_period)
        self.rsi_sig = self.I(_rsi_ema,     self.data.Close, self.rsi_period, self.rsi_ema_period)
        self.atr     = self.I(_calc_atr,    self.data.High, self.data.Low, self.data.Close, 14)

        # --- EMA cross tracking (bars since) ---
        self._bars_since_cross_above     = 9999
        self._bars_since_cross_below     = 9999
        self._bars_since_rsi_cross_above = 9999
        self._bars_since_rsi_cross_below = 9999

        # --- BUG-003 fix: persistent below/above counters ---
        # Increment every bar close is on the side, reset on cross.
        self._bars_below_ema = 0   # consecutive bars close < ema200
        self._bars_above_ema = 0   # consecutive bars close > ema200

        # Snapshotted at the moment of false cross
        self._bars_before_cross_above = 0  # bars below EMA just before price crossed above
        self._bars_before_cross_below = 0  # bars above EMA just before price crossed below

    def next(self):
        if not self.should_trade():
            return

        close     = float(self.data.Close[-1])
        ema200    = float(self.ema200[-1])
        htf_ema   = float(self.htf_ema[-1])
        rsi       = float(self.rsi[-1])
        rsi_sig   = float(self.rsi_sig[-1])
        atr       = float(self.atr[-1])

        close_prv    = float(self.data.Close[-2]) if len(self.data.Close) > 1 else close
        ema200_prv   = float(self.ema200[-2])     if len(self.ema200)     > 1 else ema200
        rsi_prv      = float(self.rsi[-2])        if len(self.rsi)        > 1 else rsi
        rsi_sig_prv  = float(self.rsi_sig[-2])    if len(self.rsi_sig)    > 1 else rsi_sig

        if not all(np.isfinite([close, ema200, htf_ema, rsi, rsi_sig, atr])) or atr <= 0:
            return

        # ── EMA cross detection ────────────────────────────────────────────
        cross_above = close_prv < ema200_prv and close >= ema200   # just crossed above EMA200
        cross_below = close_prv > ema200_prv and close <= ema200   # just crossed below EMA200

        rsi_cross_above = rsi_prv < rsi_sig_prv and rsi >= rsi_sig
        rsi_cross_below = rsi_prv > rsi_sig_prv and rsi <= rsi_sig

        # ── BUG-003 fix: update persistent side counters ──────────────────
        # Save last bar's counts FIRST (used as snapshot when a cross fires this bar)
        prev_bars_below = self._bars_below_ema
        prev_bars_above = self._bars_above_ema

        if close < ema200:
            self._bars_below_ema += 1
            self._bars_above_ema = 0
        elif close > ema200:
            self._bars_above_ema += 1
            self._bars_below_ema = 0
        else:
            self._bars_below_ema = 0
            self._bars_above_ema = 0

        # When price crosses ABOVE EMA200 this bar:
        #   close >= ema200, close_prv < ema200_prv
        #   prev_bars_below = how many consecutive bars were below EMA before this cross
        if cross_above:
            self._bars_since_cross_above = 0
            self._bars_before_cross_above = prev_bars_below
        else:
            self._bars_since_cross_above += 1

        if cross_below:
            self._bars_since_cross_below = 0
            self._bars_before_cross_below = prev_bars_above
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

        # ── Rejection signals ──────────────────────────────────────────────
        # Short: HTF downtrend, was below N bars, crossed above (false recovery), now back below
        short_ema_rejection = cross_below and self._bars_since_cross_above <= self.rejection_lookback
        short_stayed_below  = self._bars_before_cross_above >= self.min_bars_below_ema
        short_rsi_confirm   = self._bars_since_rsi_cross_below <= self.rsi_confirm_window
        short_rsi_threshold = rsi < self.rsi_threshold_short
        htf_downtrend       = close < htf_ema

        # Long: HTF uptrend, was above N bars, crossed below (false breakdown), now back above
        long_ema_rejection  = cross_above and self._bars_since_cross_below <= self.rejection_lookback
        long_stayed_above   = self._bars_before_cross_below >= self.min_bars_above_ema
        long_rsi_confirm    = self._bars_since_rsi_cross_above <= self.rsi_confirm_window
        long_rsi_threshold  = rsi > self.rsi_threshold_long
        htf_uptrend         = close > htf_ema

        stop_dist = atr * self.atr_stop_mult
        risk_amt  = self.equity * self.risk_pct / 100.0
        qty       = risk_amt / stop_dist if stop_dist > 0 else 0

        if not self.position and qty > 0:
            if htf_downtrend and short_ema_rejection and short_stayed_below and short_rsi_confirm and short_rsi_threshold:
                self.enter_short_position(
                    stop_loss   = close + stop_dist,
                    take_profit = close - stop_dist * self.rr_ratio,
                    atr_value   = atr,
                )
            elif htf_uptrend and long_ema_rejection and long_stayed_above and long_rsi_confirm and long_rsi_threshold:
                self.enter_long_position(
                    stop_loss   = close - stop_dist,
                    take_profit = close + stop_dist * self.rr_ratio,
                    atr_value   = atr,
                )
