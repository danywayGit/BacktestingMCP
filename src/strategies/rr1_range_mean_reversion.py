"""
RR1 — Range Mean Reversion Strategy
=====================================
Timeframe:  4H
Direction:  Long and Short
Target R:R: 1:1.5 (TP1 50%) / 1:2.5 (TP2 50%)

Entry Logic:
  Long:  ADX < adx_threshold AND close <= lower BB
         AND RSI < rsi_oversold
         AND Stochastic %K crosses above %D while both < 20
  Short: ADX < adx_threshold AND close >= upper BB
         AND RSI > rsi_overbought
         AND Stochastic %K crosses below %D while both > 80

Stop Loss:
  Long:  range_high + ATR(14) × sl_buffer_atr  (range_high = highest high over bb_length bars)
  Short: range_low  − ATR(14) × sl_buffer_atr  (range_low  = lowest  low  over bb_length bars)

Exit:
  TP1: 50% of position at SMA20 (mean)
  TP2: remaining 50% at opposite BB extreme
  SL:  full exit
  ADX exit: if ADX crosses above 25 while holding, close all
  Max hold: 200 bars
"""

import numpy as np
import pandas as pd

from ..core.backtesting_engine import BaseStrategy

try:
    import talib

    def _calc_rsi(close, period):
        return pd.Series(talib.RSI(pd.Series(close), timeperiod=period)).ffill().bfill().fillna(50)

    def _calc_atr(high, low, close, period):
        return pd.Series(
            talib.ATR(pd.Series(high), pd.Series(low), pd.Series(close), timeperiod=period)
        ).ffill().bfill().fillna(0)

    def _calc_adx(high, low, close, period):
        return pd.Series(
            talib.ADX(pd.Series(high), pd.Series(low), pd.Series(close), timeperiod=period)
        ).ffill().bfill().fillna(0)

    def _calc_stoch_k(high, low, close, k, d, slow):
        slowk, _slowd = talib.STOCH(
            pd.Series(high), pd.Series(low), pd.Series(close),
            fastk_period=k, slowk_period=slow, slowd_period=d
        )
        return pd.Series(slowk).ffill().bfill().fillna(50)

    def _calc_stoch_d(high, low, close, k, d, slow):
        _slowk, slowd = talib.STOCH(
            pd.Series(high), pd.Series(low), pd.Series(close),
            fastk_period=k, slowk_period=slow, slowd_period=d
        )
        return pd.Series(slowd).ffill().bfill().fillna(50)

except ImportError:
    import ta

    def _calc_rsi(close, period):
        return ta.momentum.rsi(pd.Series(close), window=period).ffill().bfill().fillna(50)

    def _calc_atr(high, low, close, period):
        return ta.volatility.average_true_range(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).ffill().bfill().fillna(0)

    def _calc_adx(high, low, close, period):
        return ta.trend.ADXIndicator(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).adx().ffill().bfill().fillna(0)

    def _calc_stoch_k(high, low, close, k, d, slow):
        stoch = ta.momentum.StochasticOscillator(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close),
            window=k, smooth_window=d
        )
        return stoch.stoch().ffill().bfill().fillna(50)

    def _calc_stoch_d(high, low, close, k, d, slow):
        stoch = ta.momentum.StochasticOscillator(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close),
            window=k, smooth_window=d
        )
        return stoch.stoch_signal().ffill().bfill().fillna(50)


def _calc_bb_upper(close, length, mult):
    s = pd.Series(close)
    sma = s.rolling(length, min_periods=1).mean()
    std = s.rolling(length, min_periods=1).std().fillna(0)
    return (sma + mult * std).ffill().bfill()


def _calc_bb_lower(close, length, mult):
    s = pd.Series(close)
    sma = s.rolling(length, min_periods=1).mean()
    std = s.rolling(length, min_periods=1).std().fillna(0)
    return (sma - mult * std).ffill().bfill()


def _calc_sma(close, length):
    return pd.Series(close).rolling(length, min_periods=1).mean().ffill().bfill()


def _calc_range_high(high, length):
    return pd.Series(high).rolling(length, min_periods=1).max().ffill().bfill()


def _calc_range_low(low, length):
    return pd.Series(low).rolling(length, min_periods=1).min().ffill().bfill()


class RR1RangeMeanReversionStrategy(BaseStrategy):
    """
    RR1 Range Mean Reversion — RSI + Bollinger Bands + Stochastic.
    Enters counter-trend when price hits BB extreme, RSI confirms exhaustion,
    and Stochastic crosses show momentum reversal. ADX confirms ranging market.
    """

    # === Parameters ===
    adx_threshold   = 20
    adx_period      = 14
    rsi_period      = 14
    rsi_oversold    = 30
    rsi_overbought  = 70
    bb_length       = 20
    bb_mult         = 2.0
    stoch_k         = 14
    stoch_d         = 3
    stoch_slow      = 3
    tp1_pct         = 50      # % of position to close at TP1
    sl_buffer_atr   = 0.5
    risk_pct        = 1.0
    max_hold_bars   = 200

    def init(self):
        self.rsi        = self.I(_calc_rsi,  self.data.Close, self.rsi_period)
        self.atr        = self.I(_calc_atr,  self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.adx        = self.I(_calc_adx,  self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.stoch_k_   = self.I(_calc_stoch_k, self.data.High, self.data.Low, self.data.Close,
                                 self.stoch_k, self.stoch_d, self.stoch_slow)
        self.stoch_d_   = self.I(_calc_stoch_d, self.data.High, self.data.Low, self.data.Close,
                                 self.stoch_k, self.stoch_d, self.stoch_slow)
        self.bb_upper   = self.I(_calc_bb_upper, self.data.Close, self.bb_length, self.bb_mult)
        self.bb_lower   = self.I(_calc_bb_lower, self.data.Close, self.bb_length, self.bb_mult)
        self.sma20      = self.I(_calc_sma,  self.data.Close, self.bb_length)
        self.range_high = self.I(_calc_range_high, self.data.High, self.bb_length)
        self.range_low  = self.I(_calc_range_low,  self.data.Low,  self.bb_length)

        # State tracking for partial TP1 and position management
        self._tp1_hit   = False
        self._bars_held = 0
        self._is_long   = False

    def next(self):
        if not self.should_trade():
            return

        close   = float(self.data.Close[-1])
        atr     = float(self.atr[-1])
        adx     = float(self.adx[-1])
        rsi     = float(self.rsi[-1])
        sk      = float(self.stoch_k_[-1])
        sd      = float(self.stoch_d_[-1])
        sk_prv  = float(self.stoch_k_[-2]) if len(self.stoch_k_) > 1 else sk
        sd_prv  = float(self.stoch_d_[-2]) if len(self.stoch_d_) > 1 else sd
        adx_prv = float(self.adx[-2])      if len(self.adx)      > 1 else adx
        bb_up   = float(self.bb_upper[-1])
        bb_lo   = float(self.bb_lower[-1])
        sma     = float(self.sma20[-1])
        r_high  = float(self.range_high[-1])
        r_low   = float(self.range_low[-1])

        if not all(np.isfinite([close, atr, adx, rsi, sk, sd, bb_up, bb_lo, sma, r_high, r_low])):
            return
        if atr <= 0:
            return

        # ── In-position management ──────────────────────────────────────────
        if self.position:
            self._bars_held += 1

            # ADX exit: crossed above 25
            adx_exit = adx_prv <= 25 < adx
            # Max hold exit
            time_exit = self._bars_held >= self.max_hold_bars

            if adx_exit or time_exit:
                self.position.close()
                self._tp1_hit   = False
                self._bars_held = 0
                return

            # TP1: partial close at SMA20
            if not self._tp1_hit:
                if self._is_long and close >= sma:
                    self.position.close(0.5)
                    self._tp1_hit = True
                elif not self._is_long and close <= sma:
                    self.position.close(0.5)
                    self._tp1_hit = True

            # Check if position was fully closed by framework (SL or TP2 hit)
            if not self.position:
                self._tp1_hit   = False
                self._bars_held = 0
            return

        # Reset state when flat
        self._tp1_hit   = False
        self._bars_held = 0

        # ── Entry conditions ─────────────────────────────────────────────────
        ranging = adx < self.adx_threshold

        # Stochastic crossovers
        stoch_cross_up   = sk_prv < sd_prv and sk >= sd and sk < 20 and sd < 20
        stoch_cross_down = sk_prv > sd_prv and sk <= sd and sk > 80 and sd > 80

        # Long entry
        if ranging and close <= bb_lo and rsi < self.rsi_oversold and stoch_cross_up:
            stop  = r_high + atr * self.sl_buffer_atr
            tp2   = r_low   # opposite BB extreme for long = range_low? No: TP2 = range_high for long
            # TP2 for long = range_high (opposite extreme)
            # Wait — range_high is the stop reference, so TP2 should be upper BB / range_high
            # Per spec: "TP2: remaining 50% at opposite extreme (range_high for long)"
            # But stop is also range_high + buffer — so TP2 must be above entry.
            # Use bb_upper as TP2 (the actual opposite price extreme, not the SL anchor)
            tp2   = bb_up
            stop_dist = abs(close - stop)
            if stop_dist <= 0:
                return
            risk_amt = self.equity * self.risk_pct / 100.0
            qty      = risk_amt / stop_dist
            if qty <= 0:
                return
            self._is_long   = True
            self._tp1_hit   = False
            self._bars_held = 0
            self.buy(size=qty, sl=stop, tp=tp2)

        # Short entry
        elif ranging and close >= bb_up and rsi > self.rsi_overbought and stoch_cross_down:
            stop  = r_low - atr * self.sl_buffer_atr
            tp2   = bb_lo   # opposite extreme for short = lower BB
            stop_dist = abs(close - stop)
            if stop_dist <= 0:
                return
            risk_amt = self.equity * self.risk_pct / 100.0
            qty      = risk_amt / stop_dist
            if qty <= 0:
                return
            self._is_long   = False
            self._tp1_hit   = False
            self._bars_held = 0
            self.sell(size=qty, sl=stop, tp=tp2)
