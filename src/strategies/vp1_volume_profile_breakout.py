"""
VP1 — Volume Profile Breakout (POC + VAH/VAL)
==============================================
Timeframe:  1H
Direction:  Long and Short
Target R:R: 1:2

Entry Logic:
  Long:  close > VAH  AND  volume spike  AND  close > POC  AND  ADX > threshold
  Short: close < VAL  AND  volume spike  AND  close < POC  AND  ADX > threshold

Stop Loss:  POC price (volume centre of gravity)
Take Profit:
  TP1 (50%): entry ± VA_height × tp1_mult
  TP2 (50%): entry ± VA_height × tp2_mult
Sizing:     risk_pct % of equity per trade
"""

import numpy as np
import pandas as pd

from ..core.backtesting_engine import BaseStrategy

# ---------------------------------------------------------------------------
# Indicator helpers (talib / ta fallback)
# ---------------------------------------------------------------------------
try:
    import talib

    def _calc_atr(high, low, close, period):
        return pd.Series(
            talib.ATR(pd.Series(high), pd.Series(low), pd.Series(close), timeperiod=period)
        ).ffill().bfill().fillna(0)

    def _calc_adx(high, low, close, period):
        return pd.Series(
            talib.ADX(pd.Series(high), pd.Series(low), pd.Series(close), timeperiod=period)
        ).ffill().bfill().fillna(0)

except ImportError:
    import ta

    def _calc_atr(high, low, close, period):
        return ta.volatility.average_true_range(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).ffill().bfill().fillna(0)

    def _calc_adx(high, low, close, period):
        return ta.trend.adx(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).ffill().bfill().fillna(0)


def _calc_vol_sma(volumes, period):
    return pd.Series(volumes).rolling(period, min_periods=1).mean().ffill().bfill()


# ---------------------------------------------------------------------------
# Volume Profile builder
# ---------------------------------------------------------------------------
def _build_volume_profile(highs, lows, closes, volumes, bin_size, value_area_pct):
    """
    Build approximate volume profile from OHLCV arrays.
    Returns (poc_price, vah_price, val_price) or (None, None, None).

    Each bar's volume is distributed uniformly across the price bins that
    span [low, high] of that bar.  POC is the bin with the most volume.
    VAH/VAL are found by expanding outward from POC until value_area_pct
    of total volume is captured.
    """
    if bin_size <= 0 or len(closes) < 2:
        return None, None, None

    lo = float(min(lows))
    hi = float(max(highs))
    if hi <= lo:
        return None, None, None

    n_bins = max(1, min(int((hi - lo) / bin_size) + 1, 2000))
    bins = np.zeros(n_bins)

    for i in range(len(closes)):
        bar_lo = float(lows[i])
        bar_hi = float(highs[i])
        bar_vol = float(volumes[i])

        if bar_hi <= bar_lo:
            # Point bar — put all volume in closest bin
            bin_idx = min(int((bar_lo - lo) / bin_size), n_bins - 1)
            bins[bin_idx] += bar_vol
            continue

        idx_lo = int((bar_lo - lo) / bin_size)
        idx_hi = min(int((bar_hi - lo) / bin_size), n_bins - 1)
        count = max(1, idx_hi - idx_lo + 1)
        vol_per_bin = bar_vol / count
        bins[idx_lo : idx_hi + 1] += vol_per_bin

    poc_idx = int(np.argmax(bins))
    poc_price = lo + (poc_idx + 0.5) * bin_size

    total_vol = bins.sum()
    if total_vol <= 0:
        return poc_price, None, None

    target = total_vol * value_area_pct / 100.0
    accumulated = bins[poc_idx]
    lo_idx = poc_idx
    hi_idx = poc_idx

    while accumulated < target:
        can_lo = lo_idx > 0
        can_hi = hi_idx < n_bins - 1
        if not can_lo and not can_hi:
            break
        lo_candidate = bins[lo_idx - 1] if can_lo else -1.0
        hi_candidate = bins[hi_idx + 1] if can_hi else -1.0
        if lo_candidate >= hi_candidate:
            lo_idx -= 1
            accumulated += bins[lo_idx]
        else:
            hi_idx += 1
            accumulated += bins[hi_idx]

    val_price = lo + lo_idx * bin_size
    vah_price = lo + (hi_idx + 1) * bin_size
    return poc_price, vah_price, val_price


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class VP1VolumeProfileBreakoutStrategy(BaseStrategy):
    """
    VP1 — Volume Profile Breakout.

    Trades breakouts above VAH (long) or below VAL (short), confirmed by
    volume spike and ADX trend filter.  Stop is set at POC; partial TP at
    1× VA height, full TP at 2× VA height.
    """

    # === Parameters ===
    profile_lookback   = 200
    value_area_pct     = 70
    volume_spike_mult  = 1.5
    volume_avg_period  = 20
    adx_threshold      = 20
    tp1_mult           = 1.0
    tp2_mult           = 2.0
    min_va_pct         = 0.5    # minimum VA height as % of close
    cooldown_bars      = 20
    risk_pct           = 1.0
    max_hold_bars      = 300
    atr_stop_mult      = 2.0    # used by BaseStrategy 'atr' sl_mode
    rr_ratio           = 2.0    # used by BaseStrategy 'atr' sl_mode TP

    # ADX / ATR period (not user-facing but discoverable by optimizer)
    adx_period         = 14
    atr_period         = 14

    # ------------------------------------------------------------------
    def init(self):
        self.atr     = self.I(_calc_atr,     self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.adx     = self.I(_calc_adx,     self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.vol_sma = self.I(_calc_vol_sma, self.data.Volume, self.volume_avg_period)

        # Per-trade state
        self._tp1_hit         = False
        self._bars_held       = 0
        self._bars_since_exit = 9999
        self._entry_is_long   = None   # True=long, False=short, None=flat
        self._tp1_price       = None
        self._tp2_price       = None

    # ------------------------------------------------------------------
    def next(self):
        if not self.should_trade():
            return

        close   = float(self.data.Close[-1])
        atr     = float(self.atr[-1])
        adx     = float(self.adx[-1])
        vol     = float(self.data.Volume[-1])
        vol_avg = float(self.vol_sma[-1])

        if not all(np.isfinite([close, atr, adx, vol_avg])) or atr <= 0:
            return

        # ------------------------------------------------------------------
        # Build volume profile from the last `profile_lookback` bars
        # ------------------------------------------------------------------
        n_bars = len(self.data)
        window = min(self.profile_lookback, n_bars)
        if window < 5:
            return

        highs   = np.asarray(self.data.High[-window:],   dtype=float)
        lows    = np.asarray(self.data.Low[-window:],    dtype=float)
        closes  = np.asarray(self.data.Close[-window:],  dtype=float)
        volumes = np.asarray(self.data.Volume[-window:], dtype=float)

        bin_size = atr / 4.0
        poc, vah, val = _build_volume_profile(highs, lows, closes, volumes, bin_size, self.value_area_pct)

        if poc is None or vah is None or val is None:
            return

        va_height = vah - val
        if va_height <= 0:
            return

        # VA height filter
        if (va_height / close) < (self.min_va_pct / 100.0):
            return

        # ------------------------------------------------------------------
        # In-position management
        # ------------------------------------------------------------------
        if self.position:
            if self.sl_mode in ('embedded', 'fixed_signal'):
                self._bars_held += 1

                # Max-hold exit
                if self._bars_held >= self.max_hold_bars:
                    self.position.close()
                    self._reset_trade_state()
                    return

                # Manual POC / value-area re-entry stop
                if self._entry_is_long and close <= poc:
                    self.position.close()
                    self._reset_trade_state()
                    return
                if self._entry_is_long is False and close >= poc:
                    self.position.close()
                    self._reset_trade_state()
                    return

                # TP1 partial close (50 %)
                if self._tp1_price is not None and not self._tp1_hit:
                    if self._entry_is_long and close >= self._tp1_price:
                        self.position.close(0.5)
                        self._tp1_hit = True
                    elif self._entry_is_long is False and close <= self._tp1_price:
                        self.position.close(0.5)
                        self._tp1_hit = True

                # TP2 — remaining 50 % at second target
                if self._tp1_hit:
                    if (self._entry_is_long and close >= self._tp2_price) or \
                       (self._entry_is_long is False and close <= self._tp2_price):
                        self.position.close()
                        self._reset_trade_state()
                        return

            return  # no new entries while in position

        # ------------------------------------------------------------------
        # Detect position just closed by framework (SL hit)
        # ------------------------------------------------------------------
        if not self.position and self._entry_is_long is not None:
            self._reset_trade_state()

        # ------------------------------------------------------------------
        # Cooldown check
        # ------------------------------------------------------------------
        self._bars_since_exit += 1
        if self._bars_since_exit < self.cooldown_bars:
            return

        # ------------------------------------------------------------------
        # Volume spike filter
        # ------------------------------------------------------------------
        volume_spike = vol_avg > 0 and vol > vol_avg * self.volume_spike_mult

        # ------------------------------------------------------------------
        # Entry signals
        # ------------------------------------------------------------------
        long_signal  = (close > vah) and volume_spike and (close > poc) and (adx > self.adx_threshold)
        short_signal = (close < val) and volume_spike and (close < poc) and (adx > self.adx_threshold)

        if not long_signal and not short_signal:
            return

        # ------------------------------------------------------------------
        # Position sizing
        # ------------------------------------------------------------------
        stop_dist = abs(close - poc)
        stop_dist = max(stop_dist, atr * 0.5)   # floor

        if long_signal:
            sl_price  = poc
            tp2_price = close + va_height * self.tp2_mult
            tp1_price = close + va_height * self.tp1_mult
            self._entry_is_long = True
        else:
            sl_price  = poc
            tp2_price = close - va_height * self.tp2_mult
            tp1_price = close - va_height * self.tp1_mult
            self._entry_is_long = False

        # Store TP1, TP2 for in-bar management
        self._tp1_price = tp1_price
        self._tp2_price = tp2_price
        self._tp1_hit   = False
        self._bars_held = 0

        if long_signal:
            self.enter_long_position(stop_loss=sl_price, atr_value=atr)
        else:
            self.enter_short_position(stop_loss=sl_price, atr_value=atr)

    # ------------------------------------------------------------------
    def _reset_trade_state(self):
        self._tp1_hit         = False
        self._bars_held       = 0
        self._bars_since_exit = 0
        self._entry_is_long   = None
        self._tp1_price       = None
        self._tp2_price       = None
