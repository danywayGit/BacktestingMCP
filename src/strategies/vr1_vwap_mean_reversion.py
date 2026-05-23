"""
VR1 — VWAP Mean Reversion (Z-score + Volume Exhaustion)
========================================================
Timeframe:  1H
Direction:  Long and Short
Target R:R: 1:1.5 to VWAP

Entry Logic:
  Long:  Previous bar closed below lower VWAP band AND volume exhaustion
         AND RSI not too extreme AND current bar reclaimed the band
  Short: Previous bar closed above upper VWAP band AND volume exhaustion
         AND RSI not too extreme AND current bar dropped back below band

Stop Loss:  entry ± band_width × 1.5  (floored at ATR × 1.0)
Take Profit: VWAP (mean reversion target)
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
        return pd.Series(
            talib.ATR(pd.Series(high), pd.Series(low), pd.Series(close), timeperiod=period)
        ).ffill().bfill().fillna(0)
except ImportError:
    import ta
    def calculate_rsi(close, period):
        return ta.momentum.rsi(pd.Series(close), window=period).ffill().bfill().fillna(50)
    def calculate_atr(high, low, close, period):
        return ta.volatility.average_true_range(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).ffill().bfill().fillna(0)


# ---------------------------------------------------------------------------
# VWAP helpers — operate on full numpy arrays, called once via self.I()
# ---------------------------------------------------------------------------

def _vwap_line(close, high, low, volume, index, band_std_period, vol_avg_period):
    """
    Compute daily-reset VWAP from full arrays.
    Returns vwap values as numpy array.
    index: pd.DatetimeIndex passed from self.data.index
    """
    tp = (pd.Series(high, dtype=float) + pd.Series(low, dtype=float) + pd.Series(close, dtype=float)) / 3.0
    vol_s = pd.Series(volume, dtype=float)
    idx = pd.DatetimeIndex(index)

    df = pd.DataFrame({'tp': tp.values, 'vol': vol_s.values}, index=idx)
    df['date'] = df.index.normalize()  # UTC date

    df['tpv'] = df['tp'] * df['vol']
    df['cum_tpv'] = df.groupby('date')['tpv'].cumsum()
    df['cum_vol'] = df.groupby('date')['vol'].cumsum()
    df['vwap'] = df['cum_tpv'] / df['cum_vol'].replace(0, np.nan)
    df['vwap'] = df['vwap'].ffill().bfill()

    return df['vwap'].values


def _vwap_upper_band(close, high, low, volume, index, band_mult, band_std_period, vol_avg_period):
    """VWAP upper band = VWAP + band_mult * rolling_std(close - VWAP, band_std_period)."""
    vwap = _vwap_line(close, high, low, volume, index, band_std_period, vol_avg_period)
    close_s = pd.Series(close, dtype=float)
    dev = close_s - pd.Series(vwap)
    sigma = dev.rolling(band_std_period, min_periods=1).std().fillna(0)
    upper = pd.Series(vwap) + band_mult * sigma
    return upper.values


def _vwap_lower_band(close, high, low, volume, index, band_mult, band_std_period, vol_avg_period):
    """VWAP lower band = VWAP - band_mult * rolling_std(close - VWAP, band_std_period)."""
    vwap = _vwap_line(close, high, low, volume, index, band_std_period, vol_avg_period)
    close_s = pd.Series(close, dtype=float)
    dev = close_s - pd.Series(vwap)
    sigma = dev.rolling(band_std_period, min_periods=1).std().fillna(0)
    lower = pd.Series(vwap) - band_mult * sigma
    return lower.values


def _vol_ratio(volume, index, vol_avg_period, band_std_period):
    """Volume / SMA(volume, vol_avg_period). Extra args ignored (band_std_period for arity compat)."""
    vol_s = pd.Series(volume, dtype=float)
    avg = vol_s.rolling(vol_avg_period, min_periods=1).mean().replace(0, np.nan)
    ratio = (vol_s / avg).fillna(1.0)
    return ratio.values


def _bars_since_midnight(close, high, low, volume, index, band_std_period, vol_avg_period):
    """Return number of bars elapsed since last UTC midnight reset (0-indexed)."""
    idx = pd.DatetimeIndex(index)
    dates = np.array([d.date() for d in idx])
    result = np.zeros(len(dates), dtype=int)
    counter = 0
    for i in range(len(dates)):
        if i > 0 and dates[i] != dates[i - 1]:
            counter = 0
        result[i] = counter
        counter += 1
    return result.astype(float)


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class VR1VWAPMeanReversionStrategy(BaseStrategy):
    """
    VWAP Mean Reversion with daily-reset VWAP bands, volume exhaustion filter,
    and RSI guard. Targets 1:1.5 R:R back to VWAP.
    """

    # === Parameters ===
    band_mult                   = 2.0
    band_std_period             = 50
    volume_exhaustion_threshold = 0.3
    vol_avg_period              = 20
    rsi_period                  = 14
    rsi_oversold_floor          = 35
    rsi_overbought_ceil         = 65
    min_band_width_pct          = 0.5
    max_hold_bars               = 150
    risk_pct                    = 1.0

    def init(self):
        idx = self.data.index  # pd.DatetimeIndex

        # VWAP line
        self.vwap = self.I(
            _vwap_line,
            self.data.Close, self.data.High, self.data.Low, self.data.Volume,
            idx, self.band_std_period, self.vol_avg_period,
        )

        # Upper band
        self.upper_band = self.I(
            _vwap_upper_band,
            self.data.Close, self.data.High, self.data.Low, self.data.Volume,
            idx, self.band_mult, self.band_std_period, self.vol_avg_period,
        )

        # Lower band
        self.lower_band = self.I(
            _vwap_lower_band,
            self.data.Close, self.data.High, self.data.Low, self.data.Volume,
            idx, self.band_mult, self.band_std_period, self.vol_avg_period,
        )

        # Volume ratio
        self.vol_ratio = self.I(
            _vol_ratio,
            self.data.Volume, idx, self.vol_avg_period, self.band_std_period,
        )

        # Bars since midnight (for first-bar guard)
        self.bars_since_midnight = self.I(
            _bars_since_midnight,
            self.data.Close, self.data.High, self.data.Low, self.data.Volume,
            idx, self.band_std_period, self.vol_avg_period,
        )

        # RSI and ATR
        self.rsi = self.I(calculate_rsi, self.data.Close, self.rsi_period)
        self.atr = self.I(calculate_atr, self.data.High, self.data.Low, self.data.Close, 14)

        # Entry bar counter for max_hold_bars
        self._entry_bar = 0

    def next(self):
        if not self.should_trade():
            return

        close    = float(self.data.Close[-1])
        close_p  = float(self.data.Close[-2]) if len(self.data.Close) > 1 else close

        vwap     = float(self.vwap[-1])
        upper    = float(self.upper_band[-1])
        lower    = float(self.lower_band[-1])

        upper_p  = float(self.upper_band[-2]) if len(self.upper_band) > 1 else upper
        lower_p  = float(self.lower_band[-2]) if len(self.lower_band) > 1 else lower

        vol_r_p  = float(self.vol_ratio[-2]) if len(self.vol_ratio) > 1 else 1.0
        rsi_p    = float(self.rsi[-2]) if len(self.rsi) > 1 else 50.0
        atr      = float(self.atr[-1])
        bsm      = float(self.bars_since_midnight[-1])

        # NaN / ATR guard
        if not all(np.isfinite([close, vwap, upper, lower, vol_r_p, rsi_p, atr])):
            return
        if atr <= 0:
            return

        band_width = upper - lower

        # Max-hold exit
        if self.position:
            bars_held = len(self.data) - self._entry_bar
            if bars_held >= self.max_hold_bars:
                self.position.close()
            return

        # Band width filter
        if band_width <= 0 or (band_width / close) < (self.min_band_width_pct / 100.0):
            return

        # First-2-bars guard
        if bsm < 2:
            return

        vwap_lower_half = vwap - lower  # = lower band half-width

        # --- Long entry ---
        long_cond = (
            close_p < lower_p                               # prev bar below lower band
            and vol_r_p < self.volume_exhaustion_threshold  # volume exhaustion
            and rsi_p > self.rsi_oversold_floor             # not too extreme
            and close > lower                               # current bar reclaimed band
        )

        # --- Short entry ---
        short_cond = (
            close_p > upper_p                              # prev bar above upper band
            and vol_r_p < self.volume_exhaustion_threshold # volume exhaustion
            and rsi_p < self.rsi_overbought_ceil           # not too extreme
            and close < upper                              # current bar dropped back below band
        )

        if long_cond:
            raw_stop_dist = vwap_lower_half * 1.5
            stop_dist = max(raw_stop_dist, atr * 1.0)
            stop_loss   = close - stop_dist
            take_profit = vwap  # mean reversion target
            if stop_loss >= close or take_profit <= close:
                return
            self._entry_bar = len(self.data)
            self.enter_long_position(stop_loss=stop_loss, take_profit=take_profit)

        elif short_cond:
            upper_half = upper - vwap
            raw_stop_dist = upper_half * 1.5
            stop_dist = max(raw_stop_dist, atr * 1.0)
            stop_loss   = close + stop_dist
            take_profit = vwap  # mean reversion target
            if stop_loss <= close or take_profit >= close:
                return
            self._entry_bar = len(self.data)
            self.enter_short_position(stop_loss=stop_loss, take_profit=take_profit)
