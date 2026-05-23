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
# Single-pass VWAP helper — all derived values computed in one function call
# ---------------------------------------------------------------------------

def _vwap_all(close, high, low, volume, index, band_mult, band_std_period, vol_avg_period):
    """Compute all VWAP-derived values in one pass. Returns shape (5, N) array:
    row 0: vwap, row 1: upper_band, row 2: lower_band, row 3: vol_ratio, row 4: bars_since_midnight
    """
    index = pd.DatetimeIndex(index)
    tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
    vol_s = pd.Series(volume)
    tpv = tp * vol_s
    df = pd.DataFrame({'tpv': tpv, 'vol': vol_s, 'close': pd.Series(close)}, index=index)
    df['date'] = df.index.normalize()
    df['cum_tpv'] = df.groupby('date')['tpv'].cumsum()
    df['cum_vol'] = df.groupby('date')['vol'].cumsum()
    vwap = (df['cum_tpv'] / df['cum_vol'].replace(0, np.nan)).ffill().bfill()

    dev = df['close'] - vwap
    sigma = dev.rolling(band_std_period, min_periods=1).std().ffill().bfill().fillna(0)
    upper = (vwap + band_mult * sigma).ffill().bfill()
    lower = (vwap - band_mult * sigma).ffill().bfill()

    vol_avg = vol_s.rolling(vol_avg_period, min_periods=1).mean()
    vol_ratio = (vol_s / vol_avg.replace(0, np.nan)).ffill().bfill().fillna(1.0)

    # bars since midnight (resets to 0 at each day boundary)
    bsm = np.zeros(len(df), dtype=float)
    current_date = None
    counter = 0
    for i, d in enumerate(df['date']):
        if d != current_date:
            current_date = d
            counter = 0
        bsm[i] = counter
        counter += 1

    return np.vstack([vwap.values, upper.values, lower.values, vol_ratio.values, bsm])


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

        # Single-pass VWAP computation: rows [vwap, upper, lower, vol_ratio, bars_since_midnight]
        self._vwap_data = self.I(_vwap_all, self.data.Close, self.data.High, self.data.Low,
                                  self.data.Volume, idx,
                                  self.band_mult, self.band_std_period, self.vol_avg_period)
        # Row accessors (use as self._vwap_data[0] etc.)

        # RSI and ATR (independent indicators, kept as separate self.I() calls)
        self.rsi = self.I(calculate_rsi, self.data.Close, self.rsi_period)
        self.atr = self.I(calculate_atr, self.data.High, self.data.Low, self.data.Close, 14)

        # Entry bar counter for max_hold_bars
        self._entry_bar = 0

    def next(self):
        if not self.should_trade():
            return

        close    = float(self.data.Close[-1])
        close_p  = float(self.data.Close[-2]) if len(self.data.Close) > 1 else close

        vwap     = float(self._vwap_data[0][-1])
        upper    = float(self._vwap_data[1][-1])
        lower    = float(self._vwap_data[2][-1])

        upper_p  = float(self._vwap_data[1][-2]) if len(self._vwap_data[0]) > 1 else upper
        lower_p  = float(self._vwap_data[2][-2]) if len(self._vwap_data[0]) > 1 else lower

        vol_r_p  = float(self._vwap_data[3][-2]) if len(self._vwap_data[0]) > 1 else 1.0
        rsi_p    = float(self.rsi[-2]) if len(self.rsi) > 1 else 50.0
        atr      = float(self.atr[-1])
        bsm      = float(self._vwap_data[4][-1])

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

        lower_band_width = vwap - lower  # distance from VWAP down to lower band

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
            raw_stop_dist = lower_band_width * 1.5
            stop_dist = max(raw_stop_dist, atr * 1.0)
            stop_loss   = close - stop_dist
            take_profit = vwap  # mean reversion target
            if stop_loss >= close or take_profit <= close:
                # Skip: VWAP is below entry price — band may be miscalibrated, wait for reset
                return
            self._entry_bar = len(self.data)
            self.enter_long_position(stop_loss=stop_loss, take_profit=take_profit)

        elif short_cond:
            upper_band_width = upper - vwap  # distance from upper band down to VWAP
            raw_stop_dist = upper_band_width * 1.5
            stop_dist = max(raw_stop_dist, atr * 1.0)
            stop_loss   = close + stop_dist
            take_profit = vwap  # mean reversion target
            if stop_loss <= close or take_profit >= close:
                # Skip: VWAP is above entry price — band may be miscalibrated, wait for reset
                return
            self._entry_bar = len(self.data)
            self.enter_short_position(stop_loss=stop_loss, take_profit=take_profit)
