"""
DC1 — Donchian Channel + ATR Filter (Mini Turtle)
===================================================
Timeframe:  4H
Direction:  Long and Short
Target R:R: Trailing exit (no fixed TP)

Entry Logic:
  Long:  close > upper_band (20-period highest high, excluding current bar)
         AND ADX(14) > adx_threshold
         AND volume > SMA(volume, vol_avg_period) × vol_mult

  Short: close < lower_band
         AND ADX(14) > adx_threshold
         AND volume > SMA(volume, vol_avg_period) × vol_mult

Stop Loss:   ATR(14) × sl_atr_mult (absolute price)
Trailing:    Activated after +1R move; trail distance = ATR(14) × trail_atr_mult
Sizing:      risk_pct % of equity per trade
"""

import numpy as np
import pandas as pd

from ..core.backtesting_engine import BaseStrategy

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
        return ta.trend.ADXIndicator(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).adx().ffill().bfill().fillna(0)


def _calc_donchian_upper(high, period):
    """Highest high of last `period` bars, excluding current bar (shift by 1)."""
    return pd.Series(high).rolling(period).max().shift(1).ffill().bfill()


def _calc_donchian_lower(low, period):
    """Lowest low of last `period` bars, excluding current bar (shift by 1)."""
    return pd.Series(low).rolling(period).min().shift(1).ffill().bfill()


def _calc_vol_sma(vol, period):
    return pd.Series(vol).rolling(period, min_periods=1).mean().ffill().bfill()


class DC1DonchianChannelStrategy(BaseStrategy):
    """
    DC1 — Donchian Channel breakout with ADX trend filter and ATR-based trailing stop.
    Inspired by the classic Turtle Trading rules.
    """

    # === Parameters ===
    donchian_length       = 20
    donchian_exit_length  = 10
    adx_threshold         = 25
    adx_exit              = 20
    atr_stop_mult         = 2.0
    trail_atr_mult        = 2.0
    atr_period            = 14
    adx_period            = 14
    vol_avg_period        = 20
    vol_mult              = 1.0
    min_channel_width_pct = 1.0
    risk_pct              = 1.0
    max_hold_bars         = 200
    cooldown_bars         = 20

    def init(self):
        self.dc_upper = self.I(_calc_donchian_upper, self.data.High, self.donchian_length)
        self.dc_lower = self.I(_calc_donchian_lower, self.data.Low,  self.donchian_length)
        self.dc_exit_upper = self.I(_calc_donchian_upper, self.data.High, self.donchian_exit_length)
        self.dc_exit_lower = self.I(_calc_donchian_lower, self.data.Low,  self.donchian_exit_length)
        self.atr      = self.I(_calc_atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.adx      = self.I(_calc_adx, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.vol_sma  = self.I(_calc_vol_sma, self.data.Volume, self.vol_avg_period)

        # Trailing stop / position state
        self._trail_active      = False
        self._trail_stop        = None
        self._entry_price       = None
        self._entry_stop_dist   = None
        self._bars_held         = 0
        self._bars_since_exit   = 9999

    def _reset_trade_state(self):
        self._trail_active    = False
        self._trail_stop      = None
        self._entry_price     = None
        self._entry_stop_dist = None
        self._bars_held       = 0
        self._bars_since_exit = 0

    def next(self):
        if not self.should_trade():
            return

        close     = float(self.data.Close[-1])
        upper     = float(self.dc_upper[-1])
        lower     = float(self.dc_lower[-1])
        atr       = float(self.atr[-1])
        adx       = float(self.adx[-1])
        vol       = float(self.data.Volume[-1])
        vol_avg   = float(self.vol_sma[-1])

        if not all(np.isfinite([close, upper, lower, atr, adx, vol_avg])) or atr <= 0:
            return

        # ── In-position management ─────────────────────────────────────────────
        if self.position:
            if self.sl_mode in ('embedded', 'fixed_signal'):
                is_long  = self.position.is_long
                is_short = not is_long

                # Update / activate trailing stop
                if self._trail_active:
                    if is_long:
                        new_trail = close - atr * self.trail_atr_mult
                        if self._trail_stop is None or new_trail > self._trail_stop:
                            self._trail_stop = new_trail
                    else:
                        new_trail = close + atr * self.trail_atr_mult
                        if self._trail_stop is None or new_trail < self._trail_stop:
                            self._trail_stop = new_trail
                else:
                    # Activate after +1R move
                    if self._entry_price is not None and self._entry_stop_dist is not None:
                        move = close - self._entry_price if is_long else self._entry_price - close
                        if move >= self._entry_stop_dist:
                            self._trail_active = True
                            self._trail_stop = (
                                close - atr * self.trail_atr_mult if is_long
                                else close + atr * self.trail_atr_mult
                            )

                # Exit conditions (in priority order)
                exit_reason = None

                # 0. Initial stop loss guard (before trail activates)
                if not self._trail_active and exit_reason is None:
                    if is_long  and close <= self._entry_price - self._entry_stop_dist:
                        exit_reason = "initial_stop"
                    elif not is_long and close >= self._entry_price + self._entry_stop_dist:
                        exit_reason = "initial_stop"

                # 1. Trailing stop hit
                if self._trail_active and self._trail_stop is not None and exit_reason is None:
                    if is_long  and close <= self._trail_stop:
                        exit_reason = "trail_stop_long"
                    elif is_short and close >= self._trail_stop:
                        exit_reason = "trail_stop_short"

                # 2. ADX weakened
                if exit_reason is None and adx < self.adx_exit:
                    exit_reason = "adx_exit"

                # 3. Price returned inside exit channel (classic Turtle 10-bar channel)
                if exit_reason is None:
                    dc_exit_upper = float(self.dc_exit_upper[-1])
                    dc_exit_lower = float(self.dc_exit_lower[-1])
                    if is_long  and close < dc_exit_lower:
                        exit_reason = "donchian_exit_long"
                    elif is_short and close > dc_exit_upper:
                        exit_reason = "donchian_exit_short"

                # 4. Max holding period
                if exit_reason is None and self._bars_held >= self.max_hold_bars:
                    exit_reason = "max_hold"

                # Increment bars_held AFTER exit checks (so entry bar = 0, exits at exactly max_hold_bars)
                self._bars_held += 1

                if exit_reason:
                    self.position.close()
                    self._reset_trade_state()
            return

        # ── No position — increment cooldown counter ───────────────────────────
        self._bars_since_exit += 1

        # ── Entry conditions ───────────────────────────────────────────────────
        # Cooldown guard
        if self._bars_since_exit < self.cooldown_bars:
            return

        # Minimum channel width filter
        channel_width_pct = (upper - lower) / close * 100.0
        if channel_width_pct < self.min_channel_width_pct:
            return

        # Volume filter
        high_vol = vol > vol_avg * self.vol_mult

        # ADX filter
        strong_trend = adx > self.adx_threshold

        stop_dist = atr * self.atr_stop_mult
        if stop_dist <= 0:
            return
        # Long entry
        if close > upper and strong_trend and high_vol:
            sl_price = close - stop_dist
            self.enter_long_position(stop_loss=sl_price, atr_value=atr)
            self._entry_price     = close
            self._entry_stop_dist = stop_dist
            self._bars_held       = 0
            self._trail_active    = False
            self._trail_stop      = None

        # Short entry
        elif close < lower and strong_trend and high_vol:
            sl_price = close + stop_dist
            self.enter_short_position(stop_loss=sl_price, atr_value=atr)
            self._entry_price     = close
            self._entry_stop_dist = stop_dist
            self._bars_held       = 0
            self._trail_active    = False
            self._trail_stop      = None
