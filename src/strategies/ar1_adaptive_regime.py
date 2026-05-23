"""
AR1 — Adaptive Regime Switcher (Meta-Strategy)
===============================================
Timeframe:  4H
Direction:  Long and Short
R:R:        Inherited from sub-strategies (SWING3 / RR1)

Regime Classification (every reclass_interval bars):
  Trend Up:   ADX > adx_up   AND close > EMA(ema_trend)  → SWING3 long-only
  Trend Down: ADX > adx_down AND close < EMA(ema_trend)  → SWING3 short-only
  Range:      ADX < adx_range                             → RR1 both directions
  Grey Zone:  adx_range ≤ ADX ≤ adx_up/down              → standby (no trades)

Regime switch override:
  If regime changes while in a trade:
    - Trade profitable (+0.5R): close immediately
    - Not profitable: hold until sub-strategy's own exit

Churn protection:
  If regime switched >churn_threshold_switches times in last 20 bars → 50% size reduction
"""

import numpy as np
import pandas as pd

from ..core.backtesting_engine import BaseStrategy
from .swing3_supertrend_adx import (
    _supertrend_direction,
    _supertrend_level,
    _calculate_adx,
    calculate_ema,
    calculate_atr,
)
from .rr1_range_mean_reversion import (
    _stoch_both,
    _calc_rsi,
    _calc_bb_upper,
    _calc_bb_lower,
    _calc_sma,
    _calc_range_high,
    _calc_range_low,
)


class AR1AdaptiveRegimeStrategy(BaseStrategy):
    """
    AR1 Adaptive Regime Switcher — routes signal logic to SWING3 (trend) or
    RR1 (range) based on ADX + EMA regime classification.
    Sub-strategy logic is inlined (no sub-object instantiation).
    """

    # ── Regime parameters ──────────────────────────────────────────────────
    adx_up                   = 25
    adx_down                 = 25
    adx_range                = 20
    ema_trend                = 200
    adx_lookback             = 20
    reclass_interval         = 50
    profit_take_on_switch_r  = 0.5   # R-multiple threshold to close on regime switch
    churn_threshold_switches = 2

    # ── SWING3 parameters ──────────────────────────────────────────────────
    st_period      = 10
    st_factor      = 3.0
    adx_st_period  = 14
    adx_threshold  = 30    # ADX filter for SWING3 entries
    ema_filter     = 100   # EMA filter for SWING3 entries
    atr_period     = 14
    atr_stop_mult  = 2.5

    # ── RR1 parameters ─────────────────────────────────────────────────────
    rsi_period      = 14
    rsi_oversold    = 30
    rsi_overbought  = 70
    bb_length       = 20
    bb_mult         = 2.0
    stoch_k         = 14
    stoch_d         = 3
    stoch_slow      = 3
    sl_buffer_atr   = 0.5
    max_hold_bars   = 200
    tp1_pct         = 50    # % of position to close at TP1 (SMA20)

    # ── Shared ─────────────────────────────────────────────────────────────
    risk_pct = 1.0

    # ──────────────────────────────────────────────────────────────────────

    def init(self):
        # Regime classification indicators
        self.adx_regime = self.I(_calculate_adx, self.data.High, self.data.Low, self.data.Close, self.adx_lookback)
        self.ema_t      = self.I(calculate_ema,  self.data.Close, self.ema_trend)

        # SWING3 indicators
        self.st_dir     = self.I(_supertrend_direction, self.data.High, self.data.Low, self.data.Close, self.st_period, self.st_factor)
        self.st_level   = self.I(_supertrend_level,     self.data.High, self.data.Low, self.data.Close, self.st_period, self.st_factor)
        self.adx_st     = self.I(_calculate_adx, self.data.High, self.data.Low, self.data.Close, self.adx_st_period)
        self.ema_f      = self.I(calculate_ema,  self.data.Close, self.ema_filter)
        self.atr        = self.I(calculate_atr,  self.data.High, self.data.Low, self.data.Close, self.atr_period)

        # RR1 indicators
        self.rsi        = self.I(_calc_rsi,      self.data.Close, self.rsi_period)
        self._stoch     = self.I(_stoch_both,    self.data.High, self.data.Low, self.data.Close,
                                 self.stoch_k, self.stoch_d, self.stoch_slow)
        self.bb_upper   = self.I(_calc_bb_upper, self.data.Close, self.bb_length, self.bb_mult)
        self.bb_lower   = self.I(_calc_bb_lower, self.data.Close, self.bb_length, self.bb_mult)
        self.sma20      = self.I(_calc_sma,      self.data.Close, self.bb_length)
        self.range_high = self.I(_calc_range_high, self.data.High, self.bb_length)
        self.range_low  = self.I(_calc_range_low,  self.data.Low,  self.bb_length)

        # State
        self._regime              = None
        self._last_reclass_bar    = 0
        self._entry_price         = None
        self._entry_stop_dist     = None
        self._entry_is_long       = None
        self._regime_switches     = []   # bar indices when regime changed
        self._size_multiplier     = 1.0

        # RR1 position state
        self._tp1_hit   = False
        self._bars_held = 0
        self._is_long   = False

    # ── Regime helpers ─────────────────────────────────────────────────────

    def _classify_regime(self, adx_val, close_val, ema_val):
        if adx_val > self.adx_up and close_val > ema_val:
            return 'trend_up'
        elif adx_val > self.adx_down and close_val < ema_val:
            return 'trend_down'
        elif adx_val < self.adx_range:
            return 'range'
        else:
            return 'grey'

    def _update_churn(self, current_bar):
        self._regime_switches = [b for b in self._regime_switches if current_bar - b <= 20]
        if len(self._regime_switches) > self.churn_threshold_switches:
            self._size_multiplier = 0.5
        else:
            self._size_multiplier = 1.0

    def _check_regime_switch_exit(self, new_regime):
        """Return True if position should be closed due to a profitable regime switch."""
        if new_regime != self._regime and new_regime != 'grey':
            if self._entry_price is not None and self._entry_stop_dist is not None:
                close = float(self.data.Close[-1])
                if self._entry_is_long:
                    pnl = close - self._entry_price
                else:
                    pnl = self._entry_price - close
                r_multiple = pnl / self._entry_stop_dist if self._entry_stop_dist > 0 else 0
                if r_multiple >= self.profit_take_on_switch_r:
                    return True
        return False

    def _reset_trade_state(self):
        self._entry_price     = None
        self._entry_stop_dist = None
        self._entry_is_long   = None
        self._tp1_hit         = False
        self._bars_held       = 0
        self._is_long         = False
        self._size_multiplier = 1.0

    # ── Main loop ──────────────────────────────────────────────────────────

    def next(self):
        if not self.should_trade():
            return

        bar_idx = len(self.data.Close) - 1

        # ── Gather indicator values ────────────────────────────────────────
        close    = float(self.data.Close[-1])
        adx_reg  = float(self.adx_regime[-1])
        ema_t    = float(self.ema_t[-1])
        atr      = float(self.atr[-1])

        if not all(np.isfinite([close, adx_reg, ema_t, atr])) or atr <= 0:
            return

        # ── Regime reclassification ────────────────────────────────────────
        if bar_idx - self._last_reclass_bar >= self.reclass_interval or self._regime is None:
            new_regime = self._classify_regime(adx_reg, close, ema_t)

            if new_regime != self._regime and self._regime is not None:
                self._regime_switches.append(bar_idx)
                self._update_churn(bar_idx)

                # Regime switch override
                if self.position and self._check_regime_switch_exit(new_regime):
                    self.position.close()
                    self._reset_trade_state()

            self._regime = new_regime
            self._last_reclass_bar = bar_idx

        # ── In-position management ─────────────────────────────────────────
        if self.position:
            self._bars_held += 1
            regime = self._regime

            # ── SWING3 trailing exit (trend regimes) ──────────────────────
            if regime in ('trend_up', 'trend_down'):
                st_cur = float(self.st_dir[-1])
                st_prv = float(self.st_dir[-2]) if len(self.st_dir) > 1 else st_cur
                if not (np.isfinite(st_cur) and np.isfinite(st_prv)):
                    return
                st_flipped_bear = st_prv < 0 and st_cur > 0
                st_flipped_bull = st_prv > 0 and st_cur < 0
                if self.position.is_long and st_flipped_bear:
                    self.position.close()
                    self._reset_trade_state()
                elif self.position.is_short and st_flipped_bull:
                    self.position.close()
                    self._reset_trade_state()

            # ── RR1 in-position management (range regime) ─────────────────
            elif regime == 'range':
                adx_val  = float(self.adx_regime[-1])
                adx_prv  = float(self.adx_regime[-2]) if len(self.adx_regime) > 1 else adx_val
                sma      = float(self.sma20[-1])
                adx_exit = adx_prv <= 25 < adx_val   # ADX crosses above 25 in ranging mode
                time_exit = self._bars_held >= self.max_hold_bars

                if adx_exit or time_exit:
                    self.position.close()
                    self._reset_trade_state()
                    return

                if not self._tp1_hit:
                    if self._is_long and close >= sma:
                        self.position.close(self.tp1_pct / 100.0)
                        self._tp1_hit = True
                    elif not self._is_long and close <= sma:
                        self.position.close(self.tp1_pct / 100.0)
                        self._tp1_hit = True

                if not self.position:
                    self._reset_trade_state()
            return

        # ── Flat — reset state, check for entries ─────────────────────────
        self._reset_trade_state()

        regime = self._regime
        if regime == 'grey':
            return

        # ── SWING3 entry (trend regimes) ──────────────────────────────────
        if regime in ('trend_up', 'trend_down'):
            st_cur  = float(self.st_dir[-1])
            st_prv  = float(self.st_dir[-2]) if len(self.st_dir) > 1 else st_cur
            adx_st  = float(self.adx_st[-1])
            ema_fv  = float(self.ema_f[-1])

            if not all(np.isfinite([st_cur, st_prv, adx_st, ema_fv])):
                return

            st_flipped_bull = st_prv > 0 and st_cur < 0
            st_flipped_bear = st_prv < 0 and st_cur > 0
            adx_strong      = adx_st > self.adx_threshold
            stop_dist       = atr * self.atr_stop_mult
            if stop_dist <= 0:
                return

            size_mult = self._size_multiplier
            risk_amt  = self.equity * self.risk_pct / 100.0 * size_mult

            if regime == 'trend_up' and st_flipped_bull and adx_strong and close > ema_fv:
                qty = risk_amt / stop_dist
                if qty <= 0:
                    return
                self._entry_price     = close
                self._entry_stop_dist = stop_dist
                self._entry_is_long   = True
                self._bars_held       = 0
                self.buy(size=qty, sl=close - stop_dist)

            elif regime == 'trend_down' and st_flipped_bear and adx_strong and close < ema_fv:
                qty = risk_amt / stop_dist
                if qty <= 0:
                    return
                self._entry_price     = close
                self._entry_stop_dist = stop_dist
                self._entry_is_long   = False
                self._bars_held       = 0
                self.sell(size=qty, sl=close + stop_dist)

        # ── RR1 entry (range regime) ──────────────────────────────────────
        elif regime == 'range':
            adx_rr   = float(self.adx_regime[-1])
            rsi      = float(self.rsi[-1])
            sk       = float(self._stoch[0][-1])
            sd       = float(self._stoch[1][-1])
            sk_prv   = float(self._stoch[0][-2]) if len(self._stoch[0]) > 1 else sk
            sd_prv   = float(self._stoch[1][-2]) if len(self._stoch[1]) > 1 else sd
            bb_up    = float(self.bb_upper[-1])
            bb_lo    = float(self.bb_lower[-1])
            r_high   = float(self.range_high[-1])
            r_low    = float(self.range_low[-1])

            if not all(np.isfinite([adx_rr, rsi, sk, sd, bb_up, bb_lo, r_high, r_low])):
                return

            ranging          = adx_rr < self.adx_range
            stoch_cross_up   = sk_prv < sd_prv and sk >= sd and sk < 20 and sd < 20
            stoch_cross_down = sk_prv > sd_prv and sk <= sd and sk > 80 and sd > 80
            size_mult        = self._size_multiplier

            # Long entry
            if ranging and close <= bb_lo and rsi < self.rsi_oversold and stoch_cross_up:
                stop      = r_low - atr * self.sl_buffer_atr
                tp2       = bb_up
                stop_dist = abs(close - stop)
                if stop_dist <= 0:
                    return
                risk_amt = self.equity * self.risk_pct / 100.0 * size_mult
                qty      = risk_amt / stop_dist
                if qty <= 0:
                    return
                self._entry_price     = close
                self._entry_stop_dist = stop_dist
                self._entry_is_long   = True
                self._is_long         = True
                self._tp1_hit         = False
                self._bars_held       = 0
                self.buy(size=qty, sl=stop, tp=tp2)

            # Short entry
            elif ranging and close >= bb_up and rsi > self.rsi_overbought and stoch_cross_down:
                stop      = r_high + atr * self.sl_buffer_atr
                tp2       = bb_lo
                stop_dist = abs(close - stop)
                if stop_dist <= 0:
                    return
                risk_amt = self.equity * self.risk_pct / 100.0 * size_mult
                qty      = risk_amt / stop_dist
                if qty <= 0:
                    return
                self._entry_price     = close
                self._entry_stop_dist = stop_dist
                self._entry_is_long   = False
                self._is_long         = False
                self._tp1_hit         = False
                self._bars_held       = 0
                self.sell(size=qty, sl=stop, tp=tp2)
