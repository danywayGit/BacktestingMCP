"""
SFP1 — Swing Failure Pattern + Fair Value Gap (5m / synthesized 1H)
=====================================================================
Timeframe:  5m primary feed; 1H logic synthesized via resampling
Direction:  Long and Short
Target R:R: 2.0 (fixed) or partial (50% @ 2R, trail rest)

Entry Logic:
  1. Detect a 1H Swing Failure Pattern (SFP) at the top of the hour:
     Long SFP:  current 1H candle sweeps a swing low (low < swing_low)
                AND closes back above it (close > swing_low)
     Short SFP: current 1H candle sweeps a swing high (high > swing_high)
                AND closes back below it (close < swing_high)

  2. Optional bias filter: 1H EMA50/EMA200 cross
     — long SFPs only when EMA50 > EMA200 (bullish bias)
     — short SFPs only when EMA50 < EMA200 (bearish bias)

  3. Optional session filter: NY / London / Asian / Any

  4. Within max_ltf_wait_bars of SFP, scan 5m bars for a Fair Value Gap (FVG):
     Bullish FVG: low[-1] > high[-3]  (gap between candle-3 high and candle-1 low)
     Bearish FVG: high[-1] < low[-3]

  5. Enter at current bar close; SL = FVG middle-candle low/high ± sl_buffer_atr × ATR5m

Stop Loss:   FVG-based (see above)
Take Profit: entry ± stop_dist × rr_ratio
Sizing:      risk_pct % of equity per trade
"""

import numpy as np
import pandas as pd

from ..core.backtesting_engine import BaseStrategy

# ---------------------------------------------------------------------------
# TA library fallback
# ---------------------------------------------------------------------------
try:
    import talib

    def _calc_atr5m(high, low, close, period):
        return pd.Series(
            talib.ATR(pd.Series(high), pd.Series(low), pd.Series(close), timeperiod=period)
        ).ffill().bfill().fillna(0)

    def _calc_ema_series(close_s, period):
        return pd.Series(
            talib.EMA(pd.Series(close_s), timeperiod=period)
        ).ffill().bfill().fillna(method=None)

except ImportError:
    import ta

    def _calc_atr5m(high, low, close, period):
        return ta.volatility.average_true_range(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).ffill().bfill().fillna(0)

    def _calc_ema_series(close_s, period):
        return ta.trend.ema_indicator(pd.Series(close_s), window=period).ffill().bfill()


# ---------------------------------------------------------------------------
# Session windows  (UTC hours)
# ---------------------------------------------------------------------------
SESSION_WINDOWS = {
    'NY':     (14, 30, 20, 0),
    'London': (7,  0,  12, 0),
    'Asian':  (1,  0,  9,  0),
    'Any':    (0,  0,  24, 0),
}


def _in_session(bar_ts, session_mode, buffer_minutes):
    """Return True if bar_ts (UTC) falls within the session window + buffer."""
    if session_mode == 'Any':
        return True
    start_h, start_m, end_h, end_m = SESSION_WINDOWS[session_mode]
    t     = bar_ts.hour * 60 + bar_ts.minute
    start = start_h * 60 + start_m
    end   = end_h   * 60 + end_m
    return start <= t <= min(start + buffer_minutes, end)


# ---------------------------------------------------------------------------
# Module-level swing detection helpers
# ---------------------------------------------------------------------------

def _find_swing_highs(highs, lookback):
    """Return indices where highs[i] is a pivot high with `lookback` bars on each side."""
    result = []
    n = len(highs)
    for i in range(lookback, n - lookback):
        if (all(highs[i] >= highs[i - lookback:i]) and
                all(highs[i] >= highs[i + 1:i + lookback + 1])):
            result.append(i)
    return result


def _find_swing_lows(lows, lookback):
    """Return indices where lows[i] is a pivot low with `lookback` bars on each side."""
    result = []
    n = len(lows)
    for i in range(lookback, n - lookback):
        if (all(lows[i] <= lows[i - lookback:i]) and
                all(lows[i] <= lows[i + 1:i + lookback + 1])):
            result.append(i)
    return result


# ---------------------------------------------------------------------------
# self.I()-compatible helper: resample 5m OHLC → 1H, forward-fill to 5m grid
# ---------------------------------------------------------------------------

def _resample_to_1h(arr_open, arr_high, arr_low, arr_close, index):
    """For each 5m bar, fill with the OHLC of its 1H parent candle.

    Returns shape (4, N) array:
      row 0 = 1h_high
      row 1 = 1h_low
      row 2 = 1h_close
      row 3 = 1h_open
    """
    idx = pd.DatetimeIndex(index)
    df = pd.DataFrame({
        'open':  np.asarray(arr_open,  dtype=float),
        'high':  np.asarray(arr_high,  dtype=float),
        'low':   np.asarray(arr_low,   dtype=float),
        'close': np.asarray(arr_close, dtype=float),
    }, index=idx)

    h1 = df.resample('1h', label='left', closed='left').agg({
        'open':  'first',
        'high':  'max',
        'low':   'min',
        'close': 'last',
    })
    h1_reindexed = h1.reindex(idx, method='ffill')

    return np.vstack([
        h1_reindexed['high'].values,
        h1_reindexed['low'].values,
        h1_reindexed['close'].values,
        h1_reindexed['open'].values,
    ])


# ---------------------------------------------------------------------------
# self.I()-compatible helper: 1H EMA forward-filled to 5m grid
# ---------------------------------------------------------------------------

def _h1_ema(close, index, period):
    """Resample 5m close to 1H, compute EMA(period), forward-fill to 5m index."""
    idx = pd.DatetimeIndex(index)
    close_s = pd.Series(np.asarray(close, dtype=float), index=idx)
    h1_close = close_s.resample('1h', label='left', closed='left').last()
    h1_ema   = _calc_ema_series(h1_close.values, period)
    h1_ema_s = pd.Series(h1_ema.values, index=h1_close.index)
    reindexed = h1_ema_s.reindex(idx, method='ffill')
    return reindexed.values.astype(float)


# ---------------------------------------------------------------------------
# Main strategy class
# ---------------------------------------------------------------------------

class SFP1SwingFailurePatternStrategy(BaseStrategy):
    """
    SFP1 — Swing Failure Pattern with Fair Value Gap entry filter.

    Detects 1H SFPs (sweeps of swing highs/lows that close back inside) via
    synthesized 1H data on a 5m feed, then waits up to max_ltf_wait_bars for a
    bullish/bearish Fair Value Gap (FVG) on the 5m chart before entering.
    """

    # === Parameters ===
    lookback_bars       = 48    # 1H bars to scan for swing points
    swing_lookback      = 5     # pivot width (bars each side)
    session_mode        = 'NY'  # 'NY', 'London', 'Asian', 'Any'
    session_buffer_min  = 120   # minutes after session open to allow entries
    max_ltf_wait_bars   = 24    # max 5m bars to wait for FVG after SFP detected
    risk_pct            = 1.0
    rr_ratio            = 2.0
    sl_buffer_atr       = 0.5
    use_bias_filter     = True
    close_at_session_end = False
    tp_mode             = 'fixed_rr'   # 'fixed_rr' or 'partial'
    atr_period          = 14
    max_hold_bars       = 300   # 300 × 5m ≈ 25 h

    # ------------------------------------------------------------------
    def init(self):
        # Synthesized 1H OHLC arrays (shape 4 × N), forward-filled to 5m index
        # row 0=high, 1=low, 2=close, 3=open
        self._h1 = self.I(
            _resample_to_1h,
            self.data.Open, self.data.High, self.data.Low, self.data.Close,
            self.data.index,
        )

        # 1H EMA 50 / 200 (forward-filled to 5m)
        self._h1_ema50  = self.I(_h1_ema, self.data.Close, self.data.index, 50)
        self._h1_ema200 = self.I(_h1_ema, self.data.Close, self.data.index, 200)

        # 5m ATR for SL buffer
        self.atr5m = self.I(_calc_atr5m, self.data.High, self.data.Low, self.data.Close, self.atr_period)

        # Internal state
        self._sfp_flag_bar   = -9999   # bar index when SFP was confirmed
        self._sfp_is_long    = None    # True = bullish SFP, False = bearish
        self._entry_is_long  = None    # set at entry, cleared on exit detection
        self._bars_held      = 0
        self._tp1_price      = None
        self._tp1_hit        = False
        self._bars_since_exit = 9999

    # ------------------------------------------------------------------
    def _reset_trade_state(self):
        self._entry_is_long  = None
        self._bars_held      = 0
        self._tp1_price      = None
        self._tp1_hit        = False
        self._bars_since_exit = 0

    # ------------------------------------------------------------------
    def _get_1h_window(self, n_bars):
        """Return (highs_1h, lows_1h, close_1h) arrays of the last n_bars 1H candles.

        We identify 1H boundaries as bars where minute == 0, then slice the
        pre-computed forward-filled arrays at those positions.
        """
        idx = pd.DatetimeIndex(self.data.index)
        is_hour = (idx.minute == 0)
        hour_pos = np.where(is_hour)[0]
        if len(hour_pos) < n_bars + 1:
            return None, None, None
        positions = hour_pos[-n_bars:]
        h1_high  = np.array(self._h1[0])[positions]
        h1_low   = np.array(self._h1[1])[positions]
        h1_close = np.array(self._h1[2])[positions]
        return h1_high, h1_low, h1_close

    # ------------------------------------------------------------------
    def next(self):
        if not self.should_trade():
            return

        bar_idx = len(self.data.Close) - 1
        bar_ts  = pd.Timestamp(self.data.index[-1])
        close   = float(self.data.Close[-1])
        atr5m   = float(self.atr5m[-1])

        if not np.isfinite(atr5m) or atr5m <= 0:
            return

        # ── In-position management ────────────────────────────────────────
        if self.position:
            self._bars_held += 1

            # Max hold period
            if self._bars_held >= self.max_hold_bars:
                self.position.close()
                self._reset_trade_state()
                return

            # Session-end close
            if self.close_at_session_end and not _in_session(bar_ts, self.session_mode, self.session_buffer_min):
                self.position.close()
                self._reset_trade_state()
                return

            # TP1 (partial mode): close half at first target
            if (self.tp_mode == 'partial'
                    and not self._tp1_hit
                    and self._tp1_price is not None):
                if (self.position.is_long  and close >= self._tp1_price) or \
                   (self.position.is_short and close <= self._tp1_price):
                    self.position.close(0.5)
                    self._tp1_hit = True

            return

        # ── Framework exit detector ───────────────────────────────────────
        # If we set _entry_is_long but now have no position, the framework
        # closed it via SL/TP.  Reset state.
        if self._entry_is_long is not None:
            self._reset_trade_state()
            return

        # ── Cooldown counter ──────────────────────────────────────────────
        self._bars_since_exit += 1

        # ── Session filter ────────────────────────────────────────────────
        if not _in_session(bar_ts, self.session_mode, self.session_buffer_min):
            # Expire stale SFP flags that were set inside the session
            if (self._sfp_flag_bar > 0
                    and bar_idx - self._sfp_flag_bar > self.max_ltf_wait_bars):
                self._sfp_flag_bar = -9999
                self._sfp_is_long  = None
            return

        # ── 1H SFP detection (only at top of each hour) ───────────────────
        if bar_ts.minute == 0:
            n_needed = self.lookback_bars + self.swing_lookback * 2
            highs_1h, lows_1h, close_1h = self._get_1h_window(n_needed)

            if highs_1h is not None and len(highs_1h) >= self.swing_lookback * 2 + 2:
                swing_lows  = _find_swing_lows(lows_1h,  self.swing_lookback)
                swing_highs = _find_swing_highs(highs_1h, self.swing_lookback)

                ema50  = float(self._h1_ema50[-1])
                ema200 = float(self._h1_ema200[-1])
                bullish_bias = np.isfinite(ema50) and np.isfinite(ema200) and ema50 > ema200

                # Current 1H values (the candle that just closed at bar_ts)
                cur_1h_high  = float(self._h1[0][-1])
                cur_1h_low   = float(self._h1[1][-1])
                cur_1h_close = float(self._h1[2][-1])

                # Bullish SFP: sweep of a recent swing low, closes back above it
                if swing_lows and (not self.use_bias_filter or bullish_bias):
                    last_sw_low = float(lows_1h[swing_lows[-1]])
                    if cur_1h_low < last_sw_low and cur_1h_close > last_sw_low:
                        self._sfp_flag_bar = bar_idx
                        self._sfp_is_long  = True

                # Bearish SFP: sweep of a recent swing high, closes back below it
                if swing_highs and (not self.use_bias_filter or not bullish_bias):
                    last_sw_high = float(highs_1h[swing_highs[-1]])
                    if cur_1h_high > last_sw_high and cur_1h_close < last_sw_high:
                        self._sfp_flag_bar = bar_idx
                        self._sfp_is_long  = False

        # ── FVG scan and entry ────────────────────────────────────────────
        # Abort if no active SFP flag or the wait window has expired
        if self._sfp_flag_bar < 0 or bar_idx - self._sfp_flag_bar > self.max_ltf_wait_bars:
            return
        if self._sfp_is_long is None:
            return

        # Need at least 3 bars for FVG check
        if len(self.data.Close) < 3:
            return

        h0 = float(self.data.High[-3])
        l0 = float(self.data.Low[-3])
        h1 = float(self.data.High[-2])   # middle candle (FVG body)
        l1 = float(self.data.Low[-2])
        h2 = float(self.data.High[-1])
        l2 = float(self.data.Low[-1])

        if self._sfp_is_long:
            # Bullish FVG: gap between candle[-3] high and candle[-1] low
            if l2 > h0:
                sl_price  = l1 - self.sl_buffer_atr * atr5m
                stop_dist = abs(close - sl_price)
                if stop_dist <= 0:
                    return
                qty      = (self.equity * self.risk_pct / 100.0) / stop_dist
                if qty <= 0:
                    return
                tp_price = close + stop_dist * self.rr_ratio
                self._entry_is_long = True
                self._bars_held     = 0
                self._tp1_hit       = False
                self._tp1_price     = (
                    close + stop_dist * self.rr_ratio * 0.5
                    if self.tp_mode == 'partial' else None
                )
                self._sfp_flag_bar  = -9999
                self._sfp_is_long   = None
                self.buy(size=qty, sl=sl_price)

        else:
            # Bearish FVG: gap between candle[-1] high and candle[-3] low
            if h2 < l0:
                sl_price  = h1 + self.sl_buffer_atr * atr5m
                stop_dist = abs(sl_price - close)
                if stop_dist <= 0:
                    return
                qty      = (self.equity * self.risk_pct / 100.0) / stop_dist
                if qty <= 0:
                    return
                self._entry_is_long = False
                self._bars_held     = 0
                self._tp1_hit       = False
                self._tp1_price     = (
                    close - stop_dist * self.rr_ratio * 0.5
                    if self.tp_mode == 'partial' else None
                )
                self._sfp_flag_bar  = -9999
                self._sfp_is_long   = None
                self.sell(size=qty, sl=sl_price)
