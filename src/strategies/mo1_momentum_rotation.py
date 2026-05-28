"""
MO1 — Cross-Asset Momentum Rotation
=====================================
Timeframe:  4H
Direction:  Long and Short

Entry Logic:
  Long:  relative_rsi (RSI(primary,14) - RSI(BTC,14)) > momentum_threshold
         AND ADX(14) > adx_trend_confirm
         AND not in cooldown

  Short: relative_rsi < -momentum_threshold
         AND ADX(14) > adx_trend_confirm
         AND not in cooldown

BTC benchmark RSI is loaded from the database in init() and aligned to the
primary feed's DatetimeIndex. If the primary asset IS BTC (or DB load fails),
relative_rsi degrades gracefully to raw RSI (or 0-based offset).

Stop Loss:  ATR(14) × sl_atr_mult trailing stop (manual, updated each bar)
Exits:      Trailing stop hit
            ADX fades below adx_exit_fade
            max_hold_bars exceeded
            Rotation trigger: relative_rsi crosses zero against position direction
Sizing:     risk_pct % of equity per trade
"""

import numpy as np
import pandas as pd

from ..core.backtesting_engine import BaseStrategy

try:
    import talib

    def _calc_rsi(close, period):
        return pd.Series(
            talib.RSI(pd.Series(close), timeperiod=period)
        ).ffill().bfill().fillna(50)

    def _calc_adx(high, low, close, period):
        return pd.Series(
            talib.ADX(pd.Series(high), pd.Series(low), pd.Series(close), timeperiod=period)
        ).ffill().bfill().fillna(0)

    def _calc_atr(high, low, close, period):
        return pd.Series(
            talib.ATR(pd.Series(high), pd.Series(low), pd.Series(close), timeperiod=period)
        ).ffill().bfill().fillna(0)

except ImportError:
    import ta

    def _calc_rsi(close, period):
        return ta.momentum.rsi(
            pd.Series(close), window=period
        ).ffill().bfill().fillna(50)

    def _calc_adx(high, low, close, period):
        return ta.trend.ADXIndicator(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).adx().ffill().bfill().fillna(0)

    def _calc_atr(high, low, close, period):
        return ta.volatility.average_true_range(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).ffill().bfill().fillna(0)


def _identity(arr):
    """Pass-through for pre-computed arrays registered via self.I()."""
    return pd.Series(arr)


class MO1MomentumRotationStrategy(BaseStrategy):
    """
    MO1 — Cross-asset momentum rotation using RSI relative to BTC benchmark.

    Run this strategy on each asset individually. The BTC RSI is loaded from
    the database and used as a benchmark; relative_rsi = RSI(asset) - RSI(BTC).
    Positive relative momentum → long; negative → short.
    """

    # === Parameters ===
    rsi_period         = 14
    momentum_threshold = 5.0
    adx_period         = 14
    adx_trend_confirm  = 20
    adx_exit_fade      = 15
    atr_stop_mult      = 2.0
    atr_period         = 14
    cooldown_bars      = 10
    max_hold_bars      = 200
    btc_symbol         = 'BTCUSDT'
    risk_pct           = 1.0

    def init(self):
        # Primary asset indicators
        self.rsi = self.I(_calc_rsi, self.data.Close, self.rsi_period)
        self.adx = self.I(_calc_adx, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.atr = self.I(_calc_atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

        # BTC benchmark RSI — load from DB and align to primary feed index
        btc_rsi_arr = self._load_btc_rsi()
        self.btc_rsi = self.I(_identity, btc_rsi_arr)

        # Trade state
        self._reset_trade_state()
        self._bars_since_exit = 9999  # allow immediate entry at start

    def _load_btc_rsi(self) -> np.ndarray:
        """
        Load BTC OHLCV from the database, compute RSI(14), align to the
        primary feed's DatetimeIndex.  Returns a numpy array of length
        len(self.data.Close).  Falls back to zeros on any error (so
        relative_rsi == primary RSI, which is the degenerate BTC-as-primary case).
        """
        n = len(self.data.Close)
        fallback = np.zeros(n)

        # If the primary feed IS BTC, relative_rsi will always be 0 → no signal.
        # That is intentional: BTC only trades if it outperforms itself, which it
        # never does, so this acts as a baseline / sanity-check run.
        try:
            from ..data.database import db  # noqa: PLC0415

            start_dt = self.data.index[0].to_pydatetime()
            end_dt   = self.data.index[-1].to_pydatetime()
            btc_data = db.get_market_data(self.btc_symbol, '4h', start_dt, end_dt)

            if btc_data is None or btc_data.empty or len(btc_data) < 20:
                return fallback

            close_col = 'close' if 'close' in btc_data.columns else 'Close'
            btc_close = btc_data[close_col]

            btc_rsi_raw = _calc_rsi(btc_close, self.rsi_period)
            # btc_rsi_raw may have a RangeIndex; give it the DatetimeIndex from DB
            if not isinstance(btc_rsi_raw.index, pd.DatetimeIndex):
                btc_rsi_raw.index = btc_data.index

            btc_rsi_aligned = (
                btc_rsi_raw
                .reindex(self.data.index)
                .ffill()
                .bfill()
                .fillna(50)
            )
            return btc_rsi_aligned.values

        except Exception:  # DB unavailable, import error, etc.
            return fallback

    def _reset_trade_state(self):
        self._trail_stop      = None
        self._entry_is_long   = None
        self._bars_held       = 0
        self._bars_since_exit = 0

    def next(self):
        if not self.should_trade():
            return

        close = float(self.data.Close[-1])
        atr   = float(self.atr[-1])
        adx   = float(self.adx[-1])
        rsi_p = float(self.rsi[-1])
        rsi_b = float(self.btc_rsi[-1])

        if not all(np.isfinite([close, atr, adx, rsi_p, rsi_b])) or atr <= 0:
            return

        relative_rsi = rsi_p - rsi_b

        # ── In-position management ─────────────────────────────────────────────
        if self.position:
            if self.sl_mode in ('embedded', 'fixed_signal'):
                is_long  = self.position.is_long
                is_short = not is_long

                # Update trailing stop (never loosen)
                if is_long:
                    candidate = close - atr * self.atr_stop_mult
                    if self._trail_stop is None or candidate > self._trail_stop:
                        self._trail_stop = candidate
                else:
                    candidate = close + atr * self.atr_stop_mult
                    if self._trail_stop is None or candidate < self._trail_stop:
                        self._trail_stop = candidate

                # Exit conditions (in priority order)
                exit_reason = None

                # 1. Trailing stop hit
                if self._trail_stop is not None and exit_reason is None:
                    if is_long  and close <= self._trail_stop:
                        exit_reason = "trail_stop_long"
                    elif is_short and close >= self._trail_stop:
                        exit_reason = "trail_stop_short"

                # 2. ADX faded
                if exit_reason is None and adx < self.adx_exit_fade:
                    exit_reason = "adx_fade"

                # 3. Max hold bars
                if exit_reason is None and self._bars_held >= self.max_hold_bars:
                    exit_reason = "max_hold"

                # 4. Rotation trigger: relative_rsi crossed zero against position
                if exit_reason is None:
                    if is_long  and relative_rsi < 0:
                        exit_reason = "rotation_exit_long"
                    elif is_short and relative_rsi > 0:
                        exit_reason = "rotation_exit_short"

                self._bars_held += 1

                if exit_reason:
                    self.position.close()
                    self._reset_trade_state()
            return  # always return after in-position block

        # ── No position — increment cooldown counter ───────────────────────────
        self._bars_since_exit += 1

        # ── Entry conditions ───────────────────────────────────────────────────
        if self._bars_since_exit < self.cooldown_bars:
            return

        strong_trend = adx > self.adx_trend_confirm
        stop_dist    = atr * self.atr_stop_mult
        if stop_dist <= 0:
            return

        # Long entry
        if relative_rsi > self.momentum_threshold and strong_trend:
            sl_price = close - stop_dist
            self.enter_long_position(stop_loss=sl_price, atr_value=atr)
            self._entry_is_long = True
            self._trail_stop    = close - stop_dist
            self._bars_held     = 0

        # Short entry
        elif relative_rsi < -self.momentum_threshold and strong_trend:
            sl_price = close + stop_dist
            self.enter_short_position(stop_loss=sl_price, atr_value=atr)
            self._entry_is_long = False
            self._trail_stop    = close + stop_dist
            self._bars_held     = 0
