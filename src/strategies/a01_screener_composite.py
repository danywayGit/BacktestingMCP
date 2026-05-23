"""
A01 — Screener Composite Strategy
===================================
Timeframe:  4H
Direction:  Long and Short

Overview:
  Combines an external altFINS-style screener signal (short_term_trend) with
  on-chart confirmation filters (EMA, ADX, volume) to gate entries.  The
  screener CSV is loaded once in init(); each bar performs a fast pandas
  look-up to find the most-recent prior screener row.

Entry Logic:
  Long:  short_term_trend == 'BUY'
         AND volume_relative > volume_relative_threshold  (default 100)
         AND close > EMA(ema_structural)
         AND ADX(adx_period) > adx_threshold

  Short: short_term_trend == 'SELL'
         AND volume_relative > volume_relative_threshold
         AND close < EMA(ema_structural)
         AND ADX(adx_period) > adx_threshold

Stop Loss:   ATR(atr_period) × sl_atr_mult (absolute price)
Take Profit: Two-target system
  TP1 (50 % of position) @ entry ± ATR × sl_atr_mult × tp1_rr
  TP2 (remainder)        @ entry ± ATR × sl_atr_mult × tp2_rr

Additional Exits:
  • Trend reversal: screener flips to opposite or NEUTRAL
  • max_hold_bars exceeded

Sizing:      risk_pct % of equity per trade

Mock Data:
  If `src/data/screener_mock.csv` is absent, init() auto-generates it via
  _generate_screener_mock() using a deterministic seeded-random pattern that
  alternates BUY/SELL/NEUTRAL blocks over the full backtest date range.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from ..core.backtesting_engine import BaseStrategy

# ── Indicator helpers ────────────────────────────────────────────────────────

try:
    import talib

    def _calc_ema(close, period):
        return pd.Series(
            talib.EMA(pd.Series(close), timeperiod=period)
        ).ffill().bfill().fillna(0)

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

    def _calc_ema(close, period):
        return ta.trend.ema_indicator(
            close=pd.Series(close), window=period
        ).ffill().bfill().fillna(0)

    def _calc_adx(high, low, close, period):
        return ta.trend.ADXIndicator(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).adx().ffill().bfill().fillna(0)

    def _calc_atr(high, low, close, period):
        return ta.volatility.average_true_range(
            high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=period
        ).ffill().bfill().fillna(0)


# ── Mock screener generator ──────────────────────────────────────────────────

def _generate_screener_mock(path: Path, index) -> None:
    """
    Generate a deterministic mock screener CSV covering every timestamp in
    `index`.  Trend blocks of 12–60 bars alternate between BUY / SELL /
    NEUTRAL; volume_relative varies uniformly in [70, 160].

    This is called automatically by A01ScreenerCompositeStrategy.init() when
    `screener_mock.csv` does not exist.  Seed is fixed (42) so results are
    reproducible across runs.
    """
    import random
    rng = random.Random(42)
    trend_options = ["BUY", "SELL", "NEUTRAL"]
    current_trend = "NEUTRAL"
    trend_run = 0
    rows = []
    for ts in index:
        trend_run -= 1
        if trend_run <= 0:
            current_trend = rng.choice(trend_options)
            trend_run = rng.randint(12, 60)
        vol_rel = rng.uniform(70, 160)
        rows.append(
            {
                "timestamp": ts.isoformat(),
                "symbol": "BTCUSDT",
                "short_term_trend": current_trend,
                "volume_relative": round(vol_rel, 1),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


# ── Strategy class ────────────────────────────────────────────────────────────

class A01ScreenerCompositeStrategy(BaseStrategy):
    """
    A01 — altFINS Screener Composite
    Blends an external trend signal with EMA / ADX confirmation and
    ATR-based two-target exits.
    """

    # === Parameters ===
    volume_relative_threshold: float = 100.0   # screener vol_relative must exceed this
    ema_structural: int   = 50                  # structural EMA period
    adx_threshold:  float = 20.0               # minimum ADX for entry
    adx_period:     int   = 14
    atr_period:     int   = 14
    sl_atr_mult:    float = 2.0                # stop distance = ATR × sl_atr_mult
    tp1_rr:         float = 1.0                # TP1 R:R (50 % close)
    tp2_rr:         float = 2.0                # TP2 R:R (remainder)
    max_hold_bars:  int   = 200
    risk_pct:       float = 1.0                # % equity risked per trade

    # ── init ──────────────────────────────────────────────────────────────────
    def init(self):
        # Vectorised indicators
        self.ema_s = self.I(_calc_ema, self.data.Close, self.ema_structural)
        self.adx   = self.I(_calc_adx, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.atr   = self.I(_calc_atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

        # ── Load / generate screener data ─────────────────────────────────────
        mock_path = Path(__file__).parent.parent / "data" / "screener_mock.csv"
        if not mock_path.exists():
            _generate_screener_mock(mock_path, self.data.index)

        screener_df = pd.read_csv(mock_path, parse_dates=["timestamp"])
        screener_df["timestamp"] = pd.to_datetime(screener_df["timestamp"], utc=True)
        screener_df.set_index("timestamp", inplace=True)
        screener_df.sort_index(inplace=True)
        self._screener_df = screener_df

        # ── Trade state ───────────────────────────────────────────────────────
        self._bars_held:    int   = 0
        self._entry_is_long: bool = False
        self._tp1_price:    float = 0.0
        self._tp2_price:    float = 0.0
        self._tp1_hit:      bool  = False

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _reset_trade_state(self) -> None:
        self._bars_held    = 0
        self._entry_is_long = False
        self._tp1_price    = 0.0
        self._tp2_price    = 0.0
        self._tp1_hit      = False

    def _get_screener_row(self):
        """Return the most-recent screener row at or before the current bar."""
        bar_ts = self.data.index[-1]
        # Ensure tz-aware comparison
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("UTC")
        mask = self._screener_df.index <= bar_ts
        if not mask.any():
            return None
        return self._screener_df[mask].iloc[-1]

    # ── next ─────────────────────────────────────────────────────────────────
    def next(self):
        if not self.should_trade():
            return

        close = float(self.data.Close[-1])
        high  = float(self.data.High[-1])
        low   = float(self.data.Low[-1])
        ema_v = float(self.ema_s[-1])
        adx_v = float(self.adx[-1])
        atr_v = float(self.atr[-1])

        # NaN / sanity guard
        if not all(np.isfinite([close, ema_v, adx_v, atr_v])) or atr_v <= 0:
            return

        # Screener lookup
        row = self._get_screener_row()
        if row is None:
            return
        trend_signal  = str(row["short_term_trend"])
        vol_relative  = float(row["volume_relative"])

        # ── In-position management ────────────────────────────────────────────
        if self.position:
            self._bars_held += 1
            is_long  = self.position.is_long
            is_short = self.position.is_short

            # TP1 partial close (50 %)
            if not self._tp1_hit:
                if (is_long  and high  >= self._tp1_price) or \
                   (is_short and low   <= self._tp1_price):
                    self.position.close(0.5)
                    self._tp1_hit = True
                    return

            # TP2 full close
            if self._tp1_hit:
                if (is_long  and high  >= self._tp2_price) or \
                   (is_short and low   <= self._tp2_price):
                    self.position.close()
                    self._reset_trade_state()
                    return

            # Signal reversal exit
            if is_long  and trend_signal in ("SELL", "NEUTRAL"):
                self.position.close()
                self._reset_trade_state()
                return
            if is_short and trend_signal in ("BUY", "NEUTRAL"):
                self.position.close()
                self._reset_trade_state()
                return

            # Max hold bars
            if self._bars_held >= self.max_hold_bars:
                self.position.close()
                self._reset_trade_state()
            return

        # ── Entry ─────────────────────────────────────────────────────────────
        high_vol      = vol_relative > self.volume_relative_threshold
        strong_trend  = adx_v > self.adx_threshold
        stop_dist     = atr_v * self.sl_atr_mult

        # Compute risk-based position size
        qty = (self.equity * self.risk_pct / 100.0) / stop_dist
        if qty <= 0:
            return

        # Long entry
        if (trend_signal == "BUY"
                and high_vol
                and close > ema_v
                and strong_trend):
            sl_price  = close - stop_dist
            tp1_price = close + stop_dist * self.tp1_rr
            tp2_price = close + stop_dist * self.tp2_rr
            self.buy(size=qty, sl=sl_price)
            self._entry_is_long = True
            self._tp1_price     = tp1_price
            self._tp2_price     = tp2_price
            self._tp1_hit       = False
            self._bars_held     = 0

        # Short entry
        elif (trend_signal == "SELL"
                and high_vol
                and close < ema_v
                and strong_trend):
            sl_price  = close + stop_dist
            tp1_price = close - stop_dist * self.tp1_rr
            tp2_price = close - stop_dist * self.tp2_rr
            self.sell(size=qty, sl=sl_price)
            self._entry_is_long = False
            self._tp1_price     = tp1_price
            self._tp2_price     = tp2_price
            self._tp1_hit       = False
            self._bars_held     = 0
