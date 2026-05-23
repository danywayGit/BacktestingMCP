"""
EC1 — Event Catalyst Strategy
==============================
Timeframe:  4H
Direction:  Long and Short
Edge:       Purely event-driven — only trades inside post-event windows.

Phases:
  Pre-event  : Event within `event_lookahead_hours` of current bar.
               → No new entries; close position early if profitable (protect gains).
  Post-event : Bar falls between [event + wait_td, event + wait_td + window_td].
               → Enter in EMA50 direction, wider ATR stop.
  Normal     : No event context → do nothing.

Entry:
  Direction determined by close vs EMA(50), cross-checked with event direction_hint.
  Stop:   ATR(14) × post_event_sl_mult  (default 3.0 — wider for event volatility)
  TP1:    entry ± stop_dist × post_event_tp_rr (default 1.5)
  Size:   (equity × risk_pct_post / 100) / stop_dist

In-position management:
  - Close 50% at TP1, then trail remainder with ATR × trail_mult.
  - Trail only tightens — never loosens.
  - Max hold: post_event_max_bars bars.
  - Pre-event tighten: close entire position if profitable.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.backtesting_engine import BaseStrategy

# ── Indicator helpers ────────────────────────────────────────────────────────

try:
    import talib

    def _calc_ema(close, period):
        return (
            pd.Series(talib.EMA(pd.Series(close), timeperiod=period))
            .ffill()
            .bfill()
            .fillna(0)
        )

    def _calc_atr(high, low, close, period):
        return (
            pd.Series(
                talib.ATR(
                    pd.Series(high), pd.Series(low), pd.Series(close), timeperiod=period
                )
            )
            .ffill()
            .bfill()
            .fillna(0)
        )

except ImportError:
    import ta

    def _calc_ema(close, period):
        return (
            ta.trend.ema_indicator(pd.Series(close), window=period)
            .ffill()
            .bfill()
            .fillna(0)
        )

    def _calc_atr(high, low, close, period):
        return (
            ta.volatility.average_true_range(
                high=pd.Series(high),
                low=pd.Series(low),
                close=pd.Series(close),
                window=period,
            )
            .ffill()
            .bfill()
            .fillna(0)
        )


# ── Strategy ─────────────────────────────────────────────────────────────────

class EC1EventCatalystStrategy(BaseStrategy):
    """
    EC1 — Event Catalyst strategy.
    Entries are gated exclusively by high-impact on-chain / macro events loaded
    from ``src/data/events_mock.csv``.
    """

    # === Parameters ===
    event_impact_min        = 7      # minimum impact_score to consider
    event_lookahead_hours   = 48     # hours before event to enter pre-event mode
    post_event_wait_bars    = 4      # bars (×4H) after event before entering
    post_event_window_bars  = 20     # bars the post-event window stays open
    entry_ema               = 50     # EMA period for directional filter
    pre_event_sl_mult       = 1.0    # (unused for re-entry; used for tighten logic)
    post_event_sl_mult      = 3.0    # ATR multiple for initial stop (wider)
    post_event_tp_rr        = 1.5    # TP1 = stop_dist × this ratio
    trail_mult              = 2.0    # ATR multiple for trailing stop
    risk_pct_post           = 0.5    # % equity risked per event trade
    post_event_max_bars     = 150    # force-close after this many bars
    atr_period              = 14
    risk_pct                = 1.0    # kept for BaseStrategy compatibility

    # ── init ──────────────────────────────────────────────────────────────────

    def init(self):
        self.ema50 = self.I(_calc_ema, self.data.Close, self.entry_ema)
        self.atr   = self.I(_calc_atr, self.data.High, self.data.Low,
                            self.data.Close, self.atr_period)

        # Load event catalogue
        events_path = Path(__file__).parent.parent / "data" / "events_mock.csv"
        if events_path.exists():
            ev_df = pd.read_csv(events_path, parse_dates=["event_timestamp"])
            ev_df["event_timestamp"] = pd.to_datetime(
                ev_df["event_timestamp"], utc=True
            )
            ev_df = ev_df[
                ev_df["impact_score"] >= self.event_impact_min
            ].sort_values("event_timestamp")
            self._events = ev_df
        else:
            self._events = pd.DataFrame(
                columns=[
                    "event_timestamp",
                    "symbol",
                    "impact_score",
                    "event_type",
                    "direction_hint",
                ]
            )

        # Per-trade state
        self._bars_held    = 0
        self._entry_is_long = True
        self._tp1_hit      = False
        self._tp1_price    = None
        self._trail_stop   = None
        self._entry_price  = None
        self._stop_dist    = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _reset_trade_state(self):
        self._bars_held     = 0
        self._entry_is_long = True
        self._tp1_hit       = False
        self._tp1_price     = None
        self._trail_stop    = None
        self._entry_price   = None
        self._stop_dist     = None

    def _event_context(self, bar_ts):
        """Return (pre_event_active, post_event_rows) for the current bar."""
        lookahead_td = pd.Timedelta(hours=self.event_lookahead_hours)
        wait_td      = pd.Timedelta(hours=self.post_event_wait_bars * 4)
        window_td    = pd.Timedelta(hours=self.post_event_window_bars * 4)

        pre_event_active = any(
            (bar_ts <= ev_ts) and (ev_ts - bar_ts <= lookahead_td)
            for ev_ts in self._events["event_timestamp"]
        )

        post_event_rows = self._events[
            (self._events["event_timestamp"] <= bar_ts - wait_td)
            & (self._events["event_timestamp"] >= bar_ts - wait_td - window_td)
        ]
        post_event_active = len(post_event_rows) > 0

        return pre_event_active, post_event_active, post_event_rows

    # ── next ──────────────────────────────────────────────────────────────────

    def next(self):
        if not self.should_trade():
            return

        close = float(self.data.Close[-1])
        atr   = float(self.atr[-1])
        ema   = float(self.ema50[-1])

        # NaN / zero guard
        if not all(np.isfinite([close, atr, ema])) or atr <= 0:
            return

        bar_ts = self.data.index[-1]
        # Ensure bar_ts is tz-aware UTC for comparison with event timestamps
        if hasattr(bar_ts, "tzinfo") and bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("UTC")

        pre_event_active, post_event_active, post_event_rows = self._event_context(
            bar_ts
        )

        # ── In-position management ────────────────────────────────────────────
        if self.position:
            is_long  = self.position.is_long
            is_short = not is_long
            self._bars_held += 1

            # 1. Pre-event tighten: close profitable positions before the event
            if pre_event_active:
                if self._entry_price is not None:
                    pnl = (
                        close - self._entry_price
                        if is_long
                        else self._entry_price - close
                    )
                    if pnl > 0:
                        self.position.close()
                        self._reset_trade_state()
                        return

            # 2. Update trailing stop (only after TP1 hit)
            if self._tp1_hit and self._trail_stop is not None:
                if is_long:
                    new_trail = close - atr * self.trail_mult
                    if new_trail > self._trail_stop:
                        self._trail_stop = new_trail
                else:
                    new_trail = close + atr * self.trail_mult
                    if new_trail < self._trail_stop:
                        self._trail_stop = new_trail

            # 3. Check TP1 (first time price reaches TP1)
            if not self._tp1_hit and self._tp1_price is not None:
                tp1_reached = (
                    (is_long  and close >= self._tp1_price)
                    or (is_short and close <= self._tp1_price)
                )
                if tp1_reached:
                    self._tp1_hit  = True
                    # Initialise trail from current price
                    self._trail_stop = (
                        close - atr * self.trail_mult
                        if is_long
                        else close + atr * self.trail_mult
                    )

            # 4. Trail stop hit
            if self._tp1_hit and self._trail_stop is not None:
                trail_hit = (
                    (is_long  and close <= self._trail_stop)
                    or (is_short and close >= self._trail_stop)
                )
                if trail_hit:
                    self.position.close()
                    self._reset_trade_state()
                    return

            # 5. Initial stop guard (before TP1 hit)
            if (
                not self._tp1_hit
                and self._entry_price is not None
                and self._stop_dist is not None
            ):
                initial_stop_hit = (
                    (is_long  and close <= self._entry_price - self._stop_dist)
                    or (is_short and close >= self._entry_price + self._stop_dist)
                )
                if initial_stop_hit:
                    self.position.close()
                    self._reset_trade_state()
                    return

            # 6. Max holding period
            if self._bars_held >= self.post_event_max_bars:
                self.position.close()
                self._reset_trade_state()
            return

        # ── No position — only trade in post-event window ─────────────────────
        if not post_event_active:
            return

        # Pre-event mode supersedes post-event entry (don't open into an event)
        if pre_event_active:
            return

        # Determine direction from EMA50
        bullish = close > ema

        # Cross-check with the most recent event's direction_hint (tiebreaker)
        if len(post_event_rows) > 0:
            latest_hint = post_event_rows.iloc[-1]["direction_hint"]
            if latest_hint == "bullish":
                bullish = True
            elif latest_hint == "bearish":
                bullish = False
            # "neutral" → keep EMA decision

        stop_dist = atr * self.post_event_sl_mult
        if stop_dist <= 0:
            return

        qty = (self.equity * (self.risk_pct_post / 100.0)) / stop_dist
        if qty <= 0:
            return

        if bullish:
            sl_price = close - stop_dist
            tp_price = close + stop_dist * self.post_event_tp_rr
            self.enter_long_position(stop_loss=sl_price, take_profit=tp_price)
            self._entry_is_long = True
        else:
            sl_price = close + stop_dist
            tp_price = close - stop_dist * self.post_event_tp_rr
            self.enter_short_position(stop_loss=sl_price, take_profit=tp_price)
            self._entry_is_long = False

        self._entry_price = close
        self._stop_dist   = stop_dist
        self._tp1_price   = tp_price  # TP1 == full TP (backtesting.py handles it)
        self._tp1_hit     = False
        self._trail_stop  = None
        self._bars_held   = 0
