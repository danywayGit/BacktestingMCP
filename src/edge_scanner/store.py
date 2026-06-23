"""Persist composite-scanner signals and resolve their forward outcomes.

This is the self-validation loop: every signal the scanner flags gets
logged with an entry price, then checked again after `horizon_hours` to
see whether price actually moved the predicted direction. Without this,
the composite score is just an opinion -- this is what turns it into a
measured win-rate.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd

from config.settings import TimeFrame
from ..core.backtesting_engine import engine
from ..data.database import db
from .composite import CandidateScore

logger = logging.getLogger(__name__)

DEFAULT_HORIZON_HOURS = 24
# Forward moves smaller than this are treated as noise, not a real win/loss.
MIN_MOVE_PCT = 0.3


def log_signals(scores: List[CandidateScore], timeframe: TimeFrame, horizon_hours: int = DEFAULT_HORIZON_HOURS) -> int:
    """Persist the actionable (LONG/SHORT) signals from a scan cycle."""
    logged = 0
    for score in scores:
        if score.direction is None or score.last_close is None:
            continue
        db.insert_edge_signal(
            symbol=score.symbol,
            pair=score.pair,
            timeframe=timeframe.value,
            direction=score.direction,
            composite_score=score.composite_score,
            components=score.components,
            entry_price=score.last_close,
            horizon_hours=horizon_hours,
        )
        logged += 1
    return logged


def _fetch_current_price(pair: str, timeframe: TimeFrame) -> float:
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=5)
    data = engine.get_data(pair, timeframe, start_date, end_date)
    if data.empty:
        raise ValueError(f"No recent data available for {pair}")
    return float(data["Close"].iloc[-1])


def resolve_due_signals() -> int:
    """Resolve every PENDING signal whose horizon has elapsed."""
    resolved = 0
    now = datetime.now(timezone.utc)
    for signal in db.get_pending_edge_signals():
        entry_time = datetime.fromisoformat(signal["entry_time"])
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        due_at = entry_time + timedelta(hours=signal["horizon_hours"])
        if now < due_at:
            continue

        try:
            exit_price = _fetch_current_price(signal["pair"], TimeFrame(signal["timeframe"]))
        except Exception as exc:  # noqa: BLE001 - missing data shouldn't block other resolutions
            logger.warning("Could not resolve signal %s (%s): %s", signal["id"], signal["pair"], exc)
            continue

        raw_return_pct = (exit_price - signal["entry_price"]) / signal["entry_price"] * 100
        directional_return_pct = raw_return_pct if signal["direction"] == "LONG" else -raw_return_pct

        if directional_return_pct > MIN_MOVE_PCT:
            outcome = "WIN"
        elif directional_return_pct < -MIN_MOVE_PCT:
            outcome = "LOSS"
        else:
            outcome = "FLAT"

        db.resolve_edge_signal(signal["id"], exit_price, directional_return_pct, outcome)
        resolved += 1
    return resolved


def performance_report(group_by: str = "symbol", min_n: int = 5, since_days: int = 90) -> pd.DataFrame:
    """Win-rate / avg return grouped by symbol, hour-of-day, or direction.

    Mirrors the win-rate-by-segment approach BackTestingSignals uses for
    Telegram/Discord calls, applied to the edge scanner's own signals.
    """
    since = datetime.now(timezone.utc) - timedelta(days=since_days)
    rows = db.get_resolved_edge_signals(since=since)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)

    if group_by == "hour":
        df["group"] = df["entry_time"].dt.hour
    elif group_by == "direction":
        df["group"] = df["direction"]
    else:
        df["group"] = df["symbol"]

    df["is_win"] = (df["outcome"] == "WIN").astype(int)
    summary = (
        df.groupby("group")
        .agg(n=("outcome", "size"), win_rate=("is_win", "mean"), avg_return_pct=("forward_return_pct", "mean"))
        .reset_index()
    )
    summary["win_rate"] = (summary["win_rate"] * 100).round(1)
    summary["avg_return_pct"] = summary["avg_return_pct"].round(3)
    summary = summary[summary["n"] >= min_n].sort_values("win_rate", ascending=False)
    return summary
