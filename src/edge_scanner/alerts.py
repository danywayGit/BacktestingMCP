"""
Telegram alert sender — proven data only, no guessing.
If resolved signal data exists for a symbol, shows win-rate stats.
If ATR/OHLCV data exists on Binance, shows volatility-based stop & target.
Otherwise shows N/A.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import httpx
import pandas as pd
from dotenv import load_dotenv

from .composite import CandidateScore

load_dotenv()

logger = logging.getLogger(__name__)

TELEGRAM_CHANNEL_ID = -1001482338614  # @CryptoAlertsTradingView
TELEGRAM_API_URL = "https://api.telegram.org"

ALERT_MIN_SCORE = 7.0
ALERT_MULTI_SOURCE = True

# Default risk parameters (only used when actual ATR data exists)
RR_RATIO = 2.0        # risk 1 → reward 2
ATR_MULT_STOP = 1.5   # stop = ATR × 1.5


def _get_atr(symbol: str) -> Optional[float]:
    """Fetch ATR(14) from existing OHLCV data. Returns None if unavailable."""
    try:
        from ..core.backtesting_engine import engine
        from config.settings import TimeFrame
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)
        data = engine.get_data(f"{symbol}USDT", TimeFrame.H1, start, end)
        if data.empty or len(data) < 20:
            return None
        high, low, close = data["High"], data["Low"], data["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else None
    except Exception:
        return None


def _get_winrate(symbol: str) -> Optional[dict]:
    """Fetch resolved win-rate for this symbol. Returns None if insufficient data."""
    try:
        from ..data.database import db
        from datetime import datetime, timezone, timedelta
        since = datetime.now(timezone.utc) - timedelta(days=90)
        signals = db.get_resolved_edge_signals(since=since)
        relevant = [s for s in signals if s.get("symbol") == symbol.upper()]
        if len(relevant) < 5:
            return None
        wins = sum(1 for s in relevant if s.get("outcome") == "WIN")
        totals = len(relevant)
        avg_ret = sum(s.get("forward_return_pct", 0) for s in relevant) / totals
        return {"n": totals, "wr": round(wins / totals * 100, 1), "avg_return": round(avg_ret, 2)}
    except Exception:
        return None


def _is_multi_source(c: CandidateScore) -> bool:
    sources = 0
    comp = c.components
    trend = comp.get("altfins_trend_score", 0)
    if c.direction == "LONG" and trend >= 7:
        sources += 1
    elif c.direction == "SHORT" and trend <= -7:
        sources += 1
    feed = comp.get("altfins_signal_feed")
    if c.direction == "LONG" and feed == "BULLISH":
        sources += 1
    elif c.direction == "SHORT" and feed == "BEARISH":
        sources += 1
    vol = comp.get("altfins_volume_relative", 1.0)
    if vol >= 2.0:
        sources += 1
    if comp.get("backtestingmcp_scanner_hits"):
        sources += 1
    netflow = comp.get("onchain_netflow_ratio")
    if netflow is not None:
        if c.direction == "LONG" and netflow > 0.05:
            sources += 1
        elif c.direction == "SHORT" and netflow < -0.05:
            sources += 1
    return sources >= 2


def _format_alert(c: CandidateScore) -> str:
    """Alert with proven data only. N/A where no data exists."""
    direction_emoji = "🟢" if c.direction == "LONG" else "🔴"
    comp = c.components

    # Entry price
    price_str = f"${c.last_close:.4f}" if c.last_close else "N/A"

    # ATR stop & target (real OHLCV data if available)
    atr = _get_atr(c.symbol)
    if atr and c.last_close and atr > 0:
        stop_distance = atr * ATR_MULT_STOP
        if c.direction == "LONG":
            stop = c.last_close - stop_distance
            target = c.last_close + stop_distance * RR_RATIO
        else:
            stop = c.last_close + stop_distance
            target = c.last_close - stop_distance * RR_RATIO
        stop_str = f"${stop:.4f}"
        target_str = f"${target:.4f}"
        rr_str = f"1:{RR_RATIO}"
    else:
        stop_str = "N/A"
        target_str = "N/A"
        rr_str = "N/A"

    # Hit rate (resolved signal data)
    wr = _get_winrate(c.symbol)
    if wr:
        wr_str = f"{wr['wr']}% ({wr['n']} trades, avg {wr['avg_return']:+.2f}%)"
    else:
        wr_str = "N/A (< 5 resolved trades)"

    # Build time label
    timeframe_label = comp.get("timeframe", "1h")
    coin_type = comp.get("coin_type", c.coin_type)

    # Source tags — compact for alert
    source_parts = []
    trend = comp.get("altfins_trend_score", 0)
    if abs(trend) >= 7:
        source_parts.append(f"Trend {'+' if trend > 0 else ''}{trend:.0f}")
    feed = comp.get("altfins_signal_feed")
    if feed:
        source_parts.append(f"Signal {feed.title()}")
    vol = comp.get("altfins_volume_relative", 1.0)
    if vol >= 1.5:
        source_parts.append(f"Vol {vol:.1f}x")
    scanner_hits = comp.get("backtestingmcp_scanner_hits", [])
    if scanner_hits:
        source_parts.append("TA breakout")
    netflow = comp.get("onchain_netflow_ratio")
    if netflow is not None:
        source_parts.append("On-chain")

    sources_str = " · ".join(source_parts) if source_parts else "—"

    lines = [
        f"{direction_emoji} *{c.symbol}* — {c.direction} ({c.config_version})",
        f"┌ Entry: `{price_str}`",
        f"├ Stop:  `{stop_str}`",
        f"├ 🎯 Tgt: `{target_str}`  R:R `{rr_str}`",
        f"└ Time:  `{timeframe_label}` · Type: `{coin_type}`",
        f"Score: `{c.composite_score:+.2f}`  |  {sources_str}",
        f"Resolved: {wr_str}",
    ]
    return "\n".join(lines)


def send_alerts(candidates: List[CandidateScore], dry_run: bool = False) -> int:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN not set — skipping alerts")
        return 0

    triggered = [
        c for c in candidates
        if c.direction is not None
        and abs(c.composite_score) >= ALERT_MIN_SCORE
        and (not ALERT_MULTI_SOURCE or _is_multi_source(c))
    ]

    if not triggered:
        return 0

    sent = 0
    for c in triggered:
        message = _format_alert(c)
        if dry_run:
            logger.info("DRY RUN alert:\n%s", message)
            sent += 1
            continue
        try:
            resp = httpx.post(
                f"{TELEGRAM_API_URL}/bot{bot_token}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHANNEL_ID,
                    "text": message,
                    "parse_mode": "Markdown",
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            sent += 1
            logger.info("Alert sent for %s (score=%+.2f)", c.symbol, c.composite_score)
        except Exception as exc:
            logger.error("Failed to send alert for %s: %s", c.symbol, exc)

    return sent
