"""
Telegram alert sender for high-confidence edge scanner signals.

Sends to the configured Telegram channel when a CandidateScore crosses
the alert threshold and has multi-source confirmation.

Requires TELEGRAM_BOT_TOKEN in the environment (loaded from .env).
Channel ID is hardcoded to the user's Crypto Alerts channel.
"""
from __future__ import annotations

import logging
import os
from typing import List

import httpx
from dotenv import load_dotenv

from .composite import CandidateScore

load_dotenv()

logger = logging.getLogger(__name__)

TELEGRAM_CHANNEL_ID = -1001482338614  # @CryptoAlertsTradingView
TELEGRAM_API_URL = "https://api.telegram.org"

# Alert thresholds
ALERT_MIN_SCORE = 7.0          # composite score abs value
ALERT_MULTI_SOURCE = True      # require at least 2 signal sources to agree


def _is_multi_source(c: CandidateScore) -> bool:
    """Return True if at least 2 independent sources confirm the direction."""
    sources = 0
    comp = c.components

    # Source 1: altFINS trend (strong = ±7+)
    trend = comp.get("altfins_trend_score", 0)
    if c.direction == "LONG" and trend >= 7:
        sources += 1
    elif c.direction == "SHORT" and trend <= -7:
        sources += 1

    # Source 2: altFINS signal feed
    feed = comp.get("altfins_signal_feed")
    if c.direction == "LONG" and feed == "BULLISH":
        sources += 1
    elif c.direction == "SHORT" and feed == "BEARISH":
        sources += 1

    # Source 3: high relative volume (>2x average = unusual activity)
    vol = comp.get("altfins_volume_relative", 1.0)
    if vol >= 2.0:
        sources += 1

    # Source 4: TA scanner hit
    if comp.get("backtestingmcp_scanner_hits"):
        sources += 1

    # Source 5: on-chain confirms direction
    netflow = comp.get("onchain_netflow_ratio")
    if netflow is not None:
        if c.direction == "LONG" and netflow > 0.05:
            sources += 1
        elif c.direction == "SHORT" and netflow < -0.05:
            sources += 1

    return sources >= 2


def _format_alert(c: CandidateScore) -> str:
    """Format a CandidateScore into a Telegram message."""
    direction_emoji = "🟢" if c.direction == "LONG" else "🔴"
    comp = c.components

    # Build source tags
    sources = []
    trend = comp.get("altfins_trend_score", 0)
    if abs(trend) >= 7:
        sources.append(f"Trend {'+' if trend > 0 else ''}{trend:.0f}/10")
    feed = comp.get("altfins_signal_feed")
    if feed:
        sources.append(f"Signal feed {feed.title()}")
    scanner_hits = comp.get("backtestingmcp_scanner_hits", [])
    if scanner_hits:
        sources.append(f"TA: {', '.join(scanner_hits)}")
    netflow = comp.get("onchain_netflow_ratio")
    if netflow is not None:
        bias = "accum." if netflow > 0 else "sell-press."
        sources.append(f"On-chain {bias} ({netflow:+.2f})")

    price_str = f"${c.last_close:.4f}" if c.last_close else "N/A"

    lines = [
        f"{direction_emoji} *{c.symbol}* — {c.direction}",
        f"Score: `{c.composite_score:+.2f}` | Price: `{price_str}`",
        f"Sources: {' · '.join(sources) if sources else 'altFINS screener'}",
        f"Pair: `{c.pair}` (Binance Futures)",
    ]
    return "\n".join(lines)


def send_alerts(candidates: List[CandidateScore], dry_run: bool = False) -> int:
    """Send Telegram alerts for candidates that cross the threshold.

    Returns the number of alerts sent.
    """
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
