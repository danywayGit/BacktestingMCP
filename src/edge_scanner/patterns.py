"""
Chart Pattern Scanner — detects technical chart patterns via altFINS signal feed.

Patterns scanned:
  - Falling Wedge (breakout up)
  - Rising Wedge (breakout down)
  - Channel Up (bullish continuation)
  - Channel Down (bearish continuation)
  - Pattern Breakouts (generic)
  - Near support/resistance

Each pattern generates a lightweight alertable signal with:
  - Symbol + direction + pattern name
  - Price at pattern detection
  - Pattern confidence / source confirmation

Only patterns confirmed by live altFINS data are shown.
No guessing — if the pattern isn't in the signal feed, it's not shown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..integrations import altfins_client
from ..integrations.altfins_client import AltfinsError, get_signal_feed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chart pattern signal types available in altFINS
# Verified against the live signal_feed_data tool schema
# ---------------------------------------------------------------------------

PATTERN_SIGNAL_TYPES: Dict[str, str] = {
    # Wedge patterns
    "SIGNALS_SUMMARY_RISING_WEDGE": "Rising Wedge",
    "SIGNALS_SUMMARY_FALLING_WEDGE": "Falling Wedge",
    # Channel patterns
    "SIGNALS_SUMMARY_CHANNEL_UP": "Channel Up",
    "SIGNALS_SUMMARY_CHANNEL_DOWN": "Channel Down",
    # Breakout patterns
    "SIGNALS_SUMMARY_PATTERN_BREAKOUTS": "Pattern Breakout",
    "SIGNALS_SUMMARY_PATTERN_BREAKOUTS_UPTREND_DOWNTREND": "Breakout (Uptrend/Downtrend)",
    "SIGNALS_SUMMARY_EMERGING_PATTERNS": "Emerging Pattern",
    # Triangle patterns
    "SIGNALS_SUMMARY_ASCENDING_TRIANGLE": "Ascending Triangle",
    "SIGNALS_SUMMARY_DESCENDING_TRIANGLE": "Descending Triangle",
    "SIGNALS_SUMMARY_TRIANGLE": "Triangle",
    # Support/resistance
    "SUPPORT_RESISTANCE_APPROACHING": "Near Resistance/Support",
    "SUPPORT_RESISTANCE_APPROACHING_OVERSOLD": "Near Support (Oversold)",
    "SUPPORT_RESISTANCE_BREAKOUT": "Support/Resistance Breakout",
    # Other chart patterns
    "SIGNALS_SUMMARY_DOUBLE_TOP": "Double Top",
    "SIGNALS_SUMMARY_DOUBLE_BOTTOM": "Double Bottom",
    "SIGNALS_SUMMARY_HEAD_AND_SHOULDERS": "Head & Shoulders",
    "SIGNALS_SUMMARY_INVERSE_HEAD_AND_SHOULDERS": "Inverse H&S",
    "SIGNALS_SUMMARY_FLAG": "Flag",
    "SIGNALS_SUMMARY_PENNANT": "Pennant",
    "SIGNALS_SUMMARY_TRIPLE_TOP": "Triple Top",
    "SIGNALS_SUMMARY_TRIPLE_BOTTOM": "Triple Bottom",
    "SIGNALS_SUMMARY_RECTANGLE": "Rectangle",
    "SIGNALS_SUMMARY_TRADING_RANGE": "Trading Range",
    "SIGNALS_SUMMARY_TRADING_RANGE_V2": "Trading Range (squeeze)",
    # Candle patterns (mild interest — signals short-term)
    "SIGNALS_SUMMARY_HAMMER": "Hammer",
    "SIGNALS_SUMMARY_INVERTED_HAMMER": "Inverted Hammer",
    "SIGNALS_SUMMARY_MORNING_STAR": "Morning Star",
    "SIGNALS_SUMMARY_EVENING_STAR": "Evening Star",
    "SIGNALS_SUMMARY_ENGULFING": "Engulfing",
    "SIGNALS_SUMMARY_KICKER": "Kicker",
    "SIGNALS_SUMMARY_HARAMI": "Harami",
}

# Focused list for the default scan — the most actionable swing patterns
DEFAULT_PATTERN_SCAN = [
    "SIGNALS_SUMMARY_RISING_WEDGE",
    "SIGNALS_SUMMARY_FALLING_WEDGE",
    "SIGNALS_SUMMARY_CHANNEL_UP",
    "SIGNALS_SUMMARY_CHANNEL_DOWN",
    "SIGNALS_SUMMARY_PATTERN_BREAKOUTS",
    "SIGNALS_SUMMARY_PATTERN_BREAKOUTS_UPTREND_DOWNTREND",
    "SIGNALS_SUMMARY_EMERGING_PATTERNS",
    "SUPPORT_RESISTANCE_APPROACHING",
    "SUPPORT_RESISTANCE_APPROACHING_OVERSOLD",
    "SUPPORT_RESISTANCE_BREAKOUT",
    "SIGNALS_SUMMARY_ASCENDING_TRIANGLE",
    "SIGNALS_SUMMARY_DESCENDING_TRIANGLE",
    "SIGNALS_SUMMARY_DOUBLE_BOTTOM",
    "SIGNALS_SUMMARY_DOUBLE_TOP",
    "SIGNALS_SUMMARY_INVERSE_HEAD_AND_SHOULDERS",
    "SIGNALS_SUMMARY_FLAG",
    "SIGNALS_SUMMARY_PENNANT",
    "SIGNALS_SUMMARY_TRADING_RANGE_V2",
]


@dataclass
class PatternSignal:
    """A single chart pattern detection from altFINS."""
    symbol: str
    pattern_key: str
    pattern_name: str
    direction: str                # "BULLISH" / "BEARISH"
    signal_name: str              # Full description from altFINS
    last_price: Optional[float]
    timestamp: Optional[str]


@dataclass
class PatternScanResult:
    """Aggregated pattern scan results."""
    scan_time: str
    total_signals: int = 0
    by_pattern: Dict[str, List[PatternSignal]] = field(default_factory=dict)
    by_direction: Dict[str, int] = field(default_factory=lambda: {"BULLISH": 0, "BEARISH": 0})


def run_pattern_scan(
    signal_types: Optional[List[str]] = None,
    lookback: str = "last 24 hours",
    symbol_filter: Optional[List[str]] = None,
) -> PatternScanResult:
    """Run a chart pattern scan via altFINS signal feed.

    Args:
        signal_types: List of altFINS signal type keys. Defaults to DEFAULT_PATTERN_SCAN.
        lookback: Time window. Use "last 24 hours", "last 7 days", etc.
        symbol_filter: Optional list of symbols to filter (e.g. ["BTC", "ETH", "SOL"]).

    Returns:
        PatternScanResult grouped by pattern type.
    """
    types = signal_types or DEFAULT_PATTERN_SCAN
    result = PatternScanResult(
        scan_time=datetime.now(timezone.utc).isoformat(),
    )

    try:
        entries = get_signal_feed(
            signal_types=types,
            lookback=lookback,
            size=100,
            symbols=symbol_filter,
        )
    except AltfinsError as exc:
        logger.warning("Pattern scan unavailable: %s", exc)
        return result

    if not entries:
        logger.info("Pattern scan: 0 signals found.")
        return result

    # Organize by pattern key
    parsed: Dict[str, List[PatternSignal]] = {}
    bullish_count = 0
    bearish_count = 0

    for e in entries:
        sym = e.get("symbol", "?")
        direction = e.get("direction", "?")
        signal_key = e.get("signalKey", "?")
        signal_name = e.get("signalName", "")
        price_str = e.get("lastPrice")
        price = None
        if price_str:
            try:
                price = float(str(price_str).replace(",", ""))
            except (ValueError, TypeError):
                pass

        # Map the full signal key back to the friendly pattern name
        # signal_key looks like "SIGNALS_SUMMARY_FALLING_WEDGE.TXT"
        raw_key = signal_key.split(".")[0] if "." in str(signal_key) else str(signal_key)
        pattern_name = PATTERN_SIGNAL_TYPES.get(raw_key, signal_key)

        ps = PatternSignal(
            symbol=sym,
            pattern_key=raw_key,
            pattern_name=pattern_name,
            direction=direction,
            signal_name=signal_name,
            last_price=price,
            timestamp=e.get("timestamp"),
        )

        if raw_key not in parsed:
            parsed[raw_key] = []
        parsed[raw_key].append(ps)

        if direction == "BULLISH":
            bullish_count += 1
        elif direction == "BEARISH":
            bearish_count += 1

    result.by_pattern = parsed
    result.total_signals = len(entries)
    result.by_direction["BULLISH"] = bullish_count
    result.by_direction["BEARISH"] = bearish_count

    logger.info("Pattern scan: %d signals, %d patterns, %d bullish / %d bearish",
                result.total_signals, len(parsed), bullish_count, bearish_count)
    return result


def format_pattern_alert(result: PatternScanResult, max_per_pattern: int = 3) -> str:
    """Format a pattern scan result into a Telegram-friendly message."""
    lines = [
        "📐 *Pattern Scanner* — Live Chart Patterns",
        f"*{result.total_signals}* signals across *{len(result.by_pattern)}* patterns",
        f"🟢 Bullish: {result.by_direction['BULLISH']}  |  🔴 Bearish: {result.by_direction['BEARISH']}",
        "",
    ]

    for pattern_key, signals in sorted(result.by_pattern.items(), key=lambda x: -len(x[1])):
        name = signals[0].pattern_name if signals else pattern_key
        bullish = sum(1 for s in signals if s.direction == "BULLISH")
        bearish = sum(1 for s in signals if s.direction == "BEARISH")
        dir_emoji = "🟢" if bullish >= bearish else "🔴"
        lines.append(f"{dir_emoji} *{name}* ({len(signals)} signals)")
        for s in signals[:max_per_pattern]:
            price_str = f" ${s.last_price:.4f}" if s.last_price else ""
            ts = f" [{s.timestamp}]" if s.timestamp else ""
            lines.append(f"     {s.symbol} {s.direction}{price_str}{ts}")
        if len(signals) > max_per_pattern:
            lines.append(f"     … +{len(signals) - max_per_pattern} more")

    lines.append("")
    lines.append("🤖 *Pattern Scanner — auto-generated*")
    return "\n".join(lines)