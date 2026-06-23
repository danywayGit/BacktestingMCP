"""
Thin client for the altFINS MCP server.

Wraps the screener, signal feed, and news tools documented in the
altfinsMCP repo (examples/python/altfins_direct_tool_call.py) so they can
be called synchronously from CLI/cron code instead of through an
MCP-connected assistant session.

Requires ALTFINS_API_KEY in the environment (or a local .env file).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

ALTFINS_MCP_URL = "https://mcp.altfins.com/mcp"

load_dotenv()


class AltfinsError(RuntimeError):
    """Raised when altFINS cannot be reached or no API key is configured."""


def _extract_blocks(result: Any) -> List[Any]:
    parsed: List[Any] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if not text:
            continue
        try:
            parsed.append(json.loads(text))
        except (json.JSONDecodeError, TypeError):
            parsed.append(text)
    return parsed


def _flatten(blocks: List[Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, list):
            entries.extend(b for b in block if isinstance(b, dict))
        elif isinstance(block, dict):
            entries.append(block)
    return entries


async def _call_tool(tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    api_key = os.getenv("ALTFINS_API_KEY")
    if not api_key:
        raise AltfinsError("Set ALTFINS_API_KEY to query altFINS.")

    # Imported lazily so importing this module doesn't require the mcp
    # package unless altFINS is actually queried.
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = {"X-Api-Key": api_key}
    async with streamablehttp_client(ALTFINS_MCP_URL, headers=headers) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            return _flatten(_extract_blocks(result))


def get_screener_data(
    display_types: Optional[List[str]] = None,
    signal_filter_value: Optional[str] = None,
    size: int = 25,
    sort_field: str = "MARKET_CAP",
) -> List[Dict[str, Any]]:
    """Discover candidates via altFINS' screener (2000+ symbol universe).

    signal_filter_value: "UP" or "DOWN" to pre-filter by SHORT_TERM_TREND
    direction, or None for no directional filter.
    """
    display_types = display_types or ["SHORT_TERM_TREND", "VOLUME_RELATIVE", "RSI14", "MACD"]
    args: Dict[str, Any] = {
        "timeInterval": "DAILY",
        "displayTypes": display_types,
        "size": size,
        "sortField": sort_field,
        "sortDirection": "DESC",
    }
    if signal_filter_value:
        args["signalFilters"] = [
            {"signalFilterType": "SHORT_TERM_TREND", "signalFilterValue": signal_filter_value}
        ]
    return asyncio.run(_call_tool("screener_getAltfinsScreenerData", args))



# Signal types most useful for swing-trading candidate discovery.
# The API requires at least one signal type in 'signals' — this default
# covers trend, momentum, and pullback categories.
DEFAULT_SIGNAL_TYPES = [
    "PULLBACK_UP_DOWN_TREND",
    "SUPPORT_RESISTANCE_BREAKOUT",
    "FRESH_MOMENTUM_MACD_SIGNAL_LINE_CROSSOVER",
    "UP_DOWN_TREND_AND_FRESH_MOMENTUM_INFLECTION",
    "SIGNALS_SUMMARY_STRONG_UP_DOWN_TREND",
]


def get_signal_feed(
    symbols: Optional[List[str]] = None,
    signal_direction: Optional[str] = None,
    signal_types: Optional[List[str]] = None,
    lookback: str = "last 7 days",
    size: int = 50,
) -> List[Dict[str, Any]]:
    """Direct trading signals from the altFINS signal feed.

    The API equivalent of altFINS' VIP Telegram signal channel.
    Response fields: symbol, name, timestamp, direction (BULLISH/BEARISH),
    signalKey, signalName, lastPrice, priceChange, marketCap.

    signal_direction: "BULLISH", "BEARISH", or None for both.
    signal_types: list of signal type keys (see DEFAULT_SIGNAL_TYPES).
                  Defaults to DEFAULT_SIGNAL_TYPES if not provided.
                  'signals' is required by the API — never omit it.
    lookback: natural-language or ISO-8601 start time, e.g. "last 7 days".
    """
    args: Dict[str, Any] = {
        "signals": signal_types or DEFAULT_SIGNAL_TYPES,
        "from": lookback,
        "size": size,
        "sortField": "timestamp",
        "sortDirection": "DESC",
    }
    if symbols:
        args["symbols"] = symbols
    if signal_direction:
        args["signalDirection"] = signal_direction
    return asyncio.run(_call_tool("signal_feed_data", args))


def get_recent_news_counts(symbols: List[str], lookback: str = "last 2 days", size: int = 100) -> Dict[str, int]:
    """Count recent news mentions per symbol (no sentiment scoring yet)."""
    args = {
        "from": lookback,
        "size": size,
        "sortField": "timestamp",
        "sortDirection": "DESC",
    }
    entries = asyncio.run(_call_tool("news_getCryptoNewsMessages", args))
    counts = {s: 0 for s in symbols}
    for entry in entries:
        mentioned = [s.strip() for s in str(entry.get("assetSymbols", "")).split(",")]
        for sym in symbols:
            if sym in mentioned:
                counts[sym] += 1
    return counts


_TREND_RE = re.compile(r"(Strong\s+)?(Up|Down)\s*\((\d+)/10\)", re.IGNORECASE)


def parse_trend_score(trend_label: Any) -> float:
    """Map altFINS' SHORT_TERM_TREND label (e.g. "Strong Up (9/10)") to a
    signed -10..+10 score. Returns 0.0 for "Neutral" or unrecognized values."""
    if not isinstance(trend_label, str):
        return 0.0
    match = _TREND_RE.search(trend_label)
    if not match:
        return 0.0
    _, direction, magnitude = match.groups()
    score = float(magnitude)
    return score if direction.lower() == "up" else -score
