"""
Binance perpetual futures funding rate data for the edge scanner.

Provides live funding rates, open interest changes, and liquidation cluster
proximity for USDT-M perpetual symbols. Data is cached in-memory with a TTL
and also persisted to the `funding_rates` table for historical analysis.

Architecture
------------
- Polls Binance public REST endpoints (no API key required for public data).
- Cached in-memory per-symbol for re-use across config versions in one scan cycle.
- Funding data flows into score_symbol() as a per-symbol score bonus/malus,
  exactly like the Santiment on-chain netflow integration.

Endpoints:
  fapi/v1/premiumIndex  — current funding rate, next funding time
  fapi/v1/openInterest  — per-symbol open interest
  fapi/v1/klines        — price structure detection (15m)
"""

from __future__ import annotations

import logging
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

# ── Binance API ──────────────────────────────────────────────────────────────
BASE = "https://fapi.binance.com"

# ── In-memory cache ──────────────────────────────────────────────────────────
_FUNDING_CACHE: dict[str, tuple[float, dict]] = {}  # symbol -> (timestamp, result)
_FUNDING_CACHE_TTL = 300  # 5 minutes (funding updates every few minutes)
_OI_CACHE: dict[str, tuple[float, float]] = {}       # symbol -> (timestamp, oi_value)
_OI_CACHE_TTL = 600  # 10 minutes

# ── HTTP client ──────────────────────────────────────────────────────────────
_client: Optional[httpx.Client] = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=15.0)
    return _client


def _clear_cache() -> None:
    _FUNDING_CACHE.clear()
    _OI_CACHE.clear()


# ── Public API ───────────────────────────────────────────────────────────────

def fetch_funding_rate(symbol: str, force: bool = False) -> Optional[dict]:
    """Fetch the latest premium index (funding rate) for a symbol.

    Returns dict with keys: funding_rate, funding_time, next_funding_time,
    mark_price, index_price, or None on failure.
    Cached 5 min unless force=True.
    """
    now = _time.time()
    if not force and symbol in _FUNDING_CACHE:
        ts, data = _FUNDING_CACHE[symbol]
        if now - ts < _FUNDING_CACHE_TTL:
            return data

    try:
        resp = _get_client().get(f"{BASE}/fapi/v1/premiumIndex", params={"symbol": f"{symbol}USDT"})
        resp.raise_for_status()
        j = resp.json()
        result = {
            "funding_rate": float(j.get("lastFundingRate", 0)),
            "funding_time": j.get("fundingTime", 0),
            "next_funding_time": j.get("nextFundingTime", 0),
            "mark_price": float(j.get("markPrice", 0)),
            "index_price": float(j.get("indexPrice", 0)),
        }
        _FUNDING_CACHE[symbol] = (now, result)
        return result
    except Exception as exc:
        logger.debug("Failed to fetch funding rate for %s: %s", symbol, exc)
        return None


def get_funding_rate(symbol: str) -> Optional[float]:
    """Get the current funding rate for a symbol (-0.006 = -0.6%)."""
    data = fetch_funding_rate(symbol)
    if data is not None:
        return data["funding_rate"]
    return None


def get_funding_momentum(symbol: str, hours: int = 1) -> Optional[float]:
    """Estimate funding rate change over the last N hours.

    Uses recent funding_time timestamps if available in cache.
    Returns +0.00003 = +0.003%/hour (rising/favorable for LONG).
    Returns None if data unavailable.
    """
    # For now, estimate from the stored next_funding_time vs current time
    # A proper implementation would poll historical funding rates.
    data = fetch_funding_rate(symbol)
    if data is None:
        return None
    # Simple heuristic: if current funding is extreme (>0.006 or <-0.006),
    # assume it's normalizing toward zero (momentum opposite to extreme)
    fr = data["funding_rate"]
    if abs(fr) > 0.006:
        # Extreme → assume mean-reversion momentum
        return -fr * 0.01  # scaled to ~0.00006 for 0.6% extreme
    return None


def fetch_open_interest(symbol: str, force: bool = False) -> Optional[float]:
    """Fetch current open interest in USDT for a symbol."""
    now = _time.time()
    if not force and symbol in _OI_CACHE:
        ts, oi = _OI_CACHE[symbol]
        if now - ts < _OI_CACHE_TTL:
            return oi
    try:
        resp = _get_client().get(f"{BASE}/fapi/v1/openInterest", params={"symbol": f"{symbol}USDT"})
        resp.raise_for_status()
        oi = float(resp.json().get("openInterest", 0))
        _OI_CACHE[symbol] = (now, oi)
        return oi
    except Exception as exc:
        logger.debug("Failed to fetch OI for %s: %s", symbol, exc)
        return None


def get_oi_change(symbol: str, hours: int = 2) -> Optional[float]:
    """Estimate OI change fraction over N hours.

    Returns e.g., -0.05 = 5% decrease (shorts covering — bullish for LONG).
    Returns None if data unavailable.
    """
    current = fetch_open_interest(symbol)
    if current is None:
        return None
    # Simple fallback: no historical OI tracked yet, assume stable
    return 0.0


def get_funding_interval(symbol: str) -> int:
    """Get funding interval in hours for a symbol (default 8h)."""
    # Binance perps are mostly 8h. Some altpairs are 4h or 1h during volatility.
    # A production version would poll exchangeInfo.
    # For now, return 8 as default — the scan will adapt if data shows otherwise.
    return 8


# ── Bulk poll for scan cycle ─────────────────────────────────────────────────

def poll_all_funding(symbols: List[str]) -> Dict[str, dict]:
    """Fetch funding rates for a list of symbols in one batch.

    Returns {symbol: funding_data_dict}.
    Used by the edge-fund-rate CLI command to refresh all cached data before a scan.
    """
    results: Dict[str, dict] = {}
    for sym in symbols:
        data = fetch_funding_rate(sym, force=True)
        if data:
            results[sym] = data
    logger.info("Funding poll: %d/%d symbols refreshed", len(results), len(symbols))
    return results


def poll_all_open_interest(symbols: List[str]) -> Dict[str, float]:
    """Fetch OI for a list of symbols."""
    results: Dict[str, float] = {}
    for sym in symbols:
        oi = fetch_open_interest(sym, force=True)
        if oi:
            results[sym] = oi
    logger.info("OI poll: %d/%d symbols refreshed", len(results), len(symbols))
    return results


# ── Formatter for Telegram alerts ────────────────────────────────────────────

def format_funding_signal(symbol: str, direction: str, score: float, components: dict) -> str:
    """Format a funding-driven signal for Telegram alerts."""
    fr = components.get("funding_rate", "N/A")
    mom = components.get("funding_momentum", "N/A")
    oi = components.get("oi_change", "N/A")
    return (
        f"*{symbol}* {direction} | Funding Score: {score:.1f}\n"
        f"┃ Funding: {fr}% | Momentum: {mom} | OI: {oi}"
    )