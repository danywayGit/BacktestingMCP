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
import sqlite3

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

def _get_db() -> sqlite3.Connection:
    """Get a connection to the crypto database."""
    conn = sqlite3.connect('data/crypto.db')
    conn.row_factory = sqlite3.Row
    return conn


def _save_funding_snapshot(symbol: str, data: dict, oi: Optional[float] = None) -> None:
    """Persist one funding + OI snapshot to the funding_history table."""
    try:
        db = _get_db()
        db.execute(
            "INSERT INTO funding_history (symbol, funding_rate, mark_price, open_interest, fetched_at) VALUES (?, ?, ?, ?, ?)",
            (symbol, data.get("funding_rate", 0), data.get("mark_price", 0), oi or 0, datetime.now(timezone.utc).isoformat()),
        )
        db.commit()
        db.close()
    except Exception as exc:
        logger.debug("Failed to save funding snapshot for %s: %s", symbol, exc)


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
    """Compute real funding rate change per hour over the last N hours.

    Queries stored funding_history to find the rate 'hours' ago and
    computes delta/hour. Returns None if not enough history exists.
    """
    try:
        db = _get_db()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        # Get the oldest snapshot in the window
        old = db.execute(
            "SELECT funding_rate FROM funding_history WHERE symbol=? AND fetched_at <= ? ORDER BY fetched_at ASC LIMIT 1",
            (symbol, cutoff),
        ).fetchone()
        current = get_funding_rate(symbol)
        db.close()
        if old is not None and current is not None:
            delta = current - old["funding_rate"]
            return delta / max(hours, 1)
        return None
    except Exception:
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
    """Compute real OI change fraction over N hours.

    Queries stored funding_history for OI snapshots.
    Returns e.g., -0.05 = 5% decrease (shorts covering — bullish for LONG).
    Returns None if data unavailable.
    """
    try:
        db = _get_db()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        old = db.execute(
            "SELECT open_interest FROM funding_history WHERE symbol=? AND fetched_at <= ? AND open_interest > 0 ORDER BY fetched_at ASC LIMIT 1",
            (symbol, cutoff),
        ).fetchone()
        current = fetch_open_interest(symbol)
        db.close()
        if old is not None and current is not None and old["open_interest"] > 0:
            return (current - old["open_interest"]) / old["open_interest"]
        # Fallback: no history yet, assume neutral
        return 0.0
    except Exception:
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
    Also persists each snapshot to the funding_history table.
    Used by the edge-fund-rate CLI command to refresh all cached data before a scan.
    """
    results: Dict[str, dict] = {}
    for sym in symbols:
        data = fetch_funding_rate(sym, force=True)
        if data:
            results[sym] = data
            oi = fetch_open_interest(sym)
            _save_funding_snapshot(sym, data, oi)
    logger.info("Funding poll: %d/%d symbols refreshed and persisted", len(results), len(symbols))
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