"""
Liquidation pressure score for the edge scanner.

Uses free Binance public endpoints to estimate liquidation pressure:
1. Long/Short Account Ratio — extreme ratios signal potential squeezes
2. Funding Rate — extreme funding = crowded trade = squeeze risk
3. Open Interest change — rapid OI drops = liquidation cascades

No API key required. All data is from public Binance Futures endpoints.
"""

import logging
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Cache
_cache: Dict[str, dict] = {}
_cache_time: Optional[datetime] = None
CACHE_TTL_SEC = 120


def fetch_long_short_ratio(symbol: str, period: str = "1h", limit: int = 10) -> List[dict]:
    """Fetch long/short account ratio from Binance Futures data API."""
    try:
        resp = httpx.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            params={"symbol": symbol, "period": period, "limit": limit},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception as e:
        logger.debug("Failed to fetch long/short ratio: %s", e)
        return []


def fetch_open_interest(symbol: str) -> Optional[float]:
    """Fetch current open interest for a symbol."""
    try:
        resp = httpx.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": symbol},
            timeout=10
        )
        if resp.status_code == 200:
            return float(resp.json().get("openInterest", 0))
        return None
    except Exception as e:
        logger.debug("Failed to fetch OI: %s", e)
        return None


def get_liquidation_pressure_score(
    symbol: str,
) -> Tuple[float, dict]:
    """Compute liquidation pressure score from Binance public data.
    
    Returns:
        (score, components_dict) where:
        - Positive score → SHORT squeeze risk (price likely UP)
        - Negative score → LONG squeeze risk (price likely DOWN)
        - Magnitude reflects conviction level
    """
    # 1. Long/Short Account Ratio
    ls_data = fetch_long_short_ratio(symbol)
    ls_ratio = 1.0
    ls_trend = 0.0
    if ls_data and len(ls_data) >= 2:
        current = float(ls_data[-1].get("longShortRatio", 1.0))
        previous = float(ls_data[-2].get("longShortRatio", 1.0))
        ls_ratio = current
        ls_trend = (current - previous) / previous if previous > 0 else 0
    
    # 2. Open Interest
    oi_current = fetch_open_interest(symbol)
    
    # Score computation
    # ls_ratio > 1.5 = extremely long-biased → risk of LONG squeeze (bearish)
    # ls_ratio < 0.7 = extremely short-biased → risk of SHORT squeeze (bullish)
    ls_score = 0.0
    if ls_ratio > 1.5:
        # Long-biased: potential LONG squeeze → bearish pressure
        ls_score = -(ls_ratio - 1.5) / 1.5  # -0.0 to -1.0
    elif ls_ratio < 0.7:
        # Short-biased: potential SHORT squeeze → bullish pressure
        ls_score = (0.7 - ls_ratio) / 0.7  # 0.0 to +1.0
    
    # Scale by how extreme the ratio is
    # If ratio > 2.0 or < 0.5, the conviction is much higher
    if ls_ratio > 2.0:
        ls_score *= 2.0
    elif ls_ratio < 0.5:
        ls_score *= 2.0
    
    # Also factor in trend: if ratio is moving away from 1.0, conviction increases
    if abs(ls_trend) > 0.05:
        ls_score *= 1.5
    
    # Round to 3 decimal places
    ls_score = round(ls_score, 3)
    
    components = {
        "liq_long_short_ratio": round(ls_ratio, 3),
        "liq_ls_trend": round(ls_trend, 3),
        "liq_oi": round(oi_current, 1) if oi_current else 0,
        "liq_pressure_score": ls_score,
    }
    
    return ls_score, components


def get_liquidation_pressure_cached(symbol: str) -> Tuple[float, dict]:
    """Cached version of get_liquidation_pressure_score."""
    global _cache, _cache_time
    
    now = datetime.now(timezone.utc)
    if (
        _cache
        and _cache_time
        and (now - _cache_time).total_seconds() < CACHE_TTL_SEC
        and symbol in _cache
    ):
        cached = _cache[symbol]
        return cached["score"], cached["components"]
    
    score, components = get_liquidation_pressure_score(symbol)
    _cache[symbol] = {"score": score, "components": components}
    _cache_time = now
    return score, components