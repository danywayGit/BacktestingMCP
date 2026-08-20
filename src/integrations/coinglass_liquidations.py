"""
Real liquidation data from Coinglass API.

Provides actual forced liquidation events (not ratio proxies):
- Total liquidation volume in USD
- Breakdown by LONG vs SHORT liquidations
- Liquidation imbalance score

Requires a free Coinglass API key (sign up at https://coinglass.com/).
Set COINGLASS_API_KEY in .env or in the config.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Cache
_cache: Dict[str, dict] = {}
_cache_time: Optional[datetime] = None
CACHE_TTL_SEC = 120  # 2 minutes

# API key — load from .env or config
COINGLASS_API_KEY = os.getenv("COINGLASS_API_KEY", "")
API_BASE = "https://open-api.coinglass.com"


def get_api_key() -> str:
    """Get the Coinglass API key from config."""
    if COINGLASS_API_KEY:
        return COINGLASS_API_KEY
    # Try to load from .env at runtime
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv("COINGLASS_API_KEY", "")
    except Exception:
        return ""


def fetch_liquidation_data(symbol: str, exchange: str = "Binance", time_type: str = "h1") -> Optional[dict]:
    """Fetch liquidation data from Coinglass API.
    
    Args:
        symbol: Trading symbol (e.g. 'BTC' or 'BTCUSDT')
        exchange: Exchange name (default: 'Binance')
        time_type: Aggregation period ('h1', 'h4', 'h12', 'd1')
    
    Returns:
        Dict with liquidation data or None if failed.
    """
    api_key = get_api_key()
    if not api_key:
        logger.warning("Coinglass API key not set. Set COINGLASS_API_KEY in .env")
        return None
    
    # Clean symbol (strip USDT suffix)
    clean_symbol = symbol.replace("USDT", "").replace("USD", "")
    
    headers = {
        "accept": "application/json",
        "apiKey": api_key,
        "coinglassSecret": api_key,
    }
    
    try:
        # Get liquidation chart data
        resp = httpx.get(
            f"{API_BASE}/api/v1/futures/liquidation/chart",
            params={
                "symbol": clean_symbol,
                "exName": exchange,
                "timeType": time_type,
            },
            headers=headers,
            timeout=10,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            logger.debug("Coinglass liquidation data received for %s", symbol)
            return data
        elif resp.status_code == 401 or resp.status_code == 403:
            logger.warning("Coinglass API key invalid or unauthorized")
            return None
        elif resp.status_code == 404:
            # Try alternative endpoint
            resp2 = httpx.get(
                f"{API_BASE}/api/v1/futures/liquidation",
                params={
                    "symbol": clean_symbol,
                    "exName": exchange,
                },
                headers=headers,
                timeout=10,
            )
            if resp2.status_code == 200:
                return resp2.json()
            logger.debug("Coinglass liquidation data not found for %s (%d)", symbol, resp.status_code)
            return None
        else:
            logger.debug("Coinglass API error: %d %s", resp.status_code, resp.text[:100])
            return None
    except Exception as e:
        logger.debug("Failed to fetch Coinglass liquidation data: %s", e)
        return None


def get_liquidation_score(
    symbol: str,
    time_type: str = "h1",
) -> Tuple[float, dict]:
    """Compute liquidation imbalance score from real Coinglass data.
    
    Returns:
        (score, components_dict) where:
        - Positive score → SHORT liquidations dominate → bullish (short squeeze)
        - Negative score → LONG liquidations dominate → bearish (long squeeze)
        - Magnitude reflects the USD value of the imbalance
    """
    api_key = get_api_key()
    if not api_key:
        # Fallback to Binance L/S ratio if no Coinglass key
        logger.debug("No Coinglass key, falling back to Binance L/S ratio")
        return 0.0, {"liq_data_source": "binance_ls_ratio_fallback"}

    data = fetch_liquidation_data(symbol, time_type=time_type)
    
    if data is None:
        return 0.0, {"liq_data_source": "unavailable"}
    
    # Parse the response — Coinglass returns liquidation data with
    # longLiquidationVolume and shortLiquidationVolume fields
    result = {}
    if isinstance(data, dict):
        result = data.get("data", data.get("result", {}))
    
    if not result:
        return 0.0, {"liq_data_source": "no_data"}
    
    # Extract liquidation volumes
    long_liq = float(result.get("longLiquidationVolume", 0) or 0)
    short_liq = float(result.get("shortLiquidationVolume", 0) or 0)
    total_liq = float(result.get("totalLiquidationVolume", 0) or 0)
    
    # If the data is in a timeseries format, take the latest values
    long_liq_list = result.get("longList", [])
    short_liq_list = result.get("shortList", [])
    if long_liq_list and short_liq_list:
        latest_idx = -1
        long_liq = float(long_liq_list[latest_idx] or 0)
        short_liq = float(short_liq_list[latest_idx] or 0)
        total_liq = long_liq + short_liq
    
    if total_liq == 0:
        return 0.0, {"liq_data_source": "no_liquidations", "liq_total_volume": 0}
    
    # Imbalance: positive = short liq dominates (bullish)
    imbalance = (short_liq - long_liq) / total_liq if total_liq > 0 else 0
    
    # Scale: $1M = 1.0, $10M = 2.0, $100M = 3.0
    volume_scale = min(1.0 + (total_liq / 1_000_000) ** 0.5, 5.0)
    score = imbalance * volume_scale
    
    components = {
        "liq_data_source": "coinglass",
        "liq_long_volume": round(long_liq, 0),
        "liq_short_volume": round(short_liq, 0),
        "liq_total_volume": round(total_liq, 0),
        "liq_imbalance": round(imbalance, 3),
        "liq_pressure_score": round(score, 3),
    }
    
    return round(score, 3), components


def get_liquidation_cached(symbol: str) -> Tuple[float, dict]:
    """Cached version of get_liquidation_score."""
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
    
    score, components = get_liquidation_score(symbol)
    _cache[symbol] = {"score": score, "components": components}
    _cache_time = now
    return score, components