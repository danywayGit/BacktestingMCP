"""
Real liquidation data from Coinglass API v4.

Provides actual forced liquidation events:
- Total liquidation volume in USD
- Breakdown by LONG vs SHORT liquidations
- Liquidation imbalance score

Requires Coinglass API key (https://coinglass.com/api).
Free tier: limited. Pro/Enterprise tier: full liquidation data.
Auth: CG-API-KEY header
Base: https://open-api-v4.coinglass.com
"""

import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_cache: Dict[str, dict] = {}
_cache_time: Optional[datetime] = None
CACHE_TTL_SEC = 120

# Circuit breaker: once Coinglass reports "upgrade required", skip all
# further per-symbol attempts (they all return the same plan error anyway).
_UPGRADE_DETECTED = False

API_BASE = "https://open-api-v4.coinglass.com"


def _get_api_key() -> str:
    """Get Coinglass API key from environment."""
    key = os.getenv("COINGLASS_API_KEY", "")
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            key = os.getenv("COINGLASS_API_KEY", "")
        except Exception:
            pass
    return key


def fetch_liquidation_data(symbol: str) -> Optional[dict]:
    """Fetch aggregated liquidation history from Coinglass v4 API."""
    api_key = _get_api_key()
    if not api_key:
        return None

    clean = symbol.replace("USDT", "").replace("USD", "")
    
    try:
        resp = httpx.get(
            f"{API_BASE}/api/futures/liquidation/aggregated-history",
            params={"symbol": clean, "exName": "Binance"},
            headers={"accept": "application/json", "CG-API-KEY": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            code = data.get("code", "")
            if code == "200" or code == 200:
                return data
            if "Upgrade" in data.get("msg", ""):
                return {"_upgrade_required": True}
        return None
    except Exception as e:
        logger.debug("Coinglass fetch error: %s", e)
        return None


def get_liquidation_score(symbol: str) -> Tuple[float, dict]:
    """Compute liquidation imbalance score from Coinglass v4."""
    global _UPGRADE_DETECTED
    api_key = _get_api_key()
    if not api_key:
        return 0.0, {"liq_data_source": "no_key", "liq_note": "Set COINGLASS_API_KEY in .env"}

    # Short-circuit once the plan is known insufficient.
    if _UPGRADE_DETECTED:
        return 0.0, {"liq_data_source": "upgrade_required", "liq_note": "Coinglass plan needs upgrade for liquidation data"}

    data = fetch_liquidation_data(symbol)

    if data is None:
        return 0.0, {"liq_data_source": "unavailable"}

    if data.get("_upgrade_required"):
        _UPGRADE_DETECTED = True
        return 0.0, {"liq_data_source": "upgrade_required", "liq_note": "Coinglass plan needs upgrade for liquidation data"}
    
    result = data.get("data", {})
    if not result:
        return 0.0, {"liq_data_source": "no_data"}
    
    long_liq = float(result.get("longVol", result.get("longLiquidationVolume", 0)) or 0)
    short_liq = float(result.get("shortVol", result.get("shortLiquidationVolume", 0)) or 0)
    total_liq = long_liq + short_liq
    
    if total_liq == 0:
        return 0.0, {"liq_data_source": "no_liquidations", "liq_total_volume": 0}
    
    imbalance = (short_liq - long_liq) / total_liq
    volume_scale = min(1.0 + (total_liq / 1_000_000) ** 0.5, 5.0)
    score = imbalance * volume_scale
    
    components = {
        "liq_data_source": "coinglass_v4",
        "liq_long_volume": round(long_liq, 0),
        "liq_short_volume": round(short_liq, 0),
        "liq_total_volume": round(total_liq, 0),
        "liq_imbalance": round(imbalance, 3),
        "liq_pressure_score": round(score, 3),
    }
    return round(score, 3), components


def get_liquidation_cached(symbol: str) -> Tuple[float, dict]:
    """Cached version."""
    global _cache, _cache_time
    now = datetime.now(timezone.utc)
    if _cache and _cache_time and (now - _cache_time).total_seconds() < CACHE_TTL_SEC and symbol in _cache:
        return _cache[symbol]["score"], _cache[symbol]["components"]
    score, comp = get_liquidation_score(symbol)
    _cache[symbol] = {"score": score, "components": comp}
    _cache_time = now
    return score, comp