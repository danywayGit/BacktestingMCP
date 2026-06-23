"""
Thin client for Santiment's SanAPI (on-chain exchange-flow data).

Free tier constraints (as of evaluation): ~1,000 calls/month, most metrics
delayed up to 30 days, 1-year historical cap. That means on-chain data here
is a slow-moving directional bias to blend into the composite score, not a
real-time signal -- see docs/EDGE_SCANNER_PLAN.md Phase 3 for the rationale
(CryptoQuant's netflow API requires a paid plan; Santiment's free tier was
the only one with programmatic free access to exchange flow + whale data).

Requires SANTIMENT_API_KEY in the environment (or a local .env file).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import httpx
from dotenv import load_dotenv

SANTIMENT_GRAPHQL_URL = "https://api.santiment.net/graphql"

load_dotenv()


class SantimentError(RuntimeError):
    """Raised when Santiment cannot be reached, has no API key, or the
    symbol has no known slug mapping."""


# Santiment identifies assets by project "slug", not exchange ticker.
# Free-tier exchange-flow/whale coverage is best for large/mid caps; symbols
# missing here degrade to "no on-chain signal" rather than crashing the scan.
_SLUG_MAP: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binance-coin",
    "SOL": "solana",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche",
    "DOT": "polkadot",
    "LINK": "chainlink",
    "MATIC": "matic-network",
    "LTC": "litecoin",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "NEAR": "near-protocol",
    "APT": "aptos",
    "ARB": "arbitrum",
    "OP": "optimism",
    "FIL": "filecoin",
    "TRX": "tron",
    "ETC": "ethereum-classic",
    "XLM": "stellar",
    "ALGO": "algorand",
    "VET": "vechain",
    "SAND": "the-sandbox",
    "MANA": "decentraland",
    "AAVE": "aave",
    "MKR": "maker",
    "SHIB": "shiba-inu",
    "PEPE": "pepe",
}

# USD-denominated so values are comparable across assets of very different
# market cap (raw token-unit inflow/outflow isn't).
_METRICS = ["exchange_inflow_usd", "exchange_outflow_usd"]


def slug_for(symbol: str) -> Optional[str]:
    return _SLUG_MAP.get(symbol.upper())


def _build_query(slug: str, from_iso: str, to_iso: str) -> str:
    aliases = "".join(
        f'''
        m{i}: getMetric(metric: "{metric}") {{
            timeseriesData(slug: "{slug}", from: "{from_iso}", to: "{to_iso}", interval: "1d") {{
                datetime
                value
            }}
        }}'''
        for i, metric in enumerate(_METRICS)
    )
    return "query {" + aliases + "\n}"


def get_onchain_snapshot(symbol: str, lookback_days: int = 7) -> Dict[str, float]:
    """Sum of exchange inflow/outflow USD over a recent window.

    Santiment free tier has up to a ~30-day data lag — the most recent
    ~30 days are not available. We query a 7-day window ending 35 days ago
    (safely inside the free-tier allowed range) so the call always succeeds
    without a paid plan. This makes the on-chain signal a slow directional
    bias, not a real-time indicator — see docs/EDGE_SCANNER_PLAN.md Phase 3.

    Raises SantimentError if SANTIMENT_API_KEY is unset, the symbol has no
    slug mapping, or the request fails -- callers should catch this and
    treat it as "no on-chain component", matching how altfins_client's
    AltfinsError is handled in composite.py.
    """
    api_key = os.getenv("SANTIMENT_API_KEY")
    if not api_key:
        raise SantimentError("Set SANTIMENT_API_KEY to query Santiment.")

    slug = slug_for(symbol)
    if not slug:
        raise SantimentError(f"No Santiment slug mapping for {symbol!r}.")

    # Free tier: data available from ~1 year ago up to ~30 days ago.
    # Query a window ending 35 days ago to stay safely within the free range.
    LAG_DAYS = 35
    to_dt = datetime.now(timezone.utc) - timedelta(days=LAG_DAYS)
    from_dt = to_dt - timedelta(days=lookback_days)
    query = _build_query(slug, from_dt.isoformat(), to_dt.isoformat())

    try:
        response = httpx.post(
            SANTIMENT_GRAPHQL_URL,
            json={"query": query},
            headers={"Authorization": f"Apikey {api_key}"},
            timeout=15.0,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SantimentError(f"Santiment request failed for {slug}: {exc}") from exc

    if "errors" in payload:
        raise SantimentError(f"Santiment API error for {slug}: {payload['errors']}")

    data = payload.get("data") or {}
    result: Dict[str, float] = {}
    for i, metric in enumerate(_METRICS):
        series = (data.get(f"m{i}") or {}).get("timeseriesData") or []
        result[metric] = sum(float(point.get("value") or 0.0) for point in series)
    return result
