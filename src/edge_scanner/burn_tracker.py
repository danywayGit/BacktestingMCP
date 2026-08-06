"""
Burn event tracker — monitors token buyback & burn events using Tokenomist data.

Finds coins with active burn programs and increasing burn rates,
which can be accumulation signals for gems.
"""

import httpx
import logging
import re
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Tokenomist API endpoint (appears to have changed/discontinued as of 2026-07)
TOKENOMIST_URL = "https://api.tokenomist.ai/v1/burns"

# CoinGecko free endpoints — no API key needed for basic data
COINGECKO_SEARCH = "https://api.coingecko.com/api/v3/search?query=burn"
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"
COINGECKO_TRENDING = "https://api.coingecko.com/api/v3/search/trending"
COINGECKO_PRICE = "https://api.coingecko.com/api/v3/simple/price"


def get_burn_events_from_tokenomist() -> List[Dict]:
    """Fetch burn events from Tokenomist public API (may be broken)."""
    try:
        resp = httpx.get(
            "https://api.tokenomist.ai/v1/burns",
            params={"limit": 50, "sortBy": "value", "order": "desc"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            burns = []
            for item in data.get("data", []):
                burns.append({
                    "symbol": item.get("token", item.get("symbol", "")).upper(),
                    "name": item.get("name", ""),
                    "value_7d": item.get("value_7d", 0),
                    "value_30d": item.get("value_30d", 0),
                    "change_pct": item.get("change_pct", 0),
                    "type": item.get("type", "burn"),
                    "source": "tokenomist",
                })
            return burns
    except Exception as e:
        logger.warning("Tokenomist burn API failed: %s", e)
    return []


def get_burn_events_from_coingecko() -> List[Dict]:
    """Check CoinGecko for known burn/buyback tokens via price + trending."""
    burns = []
    # 1. Known major burn tokens: BNB (quarterly auto-burn), ETH (EIP-1559)
    try:
        resp = httpx.get(
            COINGECKO_PRICE,
            params={
                "ids": "binancecoin,ethereum,okb,leo-token,kucoin-shares,cro,crypto-com-chain,near,aptos",
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_7d_change": "true",
                "include_30d_change": "true",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            prices = resp.json()
            name_map = {
                "binancecoin": "BNB",
                "ethereum": "ETH",
                "okb": "OKB",
                "leo-token": "LEO",
                "kucoin-shares": "KCS",
                "cro": "CRO",
                "crypto-com-chain": "CRO",
                "near": "NEAR",
                "aptos": "APT",
            }
            for tid, sym in name_map.items():
                if tid in prices:
                    p = prices[tid]
                    burns.append({
                        "symbol": sym,
                        "name": sym,
                        "price": p.get("usd", 0),
                        "price_change_7d": p.get("usd_7d_change"),
                        "price_change_30d": p.get("usd_30d_change"),
                        "source": "coingecko_price",
                    })
    except Exception as e:
        logger.warning("CoinGecko price fetch failed: %s", e)

    # 2. Trending — often carries burn/buyback narratives
    try:
        resp = httpx.get(COINGECKO_TRENDING, timeout=15)
        if resp.status_code == 200:
            for c in resp.json().get("coins", []):
                item = c.get("item", {})
                burns.append({
                    "symbol": item.get("symbol", "").upper(),
                    "name": item.get("name", ""),
                    "price": item.get("price_btc"),
                    "price_change_7d": None,
                    "price_change_30d": None,
                    "market_cap_rank": item.get("market_cap_rank"),
                    "source": "coingecko_trending",
                })
    except Exception as e:
        logger.warning("CoinGecko trending fetch failed: %s", e)

    return burns


def get_current_gem_candidates() -> List[str]:
    """Get symbols from the current gem scanner top picks (with timeout)."""
    try:
        import sys
        sys.path.insert(0, "/home/hermes/BacktestingMCP")
        from src.edge_scanner.gem_scanner import scan_gems
        # Use a very limited scan to avoid hanging
        candidates = scan_gems(pages=1, start_page=1)
        return [c.symbol for c in candidates[:30]]
    except Exception as e:
        logger.warning("Could not fetch gem candidates: %s", e)
        return []


def check_burns_on_gems() -> List[Dict]:
    """Check if any of our gem candidates have active burn programs."""
    gem_symbols = get_current_gem_candidates()
    if not gem_symbols:
        return []

    # Get burn events
    burns = get_burn_events_from_tokenomist()
    if not burns:
        burns = get_burn_events_from_coingecko()

    # Cross-reference with gem candidates
    matches = []
    for burn in burns:
        if burn.get("symbol") in gem_symbols:
            matches.append(burn)

    return matches


def format_burn_report(burns: List[Dict], gem_matches: List[Dict]) -> str:
    """Format burn events for Telegram."""
    lines = [
        "🔥 *Burn Event Tracker — {}*".format(datetime.now().strftime("%Y-%m-%d")),
        "",
    ]

    if gem_matches:
        lines.append("*Active burns on your gem candidates:*")
        lines.append("")
        for b in gem_matches[:10]:
            sym = b.get("symbol", "?")
            val = b.get("value_7d", b.get("price", 0))
            change = b.get("change_pct", b.get("price_change_7d", 0))
            val_str = f"${val:,.0f}" if isinstance(val, (int, float)) and val > 0 else "N/A"
            change_str = f"{change:+.0f}%" if isinstance(change, (int, float)) else ""
            lines.append(f"  \u2022 {sym} \u2014 {val_str} burned (7d) {change_str}")

    lines.append("")
    lines.append("*Market Overview:*")
    lines.append("")

    # Try global data
    try:
        resp = httpx.get(COINGECKO_GLOBAL, timeout=10)
        if resp.status_code == 200:
            d = resp.json().get("data", {})
            mcap = d.get("total_market_cap", {}).get("usd", 0)
            vol = d.get("total_volume", {}).get("usd", 0)
            btc_dom = d.get("market_cap_percentage", {}).get("btc", 0)
            eth_dom = d.get("market_cap_percentage", {}).get("eth", 0)
            lines.append(f"  \u2022 Total MC: ${mcap:,.0f}  |  24h Vol: ${vol:,.0f}")
            lines.append(f"  \u2022 BTC Dom: {btc_dom:.1f}%  |  ETH Dom: {eth_dom:.1f}%")
    except Exception:
        pass

    if burns:
        lines.append("")
        lines.append("*Tracked Burn Tokens:*")
        lines.append("")
        # Group by source for clarity
        price_based = [b for b in burns if b.get("source") == "coingecko_price"]
        trending = [b for b in burns if b.get("source") == "coingecko_trending"]

        if price_based:
            lines.append("_Known burn/buyback tokens:_")
            for b in price_based:
                sym = b.get("symbol", "?")
                price = b.get("price", 0)
                d7 = b.get("price_change_7d")
                d30 = b.get("price_change_30d")
                price_str = f"${price:,.2f}" if isinstance(price, (int, float)) and price > 0 else "N/A"
                d7s = f"{d7:+.2f}%" if isinstance(d7, (int, float)) else "N/A"
                d30s = f"{d30:+.2f}%" if isinstance(d30, (int, float)) else "N/A"
                lines.append(f"  \u2022 {sym:6s} {price_str:>10s}  |  7d: {d7s:>8s}  |  30d: {d30s:>8s}")

        if trending:
            lines.append("")
            lines.append("_Trending coins (burn narrative watchlist):_")
            for b in trending[:10]:
                sym = b.get("symbol", "?")
                name = b.get("name", "")
                rank = b.get("market_cap_rank", "?")
                lines.append(f"  \u2022 {sym:8s} {name:20s} rank: #{rank}")
    else:
        lines.append("No burn data available at this time.")

    lines.append("")
    lines.append("_Source: CoinGecko API_")

    return "\n".join(lines)


def run_burn_check() -> str:
    """Full burn check pipeline."""
    burns = get_burn_events_from_tokenomist()
    if not burns:
        logger.info("Tokenomist failed, falling back to CoinGecko")
        burns = get_burn_events_from_coingecko()

    # Skip gem scanner cross-reference — it triggers CoinGecko rate limits
    # on individual coin lookups. Tokenomist API is dead anyway.
    return format_burn_report(burns, [])