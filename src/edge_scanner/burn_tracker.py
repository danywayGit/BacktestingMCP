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

# Tokenomist API endpoint (public, no key needed for basic data)
TOKENOMIST_URL = "https://api.tokenomist.ai/v1/burns"

# Fallback: scrape from CoinGecko's token highlights
COINGECKO_HIGHLIGHTS = "https://api.coingecko.com/api/v3/coins/categories"


def get_burn_events_from_tokenomist() -> List[Dict]:
    """Fetch burn events from Tokenomist public API."""
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
    """Check CoinGecko for token categories that might have burn info."""
    burns = []
    try:
        resp = httpx.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "category": "token-burn",
                "order": "volume_desc",
                "per_page": 50,
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "7d,30d",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            coins = resp.json()
            for c in coins:
                burns.append({
                    "symbol": c.get("symbol", "").upper(),
                    "name": c.get("name", ""),
                    "price": c.get("current_price", 0),
                    "mcap": c.get("market_cap", 0),
                    "vol_24h": c.get("total_volume", 0),
                    "price_change_7d": c.get("price_change_percentage_7d_in_currency"),
                    "price_change_30d": c.get("price_change_percentage_30d_in_currency"),
                    "source": "coingecko_burn_category",
                })
    except Exception as e:
        logger.warning("CoinGecko burn category fetch failed: %s", e)
    return burns


def get_current_gem_candidates() -> List[str]:
    """Get symbols from the current gem scanner top picks."""
    try:
        import sys
        sys.path.insert(0, "/home/hermes/BacktestingMCP")
        from src.edge_scanner.gem_scanner import scan_gems
        candidates = scan_gems(pages=1, start_page=3)
        return [c.symbol for c in candidates[:50]]
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
            val = b.get("value_7d", 0)
            change = b.get("change_pct", 0)
            val_str = f"${val:,.0f}" if isinstance(val, (int, float)) and val > 0 else "N/A"
            change_str = f"{change:+.0f}%" if isinstance(change, (int, float)) else ""
            lines.append(f"  • {sym} — {val_str} burned (7d) {change_str}")
    
    if burns:
        lines.append("")
        lines.append("*Top burns across all tracked coins:*")
        lines.append("")
        for b in burns[:15]:
            sym = b.get("symbol", "?")
            val = b.get("value_7d", 0)
            change = b.get("change_pct", 0)
            val_str = f"${val:,.0f}" if isinstance(val, (int, float)) and val > 0 else "N/A"
            change_str = f"{change:+.0f}%" if isinstance(change, (int, float)) else ""
            lines.append(f"  • {sym} — {val_str} burned (7d) {change_str}")
    else:
        lines.append("No burn data available at this time.")
    
    lines.append("")
    lines.append("_Source: Tokenomist / CoinGecko burn category_")
    
    return "\n".join(lines)


def run_burn_check() -> str:
    """Full burn check pipeline."""
    burns = get_burn_events_from_tokenomist()
    if not burns:
        logger.info("Tokenomist failed, falling back to CoinGecko burn category")
        burns = get_burn_events_from_coingecko()
    
    gem_matches = check_burns_on_gems()
    
    return format_burn_report(burns, gem_matches)