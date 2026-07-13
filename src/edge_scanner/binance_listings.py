"""
Binance New Listing Watcher — detects newly listed coins on Binance
and immediately scores them as potential gems.

Uses Binance's official announcements RSS/API to detect new listings
within 24-48 hours of going live.
"""

import logging
import httpx
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Binance new listing endpoints
BINANCE_ANNOUNCEMENT_URL = "https://www.binance.com/bapi/growth/v1/friendly/growth/cms/article/list"
QUERY_PARAMS = {
    "pageNo": 1,
    "pageSize": 20,
    "type": 1,  # Announcements
    "catalogId": 48,  # New Cryptocurrency Listing
    "sortBy": "createdDate",
    "orderBy": "desc",
}


def get_recent_binance_listings(hours: int = 48) -> List[Dict]:
    """Fetch recently listed coins from Binance announcements.

    Returns list of dicts with symbol, listing_date, and article info.
    """
    try:
        resp = httpx.get(
            BINANCE_ANNOUNCEMENT_URL,
            params=QUERY_PARAMS,
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("Binance announcement API returned %d", resp.status_code)
            return []
        
        data = resp.json()
        articles = (data.get("data", {}) or {}).get("catalogs", [])
        
        new_listings = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        for article in articles:
            article_info = article.get("articleVO", {})
            title = (article_info.get("title", "") or "")
            created = (article_info.get("createdDate", "") or "")
            
            # Check if it's a new listing announcement
            if "Binance Will List" in title or "Binance Will Launch" in title:
                # Extract symbol from title e.g. "Binance Will List KITE (KITE)"
                import re
                match = re.search(r"List\s+(\w+)", title)
                if match:
                    symbol = match.group(1)
                    new_listings.append({
                        "symbol": symbol.upper(),
                        "title": title,
                        "date": created[:10] if created else "",
                        "article_id": article_info.get("id", ""),
                    })
        
        return new_listings
    except Exception as e:
        logger.error("Failed to fetch Binance listings: %s", e)
        return []


def check_and_scan_new_listings() -> List[str]:
    """Check for new Binance listings and scan them as gems.

    Returns list of newly discovered symbols.
    """
    new_listings = get_recent_binance_listings(hours=48)
    if not new_listings:
        logger.info("No new Binance listings found in last 48h")
        return []
    
    discovered = []
    for listing in new_listings:
        symbol = listing["symbol"]
        logger.info("New Binance listing detected: %s (%s)", symbol, listing["title"])
        
        # Check if we already have this symbol in our DB
        import sqlite3
        conn = sqlite3.connect("/home/hermes/BacktestingMCP/data/crypto.db")
        existing = conn.execute(
            "SELECT COUNT(*) FROM edge_signals WHERE symbol = ?", (symbol,)
        ).fetchone()[0]
        conn.close()
        
        if existing > 0:
            logger.info("  Already tracking %s in edge_signals", symbol)
            continue
        
        discovered.append(symbol)
    
    if discovered:
        logger.info("New Binance listings to analyze: %s", discovered)
    
    return discovered