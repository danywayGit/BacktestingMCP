"""
altFINS Pattern Cache — cached chart pattern data from altFINS TA page.

Provides a simple lookup function that the scoring pipeline uses.
The cache is refreshed by running the scraper via cron.

Usage:
    from src.edge_scanner.altfins_pattern_cache import get_pattern_for_symbol, refresh_patterns
    
    # Get pattern for a symbol
    pattern = get_pattern_for_symbol("AAVE")  # Returns {"outlook": "Bullish", "pattern": "...", "stage": "..."} or None
    
    # Force refresh
    refresh_patterns()
"""

import json, os, time
from typing import Optional, Dict, Any
from datetime import datetime, timezone

CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "altfins_patterns.json")
CACHE_MAX_AGE_SECONDS = 24 * 3600  # 24 hours

# In-memory cache
_patterns = None
_last_refresh = 0


def _load_cache() -> Dict[str, Dict[str, str]]:
    """Load pattern data from cache file."""
    global _patterns, _last_refresh

    # Check if file is fresh enough
    cache_path = os.path.abspath(CACHE_FILE)
    if os.path.exists(cache_path):
        file_age = time.time() - os.path.getmtime(cache_path)
        if file_age < CACHE_MAX_AGE_SECONDS:
            try:
                with open(cache_path, "r") as f:
                    _patterns = json.load(f)
                _last_refresh = time.time()
                return _patterns
            except (json.JSONDecodeError, IOError):
                pass

    # Cache expired or unavailable — return empty
    return {}


def get_pattern_for_symbol(symbol: str) -> Optional[Dict[str, str]]:
    """Get chart pattern data for a symbol. Returns None if no pattern found."""
    if _patterns is None:
        _load_cache()
    
    if not _patterns:
        return None
    
    # Try exact match, then uppercase
    pattern = _patterns.get(symbol) or _patterns.get(symbol.upper())
    if pattern:
        return pattern
    
    # Try matching on symbol name (some entries might have full names)
    for sym, data in _patterns.items():
        if sym.upper() == symbol.upper():
            return data
    
    return None


def refresh_patterns(force: bool = False) -> Dict[str, Dict[str, str]]:
    """Run the scraper and update the cache."""
    global _patterns, _last_refresh
    
    try:
        from src.integrations.altfins_ta_scraper import AltfinsTAScraper
        scraper = AltfinsTAScraper()
        try:
            patterns = scraper.get_patterns()
        finally:
            scraper.close()
        
        # Convert to lookup dict: symbol -> {outlook, pattern, stage}
        lookup = {}
        for p in patterns:
            sym = p.get("symbol", "").upper()
            if sym:
                lookup[sym] = {
                    "outlook": p.get("outlook", "Neutral"),
                    "pattern": p.get("pattern", ""),
                    "stage": p.get("stage", "Emerging"),
                }
        
        # Save to cache file
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        cache_path = os.path.abspath(CACHE_FILE)
        with open(cache_path, "w") as f:
            json.dump(lookup, f, indent=2)
        
        _patterns = lookup
        _last_refresh = time.time()
        
        return lookup
    except Exception as e:
        # If scraper fails, return existing cache
        if _patterns is None:
            _load_cache()
        return _patterns or {}


def get_all_patterns() -> Dict[str, Dict[str, str]]:
    """Get all pattern data."""
    if _patterns is None:
        _load_cache()
    return _patterns or {}


# Auto-refresh on import if cache is stale
if _patterns is None:
    _load_cache()


if __name__ == "__main__":
    import sys
    if "--refresh" in sys.argv:
        print("Refreshing altFINS patterns...")
        result = refresh_patterns(force=True)
        print(f"Cached {len(result)} patterns")
    else:
        patterns = get_all_patterns()
        print(f"Patterns in cache: {len(patterns)}")
        for sym, data in sorted(patterns.items()):
            print(f"  {sym:<10} {data.get('outlook',''):<12} {data.get('pattern',''):<25} {data.get('stage','')}")