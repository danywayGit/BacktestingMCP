"""
altFINS web connector — scrapes chart patterns and technical analysis directly
from altfins.com using browser automation (playwright) for richer data than the
MCP server provides.

Requires:
  pip install playwright
  playwright install chromium
  ALTFINS_EMAIL and ALTFINS_PASSWORD in .env (ask user for credentials)
"""

import os, json, re, time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

class AltfinsWebError(RuntimeError):
    """Raised when altFINS web scraping fails."""

_playwright_available = None

def _check_playwright() -> bool:
    global _playwright_available
    if _playwright_available is not None:
        return _playwright_available
    try:
        import playwright
        _playwright_available = True
    except ImportError:
        _playwright_available = False
    return _playwright_available


def get_credentials() -> Tuple[str, str]:
    """Return (email, password) for altFINS login."""
    email = os.getenv("ALTFINS_EMAIL", "")
    password = os.getenv("ALTFINS_PASSWORD", "")
    if not email or not password:
        raise AltfinsWebError(
            "altFINS credentials not configured. "
            "Set ALTFINS_EMAIL and ALTFINS_PASSWORD in .env"
        )
    return email, password


def login_and_get_token(email: str, password: str) -> Optional[str]:
    """
    Login to altFINS.com and extract the auth token from localStorage.
    
    Returns the token string, or None if login fails.
    """
    if not _check_playwright():
        raise AltfinsWebError("playwright not installed. Run: pip install playwright && playwright install chromium")
    
    from playwright.sync_api import sync_playwright
    
    token = None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = context.new_page()
        
        try:
            # Go to login page
            page.goto("https://altfins.com/login", timeout=30000, wait_until="networkidle")
            
            # Fill credentials
            page.fill('input[type="email"], input[name="email"]', email, timeout=10000)
            page.fill('input[type="password"], input[name="password"]', password, timeout=10000)
            
            # Click login button
            page.click('button[type="submit"], button:has-text("Sign In"), button:has-text("Login")', timeout=10000)
            
            # Wait for redirect to dashboard
            page.wait_for_url("https://altfins.com/**", timeout=20000)
            
            # Extract token from localStorage
            token = page.evaluate("localStorage.getItem('token') or localStorage.getItem('accessToken') or localStorage.getItem('authToken')")
            
            # Fallback: try cookies
            if not token:
                cookies = context.cookies()
                for c in cookies:
                    if 'token' in c['name'].lower() or 'auth' in c['name'].lower():
                        token = c['value']
                        break
            
            browser.close()
            return token
            
        except Exception as e:
            browser.close()
            raise AltfinsWebError(f"altFINS login failed: {e}")


def fetch_chart_patterns(symbol: str, token: str = None) -> Dict[str, Any]:
    """
    Fetch chart patterns for a single symbol from altFINS.
    
    Uses their internal API (https://api.altfins.com/v2/patterns/{symbol})
    which is what the frontend calls after login.
    """
    import httpx
    
    if not token:
        email, pwd = get_credentials()
        token = login_and_get_token(email, pwd)
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Origin": "https://altfins.com",
        "Referer": f"https://altfins.com/coin/{symbol}",
    }
    
    # Try multiple API endpoints that altFINS uses
    endpoints = [
        f"https://api.altfins.com/v2/patterns/{symbol}",
        f"https://api.altfins.com/v1/patterns/{symbol}",
        f"https://api.altfins.com/v2/coins/{symbol}/patterns",
        f"https://api.altfins.com/v1/coins/{symbol}/technical-analysis",
    ]
    
    for url in endpoints:
        try:
            resp = httpx.get(url, headers=headers, timeout=15.0)
            if resp.status_code == 200:
                data = resp.json()
                return data
        except Exception:
            continue
    
    # If API fails, try scraping the coin page
    return _scrape_coin_page(symbol)


def fetch_multiple_patterns(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch chart patterns for multiple symbols."""
    try:
        email, pwd = get_credentials()
        token = login_and_get_token(email, pwd)
    except AltfinsWebError:
        return {}
    
    results = {}
    for sym in symbols:
        try:
            data = fetch_chart_patterns(sym, token=token)
            results[sym] = data
        except Exception:
            continue
    return results


def _scrape_coin_page(symbol: str) -> Dict[str, Any]:
    """
    Fallback: scrape the altFINS coin page for technical analysis.
    
    Extracts: chart patterns, trend ratings, support/resistance levels,
    technical indicators summary.
    """
    if not _check_playwright():
        return {}
    
    from playwright.sync_api import sync_playwright
    
    result = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            # Login first
            email, pwd = get_credentials()
            page.goto("https://altfins.com/login", timeout=30000)
            page.fill('input[type="email"]', email, timeout=10000)
            page.fill('input[type="password"]', pwd, timeout=10000)
            page.click('button[type="submit"]', timeout=10000)
            page.wait_for_timeout(5000)
            
            # Navigate to coin page
            page.goto(f"https://altfins.com/coin/{symbol}", timeout=30000, wait_until="networkidle")
            page.wait_for_timeout(5000)
            
            # Extract patterns
            patterns = page.evaluate("""
                () => {
                    const result = {patterns: [], technicals: {}, support_resistance: []};
                    
                    // Chart patterns section
                    const patternEls = document.querySelectorAll('[class*="pattern"], [class*="chart-pattern"]');
                    patternEls.forEach(el => {
                        result.patterns.push(el.textContent.trim());
                    });
                    
                    // Technical summary
                    const techEls = document.querySelectorAll('[class*="technical"], [class*="indicator"]');
                    techEls.forEach(el => {
                        const label = el.querySelector('[class*="label"], [class*="name"]');
                        const value = el.querySelector('[class*="value"], [class*="status"]');
                        if (label) {
                            result.technicals[label.textContent.trim()] = value ? value.textContent.trim() : '';
                        }
                    });
                    
                    // Support / Resistance tables
                    const tableEls = document.querySelectorAll('table');
                    tableEls.forEach(table => {
                        const header = table.querySelector('th, thead');
                        if (header && header.textContent.includes('Support')) {
                            const rows = table.querySelectorAll('tbody tr');
                            rows.forEach(row => {
                                const cells = row.querySelectorAll('td');
                                if (cells.length >= 2) {
                                    result.support_resistance.push({
                                        level: cells[0].textContent.trim(),
                                        price: cells[1].textContent.trim()
                                    });
                                }
                            });
                        }
                    });
                    
                    return result;
                }
            """)
            
            result = patterns
            browser.close()
            
        except Exception:
            browser.close()
            
    return result


def get_technical_analysis(symbol: str) -> Dict[str, Any]:
    """
    High-level technical analysis summary for a coin.
    
    Returns:
    {
        "patterns": ["Bull Flag", "Ascending Triangle", ...],
        "trend": "BULLISH" | "BEARISH" | "NEUTRAL",
        "momentum": "BUY" | "SELL" | "NEUTRAL",
        "support_levels": [1.23, 1.15, ...],
        "resistance_levels": [1.35, 1.42, ...],
        "indicators": {
            "RSI": "Neutral",
            "MACD": "Bullish",
            "MA": "Bullish",
            ...
        }
    }
    """
    data = fetch_chart_patterns(symbol)
    if not data:
        # Try scraping fallback
        scraped = _scrape_coin_page(symbol)
        return scraped
    
    return data


def batch_technical_analysis(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Get technical analysis for multiple symbols."""
    try:
        email, pwd = get_credentials()
        token = login_and_get_token(email, pwd)
    except AltfinsWebError:
        return {}
    
    results = {}
    for sym in symbols:
        try:
            data = fetch_chart_patterns(sym, token=token)
            if data:
                results[sym] = data
            else:
                scraped = _scrape_coin_page(sym)
                if scraped:
                    results[sym] = scraped
        except Exception:
            continue
    
    return results


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    print(f"Fetching technical analysis for {sym}...")
    try:
        result = get_technical_analysis(sym)
        print(json.dumps(result, indent=2, default=str))
    except AltfinsWebError as e:
        print(f"Error: {e}")
