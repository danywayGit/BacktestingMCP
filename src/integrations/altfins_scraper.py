"""
altFINS Web Scraper — chart patterns & technical analysis per coin.

Uses Playwright to login to altFINS.com and extract chart pattern data,
technical indicator summaries, and support/resistance levels that the
MCP server doesn't provide directly.

Usage:
    python altfins_scraper.py BTC        # Single coin
    python altfins_scraper.py BTC,ETH    # Multiple coins
"""

import os, json, re, sys, time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv('/home/hermes/BacktestingMCP/.env.altfins')
load_dotenv('/home/hermes/BacktestingMCP/.env')


class AltfinsScraper:
    """Scrape altFINS.com for chart patterns and technical analysis."""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._browser = None
        self._page = None

    def _get_credentials(self) -> Tuple[str, str]:
        email = os.getenv("ALTFINS_EMAIL", "")
        password = os.getenv("ALTFINS_PASSWORD", "")
        if not email or not password:
            raise ValueError("Set ALTFINS_EMAIL and ALTFINS_PASSWORD in .env.altfins")
        return email, password

    def login(self):
        """Login to altFINS.com via the Vaadin login form."""
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        context = self._browser.new_context(viewport={"width": 1280, "height": 900})
        self._page = context.new_page()
        page = self._page

        email, password = self._get_credentials()

        # Go to login page
        page.goto("https://altfins.com/login", timeout=60000, wait_until="networkidle")
        page.wait_for_timeout(3000)

        # Click "Sign in" to switch from registration to login form
        sign_in = page.query_selector('span.link:has-text("Sign in")')
        if sign_in:
            sign_in.click()
            page.wait_for_timeout(1000)
        else:
            # Try the Login tab
            login_tab = page.query_selector('span:has-text("Login")')
            if login_tab:
                login_tab.click()
                page.wait_for_timeout(1000)

        # Fill login form
        username_field = page.query_selector('#vaadinLoginUsername input, vaadin-text-field#vaadinLoginUsername input')
        if not username_field:
            username_field = page.query_selector('vaadin-text-field[id*="Login"] input, [id*="vaadinLogin"] input[slot="input"]')
        
        password_field = page.query_selector('#vaadinLoginPassword input, vaadin-password-field#vaadinLoginPassword input')
        if not password_field:
            password_field = page.query_selector('vaadin-password-field[id*="Login"] input, [id*="vaadinLogin"] input[type="password"]')

        if username_field and password_field:
            username_field.fill(email)
            password_field.fill(password)
            page.wait_for_timeout(500)

            # Click Login button
            login_btn = page.query_selector('vaadin-button:has-text("Login")')
            if login_btn:
                login_btn.click()
                page.wait_for_timeout(5000)
            
            # Wait for redirect to dashboard
            page.wait_for_url("https://altfins.com/**", timeout=30000)
            page.wait_for_timeout(2000)
        else:
            raise RuntimeError("Could not find login form fields")

    def get_coin_data(self, symbol: str) -> Dict[str, Any]:
        """Scrape pattern and technical analysis for a single coin."""
        if not self._page:
            self.login()

        page = self._page
        
        # Navigate to coin page
        try:
            page.goto(f"https://altfins.com/coin/{symbol}", timeout=30000, wait_until="networkidle")
            page.wait_for_timeout(5000)
        except Exception:
            # If navigation fails (e.g. 404), return empty
            return {"symbol": symbol, "error": f"Coin page not found for {symbol}"}

        # Extract data via JS
        data = page.evaluate("""
            (symbol) => {
                const result = {
                    symbol: symbol,
                    patterns: [],
                    technical_indicators: {},
                    trend: {short: null, medium: null, long: null},
                    support_resistance: [],
                    signals: [],
                    summary: {}
                };
                
                // 1. Chart patterns
                const patternElements = document.querySelectorAll('[class*="pattern"], [class*="chart-pattern-row"], [class*="signal-row"]');
                patternElements.forEach(el => {
                    const text = el.textContent.trim();
                    if (text && text.length > 2 && text.length < 200) {
                        result.patterns.push(text);
                    }
                });
                
                // 2. Technical indicators table
                const tables = document.querySelectorAll('vaadin-grid, table, [class*="table"], [class*="grid"]');
                tables.forEach(table => {
                    const rows = table.querySelectorAll('tr, [class*="row"]');
                    rows.forEach(row => {
                        const cells = row.querySelectorAll('td, [class*="cell"]');
                        if (cells.length >= 2) {
                            const key = cells[0].textContent.trim();
                            const val = cells[cells.length-1].textContent.trim();
                            if (key && val && key.length < 50) {
                                result.technical_indicators[key] = val;
                            }
                        }
                    });
                });
                
                // 3. Trend data
                const trendEls = document.querySelectorAll('[class*="trend"], [class*="rating"]');
                trendEls.forEach(el => {
                    const text = el.textContent.trim();
                    if (text.match(/BULLISH|BEARISH|NEUTRAL|BUY|SELL/i)) {
                        const parent = el.parentElement || {};
                        const label = (parent.querySelector('[class*="label"], [class*="name"]') || {}).textContent || '';
                        result.signals.push({label, value: text});
                    }
                });
                
                // 4. Summary box
                const summaryEl = document.querySelector('[class*="summary"], [class*="overview"], [class*="rating-box"]');
                if (summaryEl) {
                    result.summary.text = summaryEl.textContent.trim().slice(0, 1000);
                }
                
                return result;
            }
        """, symbol)

        return data

    def batch_get(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get data for multiple coins."""
        results = {}
        for sym in symbols:
            try:
                results[sym] = self.get_coin_data(sym)
                print(f"  ✅ {sym}: {len(results[sym].get('patterns', []))} patterns")
            except Exception as e:
                results[sym] = {"symbol": sym, "error": str(e)}
                print(f"  ❌ {sym}: {e}")
        return results

    def close(self):
        if self._browser:
            self._browser.close()
        if hasattr(self, '_playwright'):
            self._playwright.stop()


# CLI entry point
if __name__ == "__main__":
    import sys
    symbols = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BTC", "ETH"]
    
    scraper = AltfinsScraper(headless=True)
    try:
        print(f"Logging into altFINS...")
        scraper.login()
        print(f"✅ Logged in!\n")
        
        print(f"Fetching data for {symbols}...")
        results = scraper.batch_get(symbols)
        
        print(f"\n=== RESULTS ===")
        for sym, data in results.items():
            print(f"\n--- {sym} ---")
            patterns = data.get('patterns', [])
            indicators = data.get('technical_indicators', {})
            signals = data.get('signals', [])
            print(f"  Patterns: {len(patterns)}")
            for p in patterns[:5]:
                print(f"    • {p[:100]}")
            print(f"  Indicators: {len(indicators)}")
            for k, v in list(indicators.items())[:5]:
                print(f"    • {k}: {v}")
            print(f"  Signals: {len(signals)}")
            for s in signals[:3]:
                print(f"    • {s['label']}: {s['value']}")
        
        # Save to file
        with open(f"/home/hermes/BacktestingMCP/results/altfins_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✅ Saved to results/altfins_scan_*.json")
        
    finally:
        scraper.close()