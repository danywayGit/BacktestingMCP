"""altFINS Technical Analysis & Chart Patterns Scraper.

Extracts: Near Term Outlook, Pattern Type, Pattern Stage per coin
from the altFINS technical-analysis page (Vaadin grid).
"""
import sys, os, json, csv
sys.path.insert(0, '/home/hermes/BacktestingMCP')
from dotenv import load_dotenv
from typing import List, Dict, Any
from datetime import datetime, timezone

load_dotenv('/home/hermes/BacktestingMCP/.env.altfins')

class AltfinsTAScraper:
    """Scrapes altFINS technical analysis & chart patterns."""

    def __init__(self):
        self.email = os.getenv("ALTFINS_EMAIL", "")
        self.password = os.getenv("ALTFINS_PASSWORD", "")
        self._browser = None
        self._page = None

    def _ensure_login(self):
        from playwright.sync_api import sync_playwright
        if self._browser:
            return
        
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        context = self._browser.new_context(viewport={"width": 1280, "height": 900})
        self._page = context.new_page()
        page = self._page

        # Login
        page.goto("https://altfins.com/login", timeout=20000, wait_until="domcontentloaded")
        page.wait_for_timeout(2000)
        si = page.query_selector('span.link:has-text("Sign in")')
        if si: si.click(); page.wait_for_timeout(1000)
        page.fill('vaadin-text-field[id*="Login"] input', self.email, timeout=5000)
        page.fill('vaadin-password-field[id*="Login"] input', self.password, timeout=5000)
        page.click('vaadin-button:has-text("Login")', timeout=5000)
        page.wait_for_timeout(3000)

    def get_patterns(self) -> List[Dict[str, str]]:
        """Extract all chart patterns from the technical-analysis page."""
        self._ensure_login()
        page = self._page
        page.goto("https://altfins.com/technical-analysis", timeout=20000, wait_until="domcontentloaded")
        page.wait_for_timeout(5000)

        patterns = []
        for page_num in range(5):  # Max 5 pages
            # Extract grid rows from current page
            rows = page.evaluate("""
                () => {
                    const grid = document.querySelector('vaadin-grid');
                    if (!grid) return [];
                    
                    // Get ALL vaadin-grid-cell-content elements
                    const cells = grid.querySelectorAll('vaadin-grid-cell-content');
                    const texts = Array.from(cells).map(c => c.textContent.trim()).filter(t => t);
                    
                    // Each row has 7 cells: Date, Symbol, Name, View, Outlook, Pattern, Stage
                    const rows = [];
                    for (let i = 0; i < texts.length; i += 7) {
                        if (i + 7 <= texts.length) {
                            const row = {
                                date: texts[i],
                                symbol: texts[i+1],
                                name: texts[i+2],
                                view_url: texts[i+3],
                                outlook: texts[i+4],
                                pattern: texts[i+5],
                                stage: texts[i+6]
                            };
                            // Skip header rows
                            if (row.symbol !== 'Asset Symbol' && row.symbol.length <= 10) {
                                rows.push(row);
                            }
                        }
                    }
                    return rows;
                }
            """)
            
            patterns.extend(rows)
            
            # Try to go to next page
            next_btn = page.query_selector('vaadin-button[aria-label="Next"], vaadin-button:has-text("Next")')
            if not next_btn:
                # Try the pagination ">" button
                next_btn = page.query_selector('vaadin-button:not([disabled])[part="button"]:has-text("")')
            
            if next_btn:
                is_disabled = next_btn.get_attribute('disabled')
                if is_disabled is None:
                    next_btn.click()
                    page.wait_for_timeout(3000)
                else:
                    break
            else:
                break

        return patterns

    def close(self):
        if self._browser:
            self._browser.close()
        if hasattr(self, '_pw'):
            self._pw.stop()


if __name__ == "__main__":
    scraper = AltfinsTAScraper()
    try:
        print("Fetching altFINS technical analysis & chart patterns...")
        patterns = scraper.get_patterns()
        
        print(f"\n✅ Found {len(patterns)} patterns:\n")
        print(f"{'Symbol':<10} {'Outlook':<12} {'Pattern':<25} {'Stage':<12}")
        print("-" * 60)
        
        # Group by pattern type
        by_pattern = {}
        for p in patterns:
            sym = p['symbol']
            outlook = p['outlook']
            pattern = p['pattern']
            stage = p['stage']
            print(f"{sym:<10} {outlook:<12} {pattern:<25} {stage:<12}")
            
            key = pattern
            if key not in by_pattern:
                by_pattern[key] = []
            by_pattern[key].append(sym)
        
        # Summary by pattern type
        print(f"\n{'='*60}")
        print("Summary by pattern type:")
        for pattern, coins in sorted(by_pattern.items(), key=lambda x: -len(x[1])):
            print(f"  {pattern:<25} {len(coins):>3} coins: {', '.join(coins[:10])}")
        
        # Save to JSON
        output_file = f"/home/hermes/BacktestingMCP/results/altfins_ta_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(output_file, "w") as f:
            json.dump(patterns, f, indent=2)
        print(f"\n✅ Saved to {output_file}")
        
    finally:
        scraper.close()