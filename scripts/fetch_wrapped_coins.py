#!/usr/bin/env python3
"""One-shot: fetch ALL wrapped coins from CoinGecko and save to data/wrapped_coins.json.

Sources:
  1. /coins/markets?category=wrapped-tokens  (paginated, 250/page)
  2. /coins/list (full list) — name contains 'wrapped' or symbol starts with 'w'
     AND name contains 'wrapped' (catches coins missing from the category).

Output: data/wrapped_coins.json
  {"fetched_at": "...", "coins": [{"symbol","name","id","source"}...], "symbols": [...]}
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "wrapped_coins.json"
API = "https://api.coingecko.com/api/v3"

client = httpx.Client(timeout=25.0)
coins: dict[str, dict] = {}   # key: symbol.upper() -> info


def _get(url: str, params: dict) -> dict | None:
    for attempt in range(4):
        try:
            r = client.get(url, params=params)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = 65
                print(f"  429 — sleeping {wait}s (attempt {attempt+1}/4)")
                time.sleep(wait)
                continue
            print(f"  HTTP {r.status_code} on {url} — {r.text[:120]}")
            return None
        except Exception as e:
            print(f"  ERR {e}")
            time.sleep(5)
    return None


def add(symbol: str, name: str, cid: str, source: str) -> None:
    sym = symbol.upper()
    if not sym:
        return
    if sym not in coins:
        coins[sym] = {"symbol": sym, "name": name, "id": cid, "source": source}
    else:
        coins[sym]["source"] = f"{coins[sym]['source']}+{source}"


# ── 1) wrapped-tokens category (paginated) ──────────────────────────────
print("Fetching category=wrapped-tokens ...")
page = 1
while True:
    data = _get(f"{API}/coins/markets", {
        "vs_currency": "usd", "category": "wrapped-tokens",
        "order": "market_cap_desc", "per_page": 250, "page": page,
        "sparkline": "false",
    })
    if not data:
        break
    if len(data) == 0:
        break
    print(f"  page {page}: {len(data)} coins")
    for c in data:
        add(c.get("symbol", ""), c.get("name", ""), c.get("id", ""), "category")
    if len(data) < 250:
        break
    page += 1
    time.sleep(1.5)

# ── 2) full list name/symbol scan ───────────────────────────────────────
print("Fetching /coins/list (full list) ...")
full = _get(f"{API}/coins/list", {"include_platform": "false"})
if full:
    for c in full:
        name = c.get("name", "") or ""
        sym = c.get("symbol", "") or ""
        nlow = name.lower()
        # "Wrapped X" naming, or symbol prefixed w (wBTC, wETH) with wrapped name
        if "wrapped" in nlow or "w-" in nlow or nlow.startswith("wrapped"):
            add(sym, name, c.get("id", ""), "name")
        elif sym.lower().startswith("w") and len(sym) >= 4 and "wrapped" in nlow:
            add(sym, name, c.get("id", ""), "symbol+name")
    print(f"  full list: {len(full)} coins total")
else:
    print("  WARN: /coins/list failed — category result only")

# ── Save ────────────────────────────────────────────────────────────────
symbols = sorted(coins.keys())
out = {
    "fetched_at": datetime.now(timezone.utc).isoformat(),
    "count": len(symbols),
    "coins": list(coins.values()),
    "symbols": symbols,
}
OUT.write_text(json.dumps(out, indent=2))
print(f"\nSaved {len(symbols)} wrapped coin symbols -> {OUT}")
for s in symbols:
    print(f"  {s:<12} {coins[s]['name'][:40]}")
