#!/usr/bin/env python3
"""Build the final wrapped-coin exclusion list (collision-safe).

Source of truth: CoinGecko's wrapped-tokens category (paginated), plus a
STRICT name rescue — coins whose name starts with "Wrapped " that are NOT
already in the category. No broad substring markers (they swept in real
tokens like CELR=Celer Network, MULTI=Multichain, ZRO=LayerZero).

Collision guard: a symbol is only excluded if NO real (non-wrapped) asset
carries that ticker. "Wrapped LUNC"/"Mezo Wrapped BTC" share tickers with
real LUNC/BTC — those real assets keep the symbol alive.

Output: data/wrapped_coins.json updated with symbols_to_exclude.
"""
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "wrapped_coins.json"
API = "https://api.coingecko.com/api/v3"

client = httpx.Client(timeout=25.0)


def _get(url: str, params: dict) -> dict | None:
    for attempt in range(4):
        try:
            r = client.get(url, params=params)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                print(f"  429 — sleep 65s (attempt {attempt+1}/4)")
                time.sleep(65)
                continue
            print(f"  HTTP {r.status_code} on {url}")
            return None
        except Exception as e:
            print(f"  ERR {e}")
            time.sleep(5)
    return None


# ── 1) wrapped-tokens category (paginated) — the authoritative list ───────
print("Fetching category=wrapped-tokens ...")
wrapped_symbols: set[str] = set()
page = 1
while True:
    data = _get(f"{API}/coins/markets", {
        "vs_currency": "usd", "category": "wrapped-tokens",
        "order": "market_cap_desc", "per_page": 250, "page": page,
        "sparkline": "false",
    })
    if not data or len(data) == 0:
        break
    print(f"  page {page}: {len(data)} coins")
    for c in data:
        sym = (c.get("symbol") or "").upper()
        if sym:
            wrapped_symbols.add(sym)
    if len(data) < 250:
        break
    page += 1
    time.sleep(1.5)
print(f"  category symbols: {len(wrapped_symbols)}")

# ── 2) real-asset detection from /coins/list ──────────────────────────────
# A symbol is a REAL asset if ANY coin with that symbol has a non-derivative
# name (e.g. LUNC has "Terra Classic" + "Wrapped LUNC" -> keep LUNC; WBTC has
# only "Wrapped Bitcoin"/"Bridged WBTC" -> exclude). Markers are precise:
# "wrapped"/"bridged"/"peg"/"staked"/"wormhole" catch derivatives, but NOT
# real tokens whose own names happen to contain words like celer/multichain/
# layerzero (CELR, MULTI, ZRO must stay).
DERIVATIVE_MARKERS = ("wrapped", "bridged", "bridge ", "peg", "staked",
                      "staking", "wormhole")

print("Fetching /coins/list ...")
full = _get(f"{API}/coins/list", {"include_platform": "false"})
real_symbols: set[str] = set()
name_rescue: set[str] = set()
if full:
    for c in full:
        sym = (c.get("symbol") or "").upper()
        name = (c.get("name") or "").strip().lower()
        if not sym:
            continue
        if name.startswith("wrapped "):
            name_rescue.add(sym)          # strict: "Wrapped X" not in category
        # A coin named exactly its own symbol (e.g. "WETH") is a placeholder
        # listing, not evidence of a real asset — only count it if the name
        # is informative and non-derivative.
        if name != sym.lower() and not any(m in name for m in DERIVATIVE_MARKERS):
            real_symbols.add(sym)         # real asset with this ticker
    print(f"  {len(full)} coins total; real symbols: {len(real_symbols)}; "
          f"name-rescue: {len(name_rescue)}")

# ── 3) collision-safe exclusion ──────────────────────────────────────────
wrapped_all = wrapped_symbols | name_rescue
exclude = wrapped_all - real_symbols
collisions = sorted(wrapped_all & real_symbols)
print(f"\nExclusion set: {len(exclude)} symbols (wrapped {len(wrapped_all)})")
print(f"Collisions kept (real asset exists): {len(collisions)}")
print("  " + ", ".join(collisions[:50]))

# sanity: key wrapped must be excluded, key base assets must be kept
must_block = ["WBETH", "WBTC", "WETH", "WBNB", "WSOL", "WSTETH", "WEETH",
              "WTAO", "CBBTC", "CBETH", "WNXM", "WOETH", "WRBTC", "STETH"]
must_keep = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "LINK", "LUNC",
             "W", "WIF", "WLD", "WAVES", "WOO", "WAXP", "OM", "XLM", "PROS",
             "USTC", "CC", "LUNA2", "CELR", "MULTI", "ZRO", "BNSOL"]
print("\nSanity checks:")
ok = True
for s in must_block:
    if s in exclude:
        print(f"  OK   block {s}")
    else:
        ok = False
        print(f"  FAIL block {s} (NOT excluded)")
for s in must_keep:
    if s not in exclude:
        print(f"  OK   keep {s}")
    else:
        ok = False
        print(f"  FAIL keep {s} (WRONGLY excluded)")

# ── 4) save ──────────────────────────────────────────────────────────────
saved = json.loads(OUT.read_text()) if OUT.exists() else {}
saved["symbols_to_exclude"] = sorted(exclude)
saved["collision_kept"] = collisions
saved["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
OUT.write_text(json.dumps(saved, indent=2))
print(f"\nSaved {len(exclude)} symbols -> {OUT}")
print("\nRESULT:", "ALL OK" if ok else "FAILURES")
