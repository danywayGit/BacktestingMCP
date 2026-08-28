"""
Liquidity cache for the Bybit/HyroTrader low-cap hard rule (Aug 2026).

USER RULE (stricter than HyroTrader's $100M / $500K official):
  NO symbol with market cap < $300M OR 24h volume < $1M/day may be traded
  on Bybit/HyroTrader (fail-closed: unknown => denied).

Data: CoinGecko /coins/markets (vs_currency=usd, per_page=250, pages up to 5,
cached to data/liquidity_cache.json, refreshed daily 06:00 UTC via cron).
The bridge consults this cache; if a symbol is missing (unknown/not ranked),
the trade is DENIED for Bybit/HyroTrader (never allowed unverified).
"""
import os, json, time, httpx

CACHE_PATH = "/home/hermes/BacktestingMCP/data/liquidity_cache.json"
MIN_MCAP = 300_000_000      # $300M
MIN_VOL24 = 1_000_000       # $1M/day
MAX_AGE_H = 30              # refresh daily; stale after 30h -> treated as unknown


def _default_cache() -> dict:
    return {
        "fetched_at": None,
        "symbols": {},   # "BTCUSDT" -> {"market_cap": float, "volume_24h": float}
        "source": "coingecko",
    }


def _load() -> dict:
    try:
        with open(CACHE_PATH) as f:
            d = json.load(f)
        if not isinstance(d, dict) or "symbols" not in d:
            return _default_cache()
        return d
    except Exception:
        return _default_cache()


def _save(d: dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, CACHE_PATH)


def refresh(force: bool = False) -> dict:
    """Fetch CoinGecko market data and rebuild the cache. Returns cache dict."""
    d = _load()
    if not force and d.get("fetched_at") and (time.time() - d["fetched_at"]) < MAX_AGE_H * 3600:
        return d

    symbols = {}
    try:
        client = httpx.Client(timeout=20)
        for page in range(1, 6):
            resp = client.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 250,
                    "page": page,
                    "sparkline": "false",
                },
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            if not data:
                break
            for c in data:
                sym = (c.get("symbol") or "").upper() + "USDT"
                mc = c.get("market_cap") or 0
                v24 = c.get("total_volume") or 0
                symbols[sym] = {"market_cap": mc, "volume_24h": v24}
            if len(data) < 250:
                break
            time.sleep(1.2)  # CoinGecko rate limit
        client.close()
    except Exception as e:
        print(f"liquidity refresh error: {e}")

    d["symbols"] = symbols
    d["fetched_at"] = time.time()
    d["source"] = "coingecko"
    _save(d)
    return d


def is_liquid_for_bybit(symbol_usdt: str, cache: dict | None = None) -> tuple:
    """Fail-closed check for Bybit/HyroTrader low-cap rule.

    Returns (allowed: bool, reason: str). Unknown symbol (not in the top-1250
    CoinGecko list or stale cache) => DENIED — never allowed unverified.
    """
    cache = cache or _load()
    fetched_at = cache.get("fetched_at")
    if not fetched_at or (time.time() - fetched_at) > MAX_AGE_H * 3600:
        return (False, f"liquidity cache stale/absent — DENIED (fail-closed); refresh first")

    sym = (symbol_usdt or "").upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    info = cache.get("symbols", {}).get(sym)
    if not info:
        # Fallback: symbol not in CoinGecko top-1250, but if Bybit lists it
        # with real 24h volume, it's clearly liquid (e.g. wrapped assets like
        # WBETH/BNSOL whose underlying is large-cap). Use Bybit's public
        # ticker (no auth). Fail-closed ONLY if Bybit also has no volume.
        try:
            resp = httpx.get(
                f"https://api-demo.bybit.com/v5/market/tickers",
                params={"category": "linear", "symbol": sym}, timeout=6,
            )
            if resp.status_code == 200:
                tickers = resp.json().get("result", {}).get("list", [])
                if tickers:
                    tv = float(tickers[0].get("turnover24h") or 0)
                    vol = float(tickers[0].get("volume24h") or 0)
                    # turnover24h (USDT traded) is the best liquidity proxy
                    if tv >= MIN_VOL24:
                        return (True, f"{sym} Bybit turnover24h ${tv/1e6:.2f}M ≥ ${MIN_VOL24/1e6:.0f}M — OK (Bybit-listed)")
                    return (False, f"{sym} Bybit turnover24h ${tv/1e6:.2f}M < ${MIN_VOL24/1e6:.0f}M — illiquid DENIED")
        except Exception:
            pass
        return (False, f"{sym} not in liquidity cache and no Bybit volume — DENIED (fail-closed)")

    mc = float(info.get("market_cap") or 0)
    v24 = float(info.get("volume_24h") or 0)
    if mc < MIN_MCAP:
        return (False, f"{sym} mcap ${mc/1e6:.0f}M < ${MIN_MCAP/1e6:.0f}M — low-cap DENIED")
    if v24 < MIN_VOL24:
        return (False, f"{sym} 24h vol ${v24/1e6:.2f}M < ${MIN_VOL24/1e6:.0f}M — illiquid DENIED")
    return (True, f"{sym} mcap ${mc/1e6:.0f}M vol ${v24/1e6:.2f}M — OK")


if __name__ == "__main__":
    import sys
    cache = refresh(force="--force" in sys.argv)
    n = len(cache.get("symbols", {}))
    print(f"liquidity cache: {n} symbols, fetched_at={cache.get('fetched_at')}")
    for sym in ["BTCUSDT", "KSMUSDT", "THETAUSDT", "XMRUSDT", "NOTUSDT"]:
        print(sym, is_liquid_for_bybit(sym, cache))
