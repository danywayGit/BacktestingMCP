"""
Real-time Binance liquidation feed via public WebSocket.

Binance's !forceOrder@arr stream broadcasts every forced liquidation
across ALL USDT-M futures pairs — no API key required.

Because liquidations are sporadic, this runs as a REST-based fetch
alternative. It reads the ws snapshot or falls back to the free
takerlongshortRatio endpoint.

Auth: none required
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Paths resolve relative to src/integrations/ — two levels up = repo root.
# (The old "../../.." went 3 levels up to ~/ which is why the snapshot
# landed in ~/data/ instead of BacktestingMCP/data/ — the docs always
# said data/liquidation_snapshot.json.)
_SNAPSHOT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "liquidation_snapshot.json")
)
_REST_CACHE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "liquidation_rest_cache.json")
)
_WINDOW_SEC = 3600  # 1h
_HEARTBEAT_SEC = 60  # daemon writes snapshot even with zero events

# Free Binance REST endpoints (no key needed)
LS_RATIO_URL = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
TAKER_RATIO_URL = "https://fapi.binance.com/futures/data/takerlongshortRatio"
TOP_LS_URL = "https://fapi.binance.com/futures/data/topLongShortAccountRatio"

# Cache for REST data — in-memory fast layer + disk-persisted layer
# so each 5-min scan process does NOT re-fetch per-symbol (the old
# per-process cache caused a REST storm: every scan cycle cold-started
# and called takerlongshortRatio + globalLongShortAccountRatio for
# every scored symbol → scan timeout (exit 124)).
_cache: Dict[str, dict] = {}
_cache_time: Optional[datetime] = None
CACHE_TTL = 120  # 2 min in-process
REST_CACHE_TTL = 600  # 10 min disk-persisted (ratios change slowly)


def _load_rest_cache() -> dict:
    try:
        with open(_REST_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_rest_cache(cache: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_REST_CACHE_PATH), exist_ok=True)
        with open(_REST_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def _cached_get(url: str, params: dict, cache_key: str) -> Optional[list]:
    """GET with disk-persisted cache so data survives process restarts.

    Returns the JSON list payload, or None on miss/error.
    """
    cache = _load_rest_cache()
    now = time.time()
    entry = cache.get(cache_key)
    if entry and now - entry.get("t", 0) < REST_CACHE_TTL:
        return entry.get("data")
    try:
        resp = httpx.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            cache[cache_key] = {"t": now, "data": data}
            _save_rest_cache(cache)
            return data
    except Exception:
        pass
    return None


def read_snapshot() -> List[dict]:
    """Read liquidation events accumulated by the background WS daemon.

    Only returns events if the daemon has written a heartbeat recently
    (file mtime < 5 min). A stale file means the daemon is dead — we
    must NOT trust its old events.
    """
    try:
        mtime = os.path.getmtime(_SNAPSHOT_PATH)
        if time.time() - mtime > 300:  # 5 min without heartbeat = daemon dead
            return []
        with open(_SNAPSHOT_PATH) as f:
            data = json.load(f)
        now = time.time()
        return [e for e in data.get("events", []) if e.get("t", 0) >= now - _WINDOW_SEC]
    except Exception:
        return []


def _write_snapshot(events: List[dict]) -> None:
    """Persist the rolling event list with a timestamp (heartbeat)."""
    try:
        os.makedirs(os.path.dirname(_SNAPSHOT_PATH), exist_ok=True)
        with open(_SNAPSHOT_PATH, "w") as f:
            json.dump({"ts": time.time(), "events": events}, f)
    except Exception:
        pass


def get_liquidation_score(symbol: str) -> Tuple[float, dict]:
    """Liquidation score from best available source.

    Priority:
    1. WebSocket snapshot (real liquidations, if daemon is running)
    2. Taker buy/sell ratio + L/S account ratio (free REST, always works)
    """
    clean = symbol.replace("USDT", "").upper()

    # 1. Try WS snapshot first (real liquidation events)
    events = read_snapshot()
    sym_events = [e for e in events if e["s"].upper() == f"{clean}USDT" or e["s"].upper() == symbol.upper()]
    if sym_events:
        long_vol = sum(e["v"] for e in sym_events if e["S"] == "SELL")
        short_vol = sum(e["v"] for e in sym_events if e["S"] == "BUY")
        total = long_vol + short_vol
        if total > 0:
            imbalance = (short_vol - long_vol) / total
            scale = min(1.0 + (total / 1_000_000) ** 0.5, 5.0)
            score = imbalance * scale
            return round(score, 3), {
                "liq_data_source": "binance_ws",
                "liq_events": len(sym_events),
                "liq_long_volume": round(long_vol, 0),
                "liq_short_volume": round(short_vol, 0),
                "liq_total_volume": round(total, 0),
                "liq_imbalance": round(imbalance, 3),
                "liq_pressure_score": round(score, 3),
            }

    # 2. Fallback: taker buy/sell ratio (orderflow proxy for liquidation pressure)
    try:
        data = _cached_get(TAKER_RATIO_URL,
                           {"symbol": f"{clean}USDT", "period": "1h", "limit": 2},
                           f"taker:{clean}USDT")
        if data:
            cur = data[-1]
            ratio = float(cur.get("buySellRatio", 1.0))
            buy_vol = float(cur.get("buyVol", 0))
            sell_vol = float(cur.get("sellVol", 0))
            # ratio > 1 = more buyers than sellers = bullish pressure
            # ratio < 1 = more sellers than buyers = bearish pressure
            total_vol = buy_vol + sell_vol
            if total_vol > 0 and ratio != 1.0:
                # Normalise: ratio 1.3 → +0.3, ratio 0.7 → -0.3
                raw = (ratio - 1.0) / 1.0
                scale = min(1.0 + (total_vol * float(cur.get("price", 0) if "price" in cur else 1) / 50_000_000), 3.0)
                score = raw * scale * 2.0  # amplify
                return round(score, 3), {
                    "liq_data_source": "binance_taker",
                    "liq_taker_buy_vol": round(buy_vol, 0),
                    "liq_taker_sell_vol": round(sell_vol, 0),
                    "liq_taker_ratio": round(ratio, 3),
                    "liq_pressure_score": round(score, 3),
                }
    except Exception:
        pass

    # 3. Final fallback: L/S account ratio
    try:
        data = _cached_get(LS_RATIO_URL,
                           {"symbol": f"{clean}USDT", "period": "1h", "limit": 2},
                           f"ls:{clean}USDT")
        if data:
            cur = data[-1]
            ratio = float(cur.get("longShortRatio", 1.0))
            score = 0.0
            if ratio > 1.5:
                score = -(ratio - 1.5) / 1.5 * 2.0
            elif ratio < 0.7:
                score = (0.7 - ratio) / 0.7 * 2.0
            return round(score, 3), {
                "liq_data_source": "binance_ls_ratio",
                "liq_long_short_ratio": round(ratio, 3),
                "liq_pressure_score": score,
            }
    except Exception:
        pass

    return 0.0, {"liq_data_source": "unavailable"}


def get_liquidation_cached(symbol: str) -> Tuple[float, dict]:
    """Cached version."""
    global _cache, _cache_time
    now = datetime.now(timezone.utc)
    if _cache and _cache_time and (now - _cache_time).total_seconds() < CACHE_TTL and symbol in _cache:
        return _cache[symbol]["score"], _cache[symbol]["components"]
    score, comp = get_liquidation_score(symbol)
    _cache[symbol] = {"score": score, "components": comp}
    _cache_time = now
    return score, comp


def run_ws_daemon():
    """Run the persistent WS listener (call from cron / nohup)."""
    import asyncio
    async def _listen():
        try:
            import websockets
        except ImportError:
            logger.error("websockets not installed — daemon can't start")
            return
        events: List[dict] = []
        while True:
            try:
                async with websockets.connect("wss://fstream.binance.com/stream?streams=!forceOrder@arr", open_timeout=15, ping_interval=20) as ws:
                    logger.info("WS liquidation daemon connected")
                    last_heartbeat = time.time()
                    # Heartbeat on connect so a quiet market still writes a
                    # fresh snapshot — read_snapshot() then knows the daemon
                    # is alive (vs dead) by file mtime.
                    _write_snapshot(events)
                    while True:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=_HEARTBEAT_SEC)
                            o = json.loads(msg).get("o", {})
                            sym, side, price, qty = o.get("s", ""), o.get("S", ""), float(o.get("p", 0) or 0), float(o.get("q", 0) or 0)
                            if sym and price > 0:
                                events.append({"s": sym, "S": side, "p": price, "q": qty, "v": price * qty, "t": time.time()})
                                # keep 1h
                                cutoff = time.time() - _WINDOW_SEC
                                events[:] = [e for e in events if e["t"] >= cutoff]
                                _write_snapshot(events)
                        except asyncio.TimeoutError:
                            # No liquidation in the window — still heartbeat so
                            # consumers can distinguish alive-but-quiet vs dead.
                            pass
                        # Periodic heartbeat regardless of event flow
                        if time.time() - last_heartbeat >= _HEARTBEAT_SEC:
                            _write_snapshot(events)
                            last_heartbeat = time.time()
            except Exception as e:
                logger.warning("WS error, reconnecting: %s", e)
                await asyncio.sleep(3)

    asyncio.run(_listen())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run_ws_daemon()


if __name__ == "__main__":
    main()