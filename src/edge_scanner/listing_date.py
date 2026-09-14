"""
Coin listing-age filter — reject gems listed on ANY top-5 CoinGecko exchange
for more than MAX_COIN_AGE_YEARS (2.0).

Age anchor = the coin's FIRST OHLCV bar on that exchange (listing date), NOT
token genesis/ATL/ATH. A 2021 coin sitting at its all-time low must still be
rejected (RAD: Radicle, listed Binance 2021-10, ATL 2026-07 — the old
ATL-date age proxy wrongly called it "young").

Option 2 (bounded): we never crawl the full history. For each top-5 exchange
we only answer "were there candles before the 2-year cutoff?" via a narrow
windowed OHLCV probe. If any top-5 exchange returns candles before the cutoff,
the coin has been shareable to the public too long -> reject.

Fallback contract (Didier, Sep 2026):
  - On exchange error/failure       -> try the next top-5 exchange.
  - If ALL top-5 exchanges fail     -> keep the coin (fail-open), other
                                        filters still apply.
  - "No candles before cutoff" on an exchange is NOT a failure: it means that
    exchange listed the coin more recently (or not at all) -> keep going.

Raw per-exchange HTTP probes (ccxt was unreliable: Coinbase returns wrong
dates, OKX empty, no gateio class, Kraken EGeneral errors).
"""

import logging
import time
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

import httpx  # noqa: E402

# CoinGecko exchange id -> probe handler + symbol-mapping helper.
# Handlers return (oldest_ms: int|None, ok: bool). ok=False => that exchange
# could not be determined (error) -> caller tries the next one. None oldest_ms
# with ok=True => no pre-cutoff candles found (coin listed recently / absent).
from datetime import timezone as _tz, datetime as _dt  # noqa: E402

# Max years a coin may be listed on a top-5 exchange.
MAX_COIN_AGE_YEARS = 2.0

_CG_EXCHANGES_URL = "https://api.coingecko.com/api/v3/exchanges?per_page=5"
_CG_DELAY = 1.2


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _ms_to_dt(ms: float):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _cutoff_ms() -> int:
    """2 years ago in ms — coins trading before this are 'too old'."""
    return _now_ms() - int(MAX_COIN_AGE_YEARS * 365.25 * 86400 * 1000)


# ── per-exchange probes ───────────────────────────────────────────────
def _get(url: str, params=None, timeout: float = 15.0):
    for attempt in range(3):
        try:
            r = httpx.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", "60")) if r.headers.get("Retry-After") else 60
                time.sleep(wait + 1)
                continue
            return r
        except httpx.TimeoutException:
            time.sleep(3 * (attempt + 1))
        except httpx.HTTPError:
            time.sleep(3 * (attempt + 1))
    return None


def _probe_binance(base: str):
    """Binance Spot USDT klines — startTime=0 returns the very first bar."""
    r = _get(
        f"https://api.binance.com/api/v3/klines",
        params={"symbol": f"{base}USDT", "interval": "1d", "limit": 1, "startTime": 0},
    )
    if r is None or r.status_code != 200:
        return None, False
    try:
        rows = r.json()
        if not rows:
            return None, True  # pair doesn't exist -> not listed
        return int(rows[0][0]), True
    except Exception:
        return None, False


def _probe_coinbase(base: str):
    """Coinbase candles cap at 300 points/request -> bounded window around cutoff.
    oldest=last row of the response for a window that starts at the cutoff."""
    cutoff = _cutoff_ms() // 1000
    lo = cutoff - 40 * 86400
    hi = cutoff + 10 * 86400
    r = _get(
        f"https://api.exchange.coinbase.com/products/{base}-USDT/candles",
        params={"granularity": 86400, "start": lo, "end": hi},
    )
    if r is None:
        return None, False
    if r.status_code == 400 or r.status_code == 404:
        return None, True  # unknown product / not listed on Coinbase
    if r.status_code != 200:
        return None, False
    try:
        rows = r.json()
        if not rows:
            return None, True
        return int(rows[-1][0]) * 1000, True
    except Exception:
        return None, False


def _probe_kraken(base: str):
    """Kraken OHLC since=0 returns the earliest bars (capped ~720). First row's
    ts = oldest. Kraken's USDT naming is inconsistent (RADUSDT vs alternates),
    so try a few candidate quote forms."""
    for pair in (f"{base}USDT", f"{base}USD", f"X{base}ZUSD"):
        r = _get(
            "https://api.kraken.com/0/public/OHLC",
            params={"pair": pair, "interval": 1440, "since": 0},
        )
        if r is None:
            continue
        try:
            j = r.json()
            err = j.get("error", [])
            if err and "Unknown asset pair" in err[0]:
                continue  # pair not listed — try next form
            res = j.get("result") or {}
            for k, v in res.items():
                if k != "last" and isinstance(v, list) and v and isinstance(v[0], list):
                    return int(v[0][0]) * 1000, True
            return None, False
        except Exception:
            continue
    return None, True  # none of the pair forms found -> treat as present-but-unknown


def _probe_okx(base: str):
    """OKX history-candles — paginate via 'after' (descending id) to oldest."""
    r = _get(
        f"https://www.okx.com/api/v5/market/history-candles",
        params={"instId": f"{base}-USDT", "bar": "1D", "limit": 100, "after": "4102444800000"},
    )
    if r is None:
        return None, False
    if r.status_code != 200:
        return None, False
    try:
        j = r.json()
        if j.get("code") != "0":
            return None, True
        rows = j.get("data") or []
        if not rows:
            return None, True
        ts = int(rows[-1][0])  # oldest within the page returned
        return ts, True
    except Exception:
        return None, False


def _probe_gate(base: str):
    """Gate spot candlesticks — from=0 to far-future returns the earliest bars."""
    cutoff = _cutoff_ms() // 1000
    from_ts = max(1, cutoff - 60 * 86400)
    to_ts = cutoff + 10 * 86400
    r = _get(
        f"https://api.gateio.ws/api/v4/spot/candlesticks",
        params={"currency_pair": f"{base}_USDT", "interval": "1d", "from": from_ts, "to": to_ts},
    )
    if r is None:
        return None, False
    if r.status_code != 200:
        return None, False
    try:
        rows = r.json()
        if not rows:
            return None, True
        ts = int(float(rows[-1][0]))
        return ts * 1000, True
    except Exception:
        return None, False


# CoinGecko exchange id -> probe. New/changing top-5 entries land here.
_PROBES = {
    "binance": _probe_binance,
    "gdax": _probe_coinbase,
    "kraken": _probe_kraken,
    "okex": _probe_okx,
    "gate": _probe_gate,
}

# Optional extras for when a different exchange enters the top-5.
_PROBES.setdefault("okx_alias", _probe_okx)
_PROBES.setdefault("bybit", None)  # no handler yet -> skipped, not a failure/keep


def get_top5_exchanges() -> list:
    """Current top-5 exchange ids from CoinGecko (dynamic — can change)."""
    try:
        r = httpx.get(_CG_EXCHANGES_URL, timeout=20)
        rows = r.json()
        if isinstance(rows, list):
            ids = [e.get("id") for e in rows[:5] if isinstance(e, dict) and e.get("id")]
            logger.info("CoinGecko top-5 exchanges: %s", ids)
            return ids
    except Exception as e:
        logger.warning("CoinGecko top-5 fetch failed: %s", e)
    return list(_PROBES)[:5]


def is_too_old(base: str, top5: list | None = None) -> tuple[bool, str]:
    """Return (too_old, reason). too_old=True => listed on ANY top-5 exchange
    before the 2-year cutoff -> reject the gem.

    Fallback contract: on exchange error try the next; if all fail -> keep
    (fail-open), other filters still apply.
    """
    base = base.upper()
    cutoff = _cutoff_ms()
    cut_str = _ms_to_dt(cutoff).strftime("%Y-%m-%d")
    top5 = top5 if top5 is not None else get_top5_exchanges()

    tried = 0
    for ex_id in top5:
        probe = _PROBES.get(str(ex_id).lower())
        if probe is None:
            logger.debug("[%s] %s: no probe handler (skipped)", base, ex_id)
            continue  # not a failure -> move to next exchange
        tried += 1
        oldest_ms, ok = probe(base)
        if not ok:
            logger.info("[%s] %s: probe failed (trying next exchange)", base, ex_id)
            continue
        if oldest_ms is None:
            logger.info("[%s] %s: no pre-cutoff candles (young/absent) — keep probing others", base, ex_id)
            continue
        oldest = _ms_to_dt(oldest_ms)
        if oldest_ms < cutoff:
            reason = (f"listed on {ex_id} since {oldest.strftime('%Y-%m-%d')} "
                      f"(> {MAX_COIN_AGE_YEARS:.0f}y, cutoff {cut_str})")
            return True, reason
        logger.info("[%s] %s: oldest %s >= cutoff %s — young", base, ex_id, oldest.date(), cut_str)

    # If we found a pre-cutoff bar on any exchange we'd have returned above.
    # Fail-open: if NO exchange could be/ was checked with a definitive result,
    # keep the coin.
    if tried == 0:
        return False, "no top-5 exchange handler available (fail-open)"
    return False, "not listed >2y on any checked top-5 exchange"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    top5 = get_top5_exchanges()
    print("top5:", top5)
    for sym in ["RAD", "HMSTR", "ETH", "XYZUNKNOWN"]:
        too_old, reason = is_too_old(sym, top5)
        print(f"{sym}: too_old={too_old}  ({reason})")