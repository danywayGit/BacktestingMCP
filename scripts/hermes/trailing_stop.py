"""
Trailing Stop Engine (shared) — runs on HERMES, coordinates the EXECUTION bot.

Architecture (per Didier):
  - HERMES (this, cron every 3 min): reads open positions via the bot's
    dashboard API (/api/bybit_risk for Bybit, /api/positions for others),
    computes R (price vs original entry/SL), decides when to trail the stop,
    and POSTs a webhook with Action=MoveStopLoss to the execution bot.
  - EXECUTION BOT: holds exchange API secrets, receives the MoveStopLoss
    webhook, and moves the SL on the exchange via its adapters.

Rule (R-anchored, standard, user-confirmed):
  - R = (price - entry) / (entry - SL0)  for LONG  (inverted for SHORT)
  - When R >= +1.0R  -> move SL to +0.5R  (locks a little profit past BE)
  - Then trail: keep SL at 0.5R behind the peak price, only ever FORWARD.

Per-exchange config: each exchange may need different account_type and
position-source. Same logic, per-exchange adjustment.
"""
import json, os, sqlite3, urllib.request, urllib.parse, time

# ── Stats DB (Hermes crypto.db — alongside strategy stats) ──
CRYPTO_DB = "/home/hermes/BacktestingMCP/data/crypto.db"
CLOSED_TRADES_LOOKBACK_DAYS = 30  # how far back to confirm closure in bot ledger

# ── Config ──
API_BASE = "http://109.123.229.200"
# Bot webhook key — live in gitignored ~/.hermes/.env as TRAILING_BOT_API_KEY
# (NEVER hardcode a credential in tracked code — git-secret-guard rule).
API_KEY = os.environ.get("TRAILING_BOT_API_KEY", "")
WEBHOOK_URL = f"{API_BASE}/webhook"
STATE_FILE = os.path.expanduser("~/.hermes/scripts/.trailing_stop_state.json")

# Rule params (global, apply to all exchanges)
BE_TRIGGER_R = 1.0     # move SL when price reaches +1R
LOCK_R = 0.5           # initial SL lock = +0.5R (a little past break-even)
TRAIL_BUFFER_R = 0.5   # trail stop 0.5R behind peak, forever

# Per-exchange scan targets: exchange -> (account_type, position-source, username, user_id)
# position-source 'bybit_risk' uses /api/bybit_risk; 'positions' uses /api/positions.
# Username/user_id match the per-firm user split (see trading-webhook-bridge skill):
#   1=Danyway (Binance), 43=Danyway_HyroTrader (Bybit Demo 10k),
#   44=Danyway_Bitfunded (15k), 45=Danyway_Velotrade (10k).
EXCHANGES = {
    "Bybit":     {"account_type": "Demo",    "source": "bybit_risk",     "username": "Danyway_HyroTrader", "user_id": 43},
    "Binance":   {"account_type": "TestNet", "source": "positions",      "username": "Danyway",            "user_id": 1},
    "Velotrade": {"account_type": "Standard", "source": "positions",     "username": "Danyway_Velotrade",  "user_id": 45},
    "Bitfunded": {"account_type": "Standard", "source": "bitfunded_risk", "username": "Danyway_Bitfunded",  "user_id": 44},
}


def _http_get(path):
    with urllib.request.urlopen(f"{API_BASE}{path}", timeout=15) as r:
        return json.loads(r.read().decode())


def _http_get_raw(url):
    """Fetch a full external URL (not the bot API). Returns parsed JSON."""
    with urllib.request.urlopen(url, timeout=12) as r:
        return json.loads(r.read().decode())


def _http_post_webhook(msg_str):
    payload = {"key": API_KEY, "telegram_alert_type": "trading_bot", "msg": msg_str}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(WEBHOOK_URL, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read().decode()


def _load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"sl_moves": {}}  # {exchange:symbol: {current_sl, peak}}


def _save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f)


# A "position" here is a plain dict with a COMMON shape regardless of source:
#   {symbol, symbol_full, side, entry, size, uPnL, stop, take, price}
# Each source has its own parser normalizing its API fields onto that shape.

def _parse_bybit_position(p: dict) -> dict:
    """Normalize one Bybit /api/bybit_risk position row."""
    return {
        "symbol": p.get("symbol", "").replace("USDT", ""),
        "symbol_full": p.get("symbol", ""),
        "side": "LONG" if p.get("side") == "BUY" else "SHORT",
        "entry": p.get("entry"),
        "size": p.get("size"),
        "uPnL": p.get("uPnL"),
        "stop": p.get("StopLoss"),       # from DB enrichment (or live SL)
        "take": p.get("TakeProfit"),
        "price": p.get("price"),          # current mark price from bot
    }


def _parse_bitfunded_position(p: dict) -> dict:
    """Normalize one Bitfunded /api/bitfunded_risk row.

    Row fields: instrument, direction (long/short), openPrice, quantity,
    SL/TP (stopLossPrice / stopProfitPrice, may be ABSENT if unprotected).
    """
    sym = str(p.get("instrument") or p.get("symbol") or "")
    side_raw = str(p.get("direction") or p.get("side") or "").lower()
    return {
        "symbol": sym.replace("USDT", "").replace("USD", ""),
        "symbol_full": sym,
        "side": "LONG" if side_raw in ("long", "buy", "1") else "SHORT",
        "entry": p.get("openPrice") or p.get("avgPrice") or p.get("entry"),
        "size": p.get("quantity") or p.get("amount") or p.get("size"),
        "uPnL": p.get("unrealisedPnl") or p.get("profitPnl") or p.get("uPnL"),
        "stop": p.get("stopLossPrice") or p.get("stopLoss") or p.get("StopLoss"),
        "take": p.get("stopProfitPrice") or p.get("takeProfit") or p.get("TakeProfit"),
        "price": p.get("markPrice") or p.get("lastPrice") or p.get("price") or None,
    }


def _parse_positions_position(p: dict) -> dict:
    """Normalize one /api/positions row (Binance / Velotrade)."""
    return {
        "symbol": str(p.get("Symbol", "")).replace("USDT", ""),
        "symbol_full": p.get("Symbol", ""),
        "side": "LONG" if str(p.get("Side", "")).upper() == "BUY" else "SHORT",
        "entry": p.get("Entry"),
        "size": p.get("Quantity"),
        "uPnL": p.get("UnrealisedPnL"),
        "stop": p.get("StopLoss"),
        "take": p.get("TakeProfit"),
        "price": p.get("Price"),
    }


def get_open_positions(exchange, account_type, source, cfg_user_id=1):
    """Fetch open positions with entry/SL/TP via the bot API, normalized to a
    common shape. Selects the right bot endpoint + row parser per source."""
    if source == "bybit_risk":
        url = (f"/api/bybit_risk?key={API_KEY}&user_id={cfg_user_id}"
               f"&account_type={account_type}&initial_balance=10000")
        rows = _http_get(url).get("open_positions", [])
        parse = _parse_bybit_position
    elif source == "bitfunded_risk":
        url = (f"/api/bitfunded_risk?key={API_KEY}&user_id={cfg_user_id}"
               f"&account_type={account_type}&initial_balance=15000")
        rows = _http_get(url).get("open_positions", [])
        parse = _parse_bitfunded_position
    else:
        # /api/positions (Binance / Velotrade), filtered by exchange + user.
        url = (f"/api/positions?key={API_KEY}&user={cfg_user_id}"
               f"&exchange={exchange.lower()}")
        rows = _http_get(url).get("positions", [])
        parse = _parse_positions_position
    return [parse(p) for p in rows]


def _price_decimals(symbol_full, cached={}):
    """Return the price decimals the exchange will store for an SL, so a
    reconcile doesn't re-send a no-op over sub-tick rounding (e.g. proposed
    2477.325 vs exchange-stored 2477.32). Tries Binance public futures
    exchangeInfo (keyless); falls back to decimals by price magnitude."""
    sym = symbol_full.upper()
    if sym in cached:
        return cached[sym]
    dec = None
    try:
        with urllib.request.urlopen(
            "https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=12) as r:
            d = json.loads(r.read().decode())
            for s in d.get("symbols", []):
                if s.get("symbol") == sym:
                    for f in s.get("filters", []):
                        if f.get("filterType") == "PRICE_FILTER":
                            t = f.get("tickSize")
                            if t:
                                dec = max(0, len(str(t).rstrip('0').split('.')[-1]))
                            break
                    break
    except Exception:
        pass
    if dec is None:
        # Fallback when exchangeInfo is unreachable or the symbol isn't found.
        # 6 dp is safe for every major (the bot's get_price_precision_futures
        # rounds finer than this, so roundtrip comparisons stay correct).
        dec = 6
    cached[sym] = dec
    return dec


def _sl_equal(a, b, dec):
    """True if a and b are equal once rounded to the symbol's price decimals."""
    return round(a, dec) == round(b, dec)


def get_live_price(symbol_full):
    """Fetch current price for any symbol from a public, reliable source
    reachable from Hermes. Tries Binance public API first (resolves fine from
    Hermes), then Bybit public, then CoinGecko. Works for Bybit/Binance/Velotrade
    symbols (by stripping the base)."""
    base = symbol_full.replace("USDT", "").replace("USD", "")
    sources = [
        f"https://api.binance.com/api/v3/ticker/price?symbol={symbol_full}",
        f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol_full}",
        f"https://api.coingecko.com/api/v3/simple/price?ids={base.lower()}&vs_currencies=usd",
    ]
    for url in sources:
        try:
            d = _http_get_raw(url)
            if "binance" in url and "price" in d:
                return float(d["price"])
            if "bybit" in url:
                lst = d.get("result", {}).get("list", [])
                if lst:
                    return float(lst[0].get("lastPrice") or lst[0].get("markPrice") or 0)
            if "coingecko" in url:
                v = d.get(base.lower(), {}).get("usd")
                if v:
                    return float(v)
        except Exception:
            continue
    return None


def compute_trail(side, entry, sl0, current_price, peak, current_sl):
    """Return new_sl (or None if no move). R-anchored trail.

    SL only moves ONCE the position has reached +1R (the break-even trigger).
    Below +1R, the SL stays at its original level (full risk). At/after +1R:
      - move SL to at least +0.5R (lock a little profit past break-even)
      - then trail at 0.5R behind the peak, only ever forward.
    Returns a tuple (new_sl_or_None, peak_stale) where peak_stale=True means the
    proposed trail lands on the WRONG side of the current price (a LONG SL at/above
    market, or SHORT SL at/below market) — i.e. the retained peak is from a move
    that never actually applied (e.g. the -4045 bug epoch). Caller must then reset
    the peak to the current price so trailing resumes from reality instead of
    infinitely re-sending an unplaceable order.
    """
    if not entry or not sl0 or not current_price:
        return None, False
    if side == "LONG":
        r_dist = entry - sl0
        if r_dist <= 0:
            return None, False
        r = (current_price - entry) / r_dist
        # Do NOT trail below the +1R break-even trigger.
        if r < BE_TRIGGER_R:
            return None, False
        peak = max(peak, current_price)
        trail_sl = peak - TRAIL_BUFFER_R * r_dist
        # At +1R: ensure SL at least +0.5R above entry (locks a little profit)
        min_sl = entry + LOCK_R * r_dist
        # Only forward (never below current SL). But never let a CORRUPT current_sl
        # (a LONG SL already at/above market — e.g. persisted from an async-rejected
        # move) act as a floor; that would force new_sl >= market forever (the
        # infinite unplaceable-SL loop observed on UNI). A LONG SL floor must sit
        # strictly below market.
        cur_floor = current_sl if (current_sl and current_sl < current_price) else sl0
        new_sl = max(trail_sl, min_sl, cur_floor)
        # A LONG SL must sit BELOW the current price or Binance rejects it
        # ("would immediately trigger") — if the retained peak puts it at/above
        # market, the peak is stale (a prior move never applied). Flag it.
        if new_sl >= current_price:
            return None, True
    else:  # SHORT
        r_dist = sl0 - entry
        if r_dist <= 0:
            return None, False
        r = (entry - current_price) / r_dist
        if r < BE_TRIGGER_R:
            return None, False
        peak = min(peak, current_price)
        trail_sl = peak + TRAIL_BUFFER_R * r_dist
        min_sl = entry - LOCK_R * r_dist
        # Mirror of the LONG branch: never let a corrupt current_sl (a SHORT SL
        # already at/below market — persisted from an async-rejected move) act as
        # a floor; that forces new_sl <= market forever. A SHORT SL floor must sit
        # strictly above market.
        cur_floor = current_sl if (current_sl and current_sl > current_price) else sl0
        new_sl = min(trail_sl, min_sl, cur_floor)
        if new_sl <= current_price:
            return None, True
    return new_sl, False


def get_live_sls(exchange):
    """Fetch the LIVE on-exchange SL map {SYM_full: {side, entry, sl, tp}} for
    an exchange from the bot's /api/live_sl endpoint. This reads the actual
    working stop/conditional orders, NOT the DB (which can be stale). The
    engine uses it as the authoritative no-op guard so it never re-POSTs an
    identical MoveStopLoss just because the DB StopLoss lags the exchange.
    Returns {} on any error (caller then falls back to the DB value)."""
    try:
        d = _http_get(f"/api/live_sl?key={API_KEY}&exchange={exchange}")
        return (d.get("live_sl") or {}).get(exchange, {})
    except Exception:
        return {}


def _live_sl_for(sym_bare, sym_full, live_sls):
    """Look up the live SL for a position in the live_sls map, tolerant of
    symbol-key variance (full 'UNIUSDT' vs bare 'UNI'). Returns None if absent."""
    if not live_sls:
        return None
    if sym_full:
        v = live_sls.get(sym_full.upper())
        if v and v.get("sl"):
            return v["sl"]
    for k, v in live_sls.items():
        if isinstance(v, dict) and v.get("sl") and k.upper() in (sym_bare.upper(), sym_bare.upper() + "USDT", sym_full):
            return v["sl"]
    return None


def send_move_sl(exchange, account_type, symbol_full, side, new_sl, username="Danyway"):
    msg = "\n".join([
        f"Username: {username}",
        f"AccountType: {account_type}",
        f"Exchange: {exchange}",
        f"Strategy: EdgeScanner",
        f"Action: MoveStopLoss",
        f"Symbol: {symbol_full}",
        f"Side: {side}",
        f"StopLoss: {new_sl}",
    ])
    return _http_post_webhook(msg)


# ── Stats: archive closed trailed trades to Hermes crypto.db (no PnL here) ──

def _fingerprint(exchange, account_type, symbol_full):
    """Stable identity for a live position. One-Way mode => 1 pos per symbol/account."""
    return f"{exchange}::{account_type}::{symbol_full}"


def _ensure_table():
    with sqlite3.connect(CRYPTO_DB) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS trailed_trades (
                TradeID        TEXT PRIMARY KEY,
                Exchange       TEXT NOT NULL,
                Symbol         TEXT NOT NULL,
                Side           TEXT NOT NULL,
                OriginalSL     REAL,
                FinalSL        REAL,
                Entry          REAL,
                TrailStartAt   TEXT,
                TrailEndAt     TEXT,
                created_at     TEXT DEFAULT (datetime('now'))
            )
        """)
        c.commit()


def _upsert_trailed_trade(row: dict):
    """Insert/replace a closed trailed trade's METADATA into crypto.db.
    PnL is NOT stored here — it is always read live by joining TradeID
    to the bot's authoritative trades.db ledger (option A)."""
    _ensure_table()
    with sqlite3.connect(CRYPTO_DB) as c:
        c.execute("""
            INSERT OR REPLACE INTO trailed_trades
                (TradeID, Exchange, Symbol, Side, OriginalSL, FinalSL, Entry, TrailStartAt, TrailEndAt)
            VALUES (:TradeID, :Exchange, :Symbol, :Side, :OriginalSL, :FinalSL, :Entry, :TrailStartAt, :TrailEndAt)
        """, row)
        c.commit()


def _archive_closed(state, exchange, account_type, open_keys):
    """Find tracked positions that are no longer open. Uses a 1-cycle grace:
    a position only gets archived after it's been absent for 2 consecutive runs,
    so transient API gaps never falsely archive (robust, not guessing).

    Returns list of archived (fingerprint, symbol)."""
    archived = []
    prefix = f"{exchange}::{account_type}::"
    for key in list(state["sl_moves"].keys()):
        if not key.startswith(prefix):
            continue
        if key in open_keys:
            state["sl_moves"][key].pop("_pending_close", None)
            continue
        # still tracked but not open now
        st = state["sl_moves"][key]
        if st.get("_pending_close"):
            # second consecutive absence -> really closed, archive it
            row = {
                "TradeID": st.get("trade_id") or f"{exchange}-{st.get('symbol_full','')}-{st.get('side','')}",
                "Exchange": exchange,
                "Symbol": st.get("symbol_full", key.split("::")[-1]),
                "Side": st.get("side", "LONG"),
                "OriginalSL": st.get("sl0"),
                "FinalSL": st.get("current_sl") or st.get("sl0"),
                "Entry": st.get("entry"),
                "TrailStartAt": st.get("start_at"),
                "TrailEndAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            try:
                _upsert_trailed_trade(row)
                archived.append((key, row["Symbol"]))
            except Exception as e:
                print(f"[{exchange}] archive error for {key}: {e}")
            del state["sl_moves"][key]
        else:
            st["_pending_close"] = time.time()
            state["sl_moves"][key] = st
    return archived


def _process_position(exchange, account_type, username, p, live_sls, state, open_keys, dry_run):
    """Decide and (unless dry-run) execute the SL move for ONE open position.

    Mutates state[\"sl_moves\"] for this symbol and adds its fingerprint to
    open_keys. Returns a (exchange, symbol_full, new_sl) move tuple if an SL was
    (will be) moved, else None.
    """
    sym = p["symbol"]
    symbol_full = p.get("symbol_full") or f"{sym}USDT"
    side = p.get("side", "LONG")
    sl0 = p.get("stop")
    entry = p.get("entry")
    if not sl0 or not entry:
        return None
    # Current price from bot API enrichment (mark price); fallback to direct
    # fetch if absent.
    price = p.get("price") or get_live_price(symbol_full)
    if not price:
        return None
    key = _fingerprint(exchange, account_type, symbol_full)
    open_keys.add(key)
    st = state["sl_moves"].get(key, {})
    # Seed metadata on first sight so archive has what it needs.
    # sl0/entry are the ORIGINAL R-anchor and are FROZEN at first sight — NEVER
    # overwrite them with the currently-moved SL. Overwriting was the bug: once
    # an SL was trailed above entry, entry-sl0 went negative and each subsequent
    # run computed R~0/dead, so the trail stopped (observed: ZEC SL 1146 fixed
    # at a stale value, trail dead).
    if not st:
        st = {
            "symbol_full": symbol_full,
            "side": side,
            "sl0": sl0,          # ORIGINAL stop = frozen R anchor
            "entry": entry,      # ORIGINAL entry = frozen R anchor
            "trade_id": p.get("trade_id") or p.get("TradeID"),  # may be None on absence
            "start_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    peak = st.get("peak", price)
    # current_sl = the CURRENT living SL on the exchange. Prefer the LIVE
    # working stop (from /api/live_sl — the authoritative on-exchange value) so
    # a stale DB StopLoss can never cause a redundant identical MoveStopLoss
    # POST. Fall back to the DB value (p[\"stop\"]), then to the optimistic
    # state value. The no-op guard below then compares the proposal against the
    # REAL exchange SL, only forwarding when it will actually move it.
    live_sl = _live_sl_for(sym, symbol_full, live_sls)
    if live_sl is not None:
        current_sl = live_sl
    elif p.get("stop") is not None:
        current_sl = p.get("stop")
    else:
        current_sl = st.get("current_sl", sl0)
    new_sl, peak_stale = compute_trail(side, st["entry"], st["sl0"], price, peak, current_sl)
    if peak_stale:
        # The retained peak would force an unplaceable SL (LONG SL at/above
        # market, or SHORT at/below) — a leftover from a move that never
        # actually applied. Reset the peak to the CURRENT price so trailing
        # resumes from reality; skip this cycle (nothing placeable yet).
        st["peak"] = price
        st["symbol_full"] = symbol_full
        st["side"] = side
        state["sl_moves"][key] = st
        print(f"[{exchange}] {sym}: reset stale peak {peak} -> {price} (unplaceable SL); resume trail next cycle")
        return None
    # Round to the symbol's price decimals so the reconcile compares like the
    # exchange stores it (2477.325 vs stored 2477.32 are EQUAL) — this stops the
    # sub-tick no-op re-send loop observed on ETH.
    dec = _price_decimals(symbol_full)
    new_sl = round(new_sl, dec) if new_sl else new_sl
    # ROOT-CAUSE no-op guarantee (Sep 2026): NEVER re-POST a MoveStopLoss equal
    # to what the engine already sent for this position. API-INDEPENDENT —
    # unlike the current_sl comparison above (which relies on /api/live_sl or
    # the DB and falls back to a STALE DB value when live_sl is down, causing an
    # identical-value re-POST loop, e.g. VVVUSDT 22.404 x14). `last_sent` is the
    # engine's OWN memory of the last value it POSTed, so an identical value is
    # skipped no matter how stale the live/DB sources are. Real forward moves
    # still POST normally. `new_sl == current_sl` means the exchange already holds
    # the value — no work; and `new_sl == last_sent` means we already sent it.
    last_sent = st.get("last_sent")
    needs_move = (new_sl and not _sl_equal(new_sl, current_sl, dec)
                  and (last_sent is None or not _sl_equal(new_sl, last_sent, dec)))
    if not needs_move:
        # update peak for next run; sl0/entry intentionally UNTOUCHED.
        st["peak"] = max(st.get("peak", price), price) if side == "LONG" else min(st.get("peak", price), price)
        st["symbol_full"] = symbol_full
        st["side"] = side
        state["sl_moves"][key] = st
        return None
    # PHANTOM GUARD (Sep 2026): once /api/live_sl is healthy for this exchange,
    # a symbol with NO live on-exchange working SL is a phantom — closed on the
    # exchange but the bot DB hasn't reconciled yet (position_sync runs every
    # 3h). Without this the engine re-POSTs the SAME MoveStopLoss every 3 min
    # forever (observed: SOLUSDT 102.83 x5, XRPUSDT 1.3851, BNBUSDT 727.68).
    # Skip entirely — safe (no position risk); the sync cron clears the stale DB
    # row. When live_sls is EMPTY (API read error) we DON'T gate, so a transient
    # failure never blocks trailing.
    if live_sls and _live_sl_for(sym, symbol_full, live_sls) is None:
        print(f"[{exchange}] {sym}: SKIP move to {new_sl} (no live on-exchange SL; DB-stale/closed on exchange)")
        # Do NOT record state or POST. Stays tracked so _archive_closed archives
        # it once the DB row reconciles away.
        return None
    if not dry_run:
        try:
            send_move_sl(exchange, account_type, symbol_full,
                         "BUY" if side == "LONG" else "SELL", round(new_sl, dec), username=username)
        except Exception as e:
            print(f"[{exchange}] {sym}: move SL error: {e}")
            # Don't record the failure; next cycle re-sends.
            return None
    # Only record specifics we KNOW the exchange holds (post webhook).
    st["current_sl"] = new_sl
    st["last_sent"] = new_sl  # no-op guard anchor — never re-POST same value
    st["peak"] = max(st.get("peak", price), price) if side == "LONG" else min(st.get("peak", price), price)
    state["sl_moves"][key] = st
    return (exchange, symbol_full, new_sl)


def run_once(dry_run=False):
    state = _load_state()
    moved = []
    _ensure_table()
    # Drop legacy-format keys ("Exchange:Symbol") that predate fingerprint
    # tracking — they lack verifiable entry/sl metadata, so archiving them
    # would be guessing. Real trailed trades are recorded from here on.
    for k in [k for k in state["sl_moves"] if "::" not in k]:
        del state["sl_moves"][k]
    for exchange, cfg in EXCHANGES.items():
        cfg_user_id = cfg.get("user_id", 1)
        cfg_username = cfg.get("username", "Danyway")
        account_type = cfg["account_type"]
        open_keys = set()
        # LIVE on-exchange SL map for this exchange — the authoritative no-op
        # guard. When the DB StopLoss lags the exchange (so it never equals the
        # proposed value), the engine would otherwise re-POST an identical
        # MoveStopLoss every cycle. Compare against the live SL instead.
        # Any error here -> empty map, so trailing is never blocked.
        try:
            live_sls = get_live_sls(exchange)
        except Exception:
            live_sls = {}
        try:
            pos = get_open_positions(exchange, account_type, cfg["source"], cfg_user_id=cfg_user_id)
        except Exception as e:
            print(f"[{exchange}] error fetching positions: {e}")
            continue
        for p in pos:
            move = _process_position(exchange, account_type, cfg_username,
                                     p, live_sls, state, open_keys, dry_run)
            if move:
                moved.append(move)
        # Detect + archive any tracked positions that closed since last run.
        for _key, sym in _archive_closed(state, exchange, account_type, open_keys):
            print(f"[{exchange}] archived closed position: {sym}")
    # NEVER persist state in dry-run: it would mark SLs as moved that were never
    # sent to the exchange, and the next real run would skip them (bug Sep 2026).
    if not dry_run:
        _save_state(state)
    return moved


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    result = run_once(dry_run=dry)
    for exchange, sym, sl in result:
        print(f"[{exchange}] {sym}: move SL to {sl}")
    print(f"{len(result)} SL moves" + (" (dry-run)" if dry else ""))
