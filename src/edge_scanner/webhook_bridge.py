"""
Edge Scanner → Trading-WebHook-Bot bridge.

Multi-config priority system with:
- SHORT signals supported (uses ABS score for ranking)
- Live R:R validation at send time (rejects degraded trades)
- Built-in retry on HTTP failures
- Fresh entry price from Binance at send time
- Min effective R:R floor to reject stale setups
"""
import httpx, json, logging, sqlite3, time
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple, Set

logger = logging.getLogger(__name__)

# ── Config ──
WEBHOOK_URL = "http://109.123.229.200/webhook"
WEBHOOK_KEY = "6XO7toihtxsSW7s9OgetPwVwjMCNhb4O"
EXCHANGE = "Binance"
STRATEGY = "EdgeScanner"
DB_PATH = "/home/hermes/BacktestingMCP/data/crypto.db"

# ── Execution mode (account type + market) ──────────────────────────────────
# The edge scanner can route signals to any of 5 execution modes. Each mode
# maps to the bot-valid AccountType (Standard | CopyTrading | Demo | TestNet)
# plus a MarketType (Futures | Spot) so the bot knows where to execute.
#
#   mode               AccountType    MarketType   Meaning
#   testnet            TestNet        Futures      TestNet futures (default)
#   copytrading_futures CopyTrading   Futures      Copy-trading futures account
#   copytrading_spot    CopyTrading   Spot         Copy-trading spot account
#   futures_prod        Standard      Futures      Production futures account
#   spot_prod           Standard      Spot         Production spot account
#
# Override with env var WEBHOOK_EXECUTION_MODE (e.g. in .env or edge_scan.sh).
EXECUTION_MODES = {
    "testnet":               {"account_type": "TestNet",     "market_type": "Futures"},
    "copytrading_futures":   {"account_type": "CopyTrading", "market_type": "Futures"},
    "copytrading_spot":      {"account_type": "CopyTrading", "market_type": "Spot"},
    "futures_prod":          {"account_type": "Standard",    "market_type": "Futures"},
    "spot_prod":             {"account_type": "Standard",    "market_type": "Spot"},
}
import os as _os
_EXECUTION_MODE = _os.getenv("WEBHOOK_EXECUTION_MODE", "testnet").strip().lower()
_EXEC_MODE_CFG = EXECUTION_MODES.get(_EXECUTION_MODE, EXECUTION_MODES["testnet"])
ACCOUNT_TYPE = _EXEC_MODE_CFG["account_type"]     # bot-valid AccountType
MARKET_TYPE = _EXEC_MODE_CFG["market_type"]        # Futures | Spot
# TestNet symbol validation only applies in testnet mode (the testnet
# exchangeInfo symbol list differs from prod).
TESTNET_MODE = (ACCOUNT_TYPE == "TestNet")

CONFIG_PRIORITY = [
    ("3.1", 7.0, "V3.1 ADX Trend"),         ("4.1", 7.0, "V4.1 Breakout+Vol"),
    ("5.1", 7.0, "V5.1 AI-focused"),         ("6.1", 7.0, "V6.1 Breakout Momentum"),
    ("2.1", 7.0, "V2.1 MT Alignment"),       ("4.0", 7.0, "V4.0 TR/ATR Breakout"),
    ("1.4", 7.0, "V1.4 Scanner-Focused"),    ("1.5", 7.0, "V1.5 Conservative R:R"),
    ("2.2", 7.0, "V2.2 Soft MT Alignment"),  ("8.0", 7.0, "V8.0 Funding Rate"),
    ("6.0", 7.0, "V6.0 Pullback"),           ("3.2", 7.0, "V3.2 Soft ADX"),
    ("5.2", 7.0, "V5.2 Balanced DeFi/AI"),   ("6.2", 7.0, "V6.2 Pullback Strat"),
    ("14.0", 7.0, "V14.0 Precursor BTC/ETH"),("14.1", 7.0, "V14.1 Precursor SHORT BTC/ETH"),
    ("5.0", 7.0, "V5.0 DEFI-focused"),
    ("10.0", 7.0, "V10.0 Chart Patterns"),   ("12.0", 7.0, "V12.0 Optimized Pro V2"),
    ("16.0", 7.0, "V16.0 Vol Squeeze"),      ("11.0", 7.0, "V11.0 Optimized Pro"),
    ("13.0", 7.0, "V13.0 Auto-Evolved"),
    ("22.0", 7.0, "V22.0 Liquidation LONG"), ("22.1", 7.0, "V22.1 Liquidation SHORT"),
    ("1.1", 7.0, "V1.1 Volume-Weighted"), ("1.2", 7.0, "V1.2 Signal-Focused"), ("1.3", 7.0, "V1.3 On-Chain"),
    ("7.2", 7.0, "V7.2 Filtered Quality"), ("7.5", 7.0, "V7.5 LLM Quality Gate"),
    ("7.6", 7.0, "V7.6 LLM Evolved"), ("7.8", 7.0, "V7.8 LLM Evolved v2"),
    ("3.3", 7.0, "V3.3 LLM ADX"), ("3.5", 7.0, "V3.5 LLM ADX v2"), ("6.4", 7.0, "V6.4 Flat Killer"),
    ("20.0", 5.0, "V20.0 Time-of-Day"), ("19.0", 5.0, "V19.0 Ratio Arb"),
    ("18.0", 7.0, "V18.0 Mean Reversion"), ("17.0", 7.0, "V17.0 Liquidation"),
    ("15.0", 7.0, "V15.0 Multi-TF"), ("9.0", 7.0, "V9.0 Vol Imbalance"),
]

MAX_SIGNALS_PER_BATCH = 8
MAX_SCORE_CAP = 15.0
EXCLUDED_SYMBOLS = {"BTWUSDT", "EULUSDT", "EIGENUSDT", "MORPHOUSDT", "DGBUSDT"}
MAX_SLIPPAGE_PCT = 0.5
MIN_EFFECTIVE_RR = 1.1   # Reject if effective R:R < 1.1 at send time (matches bot's min_risk_reward=1.1)
HTTP_RETRIES = 3
HTTP_RETRY_DELAY = 1.0

# ── Bybit / HyroTrader dual-route (Aug 2026) ─────────────────────────────
# Binance testnet keeps sending ALL configs (up to 8/batch). The Bybit route
# sends only the BEST configs (Option A — top 3 by WR) to the HyroTrader
# 10k challenge (Demo account). Bot-side enforces: 5 concurrent, 0.5%/trade,
# daily -3% halt, challenge -5% lost, low-cap filter, fail-closed.
# Toggle: set BYBIT_ROUTE=0 in the env to disable the Bybit pass.
BYBIT_ROUTE = _os.getenv("BYBIT_ROUTE", "1").strip().lower() in ("1", "true", "yes", "on")
BYBIT_CONFIGS = ["5.1", "1.4", "1.5"]   # top-3 WR configs (V5.1 AI, V1.4 scanner, V1.5 conservative)
BYBIT_MAX_SIGNALS = 3                    # conservative — matches 5-symbol cap with headroom
BYBIT_EXCHANGE = "Bybit"                 # routes to trade_bybit adapter
BYBIT_ACCOUNT_TYPE = "Demo"              # HyroTrader 10k challenge (mainnet demo)

# ── Live price cache ──
_live_price_cache: Dict[str, float] = {}
_live_price_cache_time: Optional[datetime] = None
PRICE_CACHE_TTL = 5  # seconds


def _get_live_price(symbol: str) -> Optional[float]:
    """Fetch live mark price from Binance, with short cache."""
    global _live_price_cache, _live_price_cache_time
    now = datetime.now(timezone.utc)
    if _live_price_cache and _live_price_cache_time and \
       (now - _live_price_cache_time).total_seconds() < PRICE_CACHE_TTL and \
       symbol in _live_price_cache:
        return _live_price_cache[symbol]
    try:
        resp = httpx.get(
            f"https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": f"{symbol}USDT"}, timeout=5,
        )
        if resp.status_code == 200:
            price = float(resp.json().get("markPrice", 0))
            if price > 0:
                _live_price_cache[symbol] = price
                _live_price_cache_time = now
                return price
    except Exception:
        pass
    return None


# ── DB ──
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_webhook_column():
    db = get_db()
    try:
        db.execute("ALTER TABLE edge_signals ADD COLUMN webhook_sent_at TIMESTAMP")
        db.commit()
        logger.info("Added webhook_sent_at column")
    except sqlite3.OperationalError:
        pass
    finally:
        db.close()


def get_pending_signals_for_config(version: str, min_score: float,
                                   cooldown_symbols: Optional[set] = None) -> List[Dict]:
    """Fetch pending signals for a config version — supports both LONG and SHORT.

    SHORT scores are negative (e.g., -11.3). We use ABS() so that
    |score| >= min_score works for both directions. Direction filtering
    is handled by the caller via the priority config's score sign.

    LIMIT is 50 (was 10) because open-position symbols are filtered in SQL
    (cooldown_symbols) — with LIMIT 10, LONGs always filled the top-10 and
    SHORTs (like MAGMA/GRASS -7.3) never got considered → direction
    starvation. Now open-position symbols are excluded BEFORE the limit.
    """
    db = get_db()
    params: list = [version, min_score]
    cooldown_clause = ""
    if cooldown_symbols:
        placeholders = ",".join("?" for _ in cooldown_symbols)
        cooldown_clause = f"AND symbol NOT IN ({placeholders})"
        params.extend(cooldown_symbols)
    rows = db.execute(f"""
        SELECT id, symbol, direction, entry_price, stop_price, target_price,
               composite_score, config_version, created_at
        FROM edge_signals
        WHERE config_version = ?
          AND outcome IS NULL
          AND ABS(composite_score) >= ?
          AND entry_price > 0
          AND stop_price > 0
          AND target_price > 0
          AND webhook_sent_at IS NULL
          AND created_at > datetime('now', '-2 hours')
          {cooldown_clause}
        ORDER BY ABS(composite_score) DESC
        LIMIT 50
    """, params).fetchall()
    db.close()
    return [dict(r) for r in rows]


def get_open_position_symbols() -> set:
    db = get_db()
    rows = db.execute("""
        SELECT DISTINCT symbol FROM edge_signals
        WHERE webhook_sent_at IS NOT NULL AND outcome IS NULL
    """).fetchall()
    db.close()
    open_syms = {r["symbol"] for r in rows}
    if open_syms:
        logger.info("Open positions: %s", ", ".join(sorted(open_syms)))
    return open_syms


def mark_signal_sent(signal_id: int):
    db = get_db()
    db.execute(
        "UPDATE edge_signals SET webhook_sent_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), signal_id)
    )
    db.commit()
    db.close()


# ── Validation ──
def _validate_signal(sig: Dict) -> Tuple[bool, str]:
    entry = sig.get("entry_price", 0)
    stop = sig.get("stop_price", 0)
    target = sig.get("target_price", 0)
    direction = sig.get("direction", "").upper()
    if entry <= 0: return False, "entry is 0 or negative"
    if stop <= 0:  return False, "stop is 0 or negative"
    if target <= 0: return False, "target is 0 or negative"
    if direction == "LONG":
        if stop >= entry:  return False, f"stop ({stop:.4f}) >= entry ({entry:.4f})"
        if target <= entry: return False, f"target ({target:.4f}) <= entry ({entry:.4f})"
    elif direction == "SHORT":
        if stop <= entry:  return False, f"stop ({stop:.4f}) <= entry ({entry:.4f})"
        if target >= entry: return False, f"target ({target:.4f}) >= entry ({entry:.4f})"
    else:
        return False, f"unknown direction: {direction}"
    return True, "ok"


def _check_effective_rr(sig: Dict) -> Tuple[bool, str, Optional[float]]:
    """Check if the signal still has acceptable R:R at the live market price.
    
    Key insight: a signal was good when detected, but by the time we're ready to
    send, the market may have moved so the original TP/SL no longer give a good
    risk/reward. We check the *effective* R:R if we entered at the live price
    with the original TP/SL levels.
    
    For LONG:  risk = entry - stop, reward = target - entry
               If live_price moved toward target, R:R improves → execute.
               If live_price moved toward stop, R:R degrades → reject if < floor.
    For SHORT: risk = stop - entry, reward = entry - target
               Same logic inverted.
    """
    live = _get_live_price(sig["symbol"])
    if live is None:
        return True, "no live price — send anyway", None
    
    entry = sig["entry_price"]
    stop = sig["stop_price"]
    target = sig["target_price"]
    direction = sig["direction"].upper()
    
    # Compute the R:R we'd actually get entering the *existing* TP/SL from live price
    if direction == "LONG":
        if live >= target:
            return False, f"LIVE ${live:.4f} ≥ TARGET ${target:.4f} — setup already completed, skip", None
        risk = live - stop
        reward = target - live
    else:  # SHORT
        if live <= target:
            return False, f"LIVE ${live:.4f} ≤ TARGET ${target:.4f} — setup already completed, skip", None
        risk = stop - live
        reward = live - target
    
    if risk <= 0:
        return False, f"price moved past stop — no risk left", None
    
    eff_rr = reward / risk
    if eff_rr < MIN_EFFECTIVE_RR:
        return False, f"effective R:R {eff_rr:.2f} < {MIN_EFFECTIVE_RR} — too degraded (live=${live:.4f}, entry=${entry:.4f})", eff_rr
    
    return True, f"ok (live=${live:.4f}, eff_RR={eff_rr:.2f})", eff_rr


# Cache the BTC regime trend once per bridge run — 8 configs × up to 8
# candidates each would otherwise create a fresh BacktestingEngine and load
# 10 days of H4 BTC data per candidate check (~50 redundant DB fetches).
_REGIME_CACHE: dict = {}
# Cache the TestNet symbol set once per bridge run.
_TESTNET_SYMBOLS: Optional[set] = None


def _check_market_regime(direction: str, market: str = "BTC") -> Tuple[bool, str]:
    from datetime import datetime, timezone, timedelta
    from src.core.backtesting_engine import BacktestingEngine
    from src.data.timeframe_converter import TimeFrame
    import pandas as pd
    pair = f"{market}USDT"
    try:
        trend = _REGIME_CACHE.get(pair)
        if trend is None:
            engine = BacktestingEngine()
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=10)
            df = engine.get_data(pair, TimeFrame.H4, start, end)
            if df.empty or len(df) < 20:
                _REGIME_CACHE[pair] = "UNKNOWN"
                return True, f"no data for {pair}"
            close = df['Close'].values
            last = float(close[-1])
            ema20 = float(pd.Series(close).ewm(span=20).mean().iloc[-1])
            trend = "UP" if last > ema20 else "DOWN"
            _REGIME_CACHE[pair] = trend
        if trend == "UNKNOWN":
            return True, f"no data for {pair}"
        if direction == "LONG" and trend == "DOWN":
            return False, f"{market} downtrend — LONG blocked"
        elif direction == "SHORT" and trend == "UP":
            return False, f"{market} uptrend — SHORT blocked"
        return True, ""
    except Exception as e:
        logger.warning("Regime check failed: %s", e)
        return True, f"check failed: {e}"


# ── Selection ──
def select_signals(bybit: bool = False, skip_symbols: Optional[Set[str]] = None) -> List[Dict]:
    from src.integrations.binance_symbols import is_on_binance_futures, is_futures_symbol_tradable
    testnet_check = TESTNET_MODE
    cooldown_symbols = get_open_position_symbols()
    selected = []
    selected_symbols = set()
    selected_configs = set()
    sent_count = 0

    # Bybit route: only the best configs, smaller batch, skip symbols already
    # sent to Binance this cycle (dual-route keeps accounts independent).
    priority = CONFIG_PRIORITY
    max_batch = MAX_SIGNALS_PER_BATCH
    if bybit:
        priority = [p for p in CONFIG_PRIORITY if p[0] in BYBIT_CONFIGS]
        max_batch = BYBIT_MAX_SIGNALS
        if skip_symbols:
            cooldown_symbols = set(cooldown_symbols) | set(skip_symbols)
        logger.info("  [BYBIT ROUTE] configs=%s max_batch=%d", [p[0] for p in priority], max_batch)

    for version, min_score, label in priority:
        if sent_count >= max_batch:
            break
        signals = get_pending_signals_for_config(version, min_score, cooldown_symbols)
        if not signals:
            continue
        for sig in signals:
            if sent_count >= max_batch:
                break
            sym = sig["symbol"]
            if sym in selected_symbols:
                continue
            valid, reason = _validate_signal(sig)
            if not valid:
                logger.info("  %s: REJECTED %s %s — %s", label, sig["direction"], sym, reason)
                continue
            
            # LIVE R:R CHECK — reject if the setup has degraded
            rr_ok, rr_reason, eff_rr = _check_effective_rr(sig)
            if not rr_ok:
                logger.info("  %s: DEGRADED %s %s — %s", label, sig["direction"], sym, rr_reason)
                continue
            logger.info("  %s: %s %s %s", label, sig["direction"], sym, rr_reason)

            from src.edge_scanner.scoring_config import ALL_CONFIGS
            _cfg = ALL_CONFIGS.get(version)
            if _cfg and _cfg.status == "disabled":
                continue
            # Hard direction restriction (belt & suspenders on top of the
            # scoring-stage check): a config like V14.1 (SHORT-only) must
            # never send a LONG even if a pre-existing/legacy signal sneaks
            # through the SQL window.
            if _cfg and _cfg.allowed_directions and \
               sig["direction"].upper() not in {d.upper() for d in _cfg.allowed_directions}:
                logger.info("  %s: DIRECTION BLOCKED %s %s (config allows %s)",
                            label, sig["direction"], sym, _cfg.allowed_directions)
                continue
            if _cfg and _cfg.symbol_whitelist and sym not in _cfg.symbol_whitelist:
                continue
            if _cfg and _cfg.market_regime_filter != "OFF":
                regime_ok, regime_reason = _check_market_regime(sig["direction"], _cfg.market_regime_filter)
                if not regime_ok:
                    logger.info("  %s: REGIME BLOCKED %s %s — %s", label, sig["direction"], sym, regime_reason)
                    continue
            if _cfg and (_cfg.allowed_start_hour > 0 or _cfg.allowed_end_hour < 24):
                current_hour = datetime.now(timezone.utc).hour
                if not (_cfg.allowed_start_hour <= current_hour < _cfg.allowed_end_hour):
                    logger.info("  %s: TIME BLOCKED %s %s", label, sig["direction"], sym)
                    continue
            if sym in EXCLUDED_SYMBOLS:
                continue
            # Normalize score to positive + cap. The score convention is now
            # ALWAYS positive (direction is carried separately in the DB and
            # webhook Direction field). abs() also protects against any legacy
            # negative rows still in the 2h window.
            sig["composite_score"] = min(abs(sig["composite_score"]), MAX_SCORE_CAP)
            if not is_on_binance_futures(sym):
                logger.info("  %s: SKIPPED %s %s — not on Binance Futures", label, sig["direction"], sym)
                continue
            if not is_futures_symbol_tradable(sym):
                logger.info("  %s: SKIPPED %s %s — not TRADING", label, sig["direction"], sym)
                continue
            # Bybit/HyroTrader LOW-CAP HARD RULE (Aug 2026): no symbol with
            # mcap < $300M or 24h vol < $1M/day. Fail-closed — unknown symbols
            # are DENIED. Applies on the Bybit route (and if EXCHANGE is
            # Bybit/HyroTrader); the Binance path is unchanged.
            if bybit or EXCHANGE.lower() in ("bybit", "hyrotrader"):
                try:
                    from src.edge_scanner.liquidity_filter import is_liquid_for_bybit
                    liq_ok, liq_reason = is_liquid_for_bybit(sym + "USDT")
                    if not liq_ok:
                        logger.info("  %s: SKIPPED %s %s — %s", label, sig["direction"], sym, liq_reason)
                        continue
                except Exception as e:
                    logger.warning("  liquidity_filter error for %s: %s — DENYING (fail-closed)", sym, e)
                    continue
            if testnet_check:
                # Cache the TestNet symbol set once per bridge run (was fetched
                # per candidate — 8+ HTTP calls per cycle).
                global _TESTNET_SYMBOLS
                testnet_syms = _TESTNET_SYMBOLS
                if testnet_syms is None:
                    testnet_syms = set()
                    try:
                        resp = httpx.get("https://testnet.binancefuture.com/fapi/v1/exchangeInfo", timeout=5)
                        if resp.status_code == 200:
                            testnet_syms = {s["symbol"][:-4] for s in resp.json()["symbols"]}
                    except Exception:
                        pass
                    _TESTNET_SYMBOLS = testnet_syms
                if sym not in testnet_syms:
                    logger.info("  %s: SKIPPED %s %s — not on TestNet", label, sig["direction"], sym)
                    continue
            if version in selected_configs:
                continue
            sig["_priority_label"] = label
            sig["_effective_rr"] = eff_rr
            selected.append(sig)
            selected_symbols.add(sym)
            selected_configs.add(version)
            sent_count += 1
            logger.info("  %s: SELECTED %s %s (score=%.1f) [%d/%d]", label, sig["direction"], sym, sig["composite_score"], sent_count, MAX_SIGNALS_PER_BATCH)

    if not selected:
        logger.info("No signals selected")
    return selected


# ── Webhook sender with retry ──
def _compute_tier_multiplier(signal: Dict) -> float:
    try:
        import sqlite3, os
        if not os.path.exists(DB_PATH):
            return 1.0
        conn = sqlite3.connect(DB_PATH)
        cv = signal.get("config_version", "")
        sc = signal.get("composite_score", 0)
        tier1 = 1.0
        wr_row = conn.execute("""
            SELECT (SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END)*1.0/
                   NULLIF(SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END)+SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END), 0)) as wr,
                   SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END)+SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) as n
            FROM edge_signals WHERE config_version=? AND webhook_sent_at IS NOT NULL AND outcome IN ('WIN','LOSS')
        """, (cv,)).fetchone()
        if wr_row and wr_row[1] and wr_row[1] >= 10:
            wr = wr_row[0]
            if wr > 0.60: tier1 = 1.2
            elif wr > 0.40: tier1 = 1.0
            elif wr > 0.20: tier1 = 0.7
            else: tier1 = 0.5
        else: tier1 = 0.5
        tier2 = 1.3 if sc >= 10.0 else 1.0 if sc >= 8.0 else 0.7 if sc >= 5.0 else 0.5
        tier3 = 1.0
        rr_row = conn.execute("SELECT DISTINCT config_json FROM scoring_configs WHERE version=?", (cv,)).fetchone()
        if rr_row:
            import json
            rr = float(json.loads(rr_row[0]).get("rr_ratio", 1.5))
            tier3 = 1.0 if rr >= 2.0 else 0.8 if rr >= 1.5 else 0.6
        conn.close()
        return round(tier1 * tier2 * tier3, 2)
    except Exception:
        return 1.0


def format_webhook_msg(signal: Dict, bybit: bool = False) -> str:
    action = "OpenLong" if signal["direction"].upper() == "LONG" else "OpenShort"
    side = "BUY" if signal["direction"].upper() == "LONG" else "SELL"
    # Score is ALWAYS positive; direction is explicit via Action/Side/Direction.
    # Negative scores (old convention) are normalized so the bot never sees
    # a negative Score (bot clamps out-of-range scores → silently wrong sizing).
    score = abs(signal["composite_score"])
    # Bybit/HyroTrader route: stamp Exchange=Bybit + AccountType=Demo so the
    # bot's router executes on the challenge account (activates 5-concurrent,
    # 0.5% risk, daily -3% / challenge -5% breakers, low-cap filter).
    exchange = BYBIT_EXCHANGE if bybit else EXCHANGE
    account_type = BYBIT_ACCOUNT_TYPE if bybit else ACCOUNT_TYPE
    lines = [
        f"Username: Danyway", f"AccountType: {account_type}", f"MarketType: {MARKET_TYPE}",
        f"Exchange: {exchange}", f"Strategy: {STRATEGY}", f"Action: {action}", f"Side: {side}",
        f"Direction: {signal['direction'].upper()}",
        f"Symbol: {signal['symbol']}USDT", f"Entry: {signal['entry_price']}",
        f"StopLoss: {signal['stop_price']}", f"TakeProfit: {signal['target_price']}",
        f"Score: {score}", f"ConfigVersion: {signal['config_version']}",
        f"TierMultiplier: {_compute_tier_multiplier(signal)}",
    ]
    return "\n".join(lines)


def send_signal_to_webhook(signal: Dict, bybit: bool = False) -> bool:
    msg_str = format_webhook_msg(signal, bybit=bybit)
    payload = {"key": WEBHOOK_KEY, "telegram_alert_type": "trading_bot", "msg": msg_str}
    
    last_err = ""
    for attempt in range(HTTP_RETRIES):
        try:
            resp = httpx.post(WEBHOOK_URL, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("  ✅ Sent %s %s @ %.2f (score=%.1f, %s, %s, attempt=%d)",
                            signal["direction"], signal["symbol"], signal["entry_price"],
                            signal["composite_score"], signal.get("_priority_label", ""),
                            "Bybit" if bybit else "Binance", attempt+1)
                mark_signal_sent(signal["id"])
                return True
            elif resp.status_code == 503:
                last_err = f"HTTP 503 (retry {attempt+1})"
                time.sleep(HTTP_RETRY_DELAY * (attempt + 1))
                continue
            else:
                logger.warning("  ❌ Webhook %d for %s: %s", resp.status_code, signal["symbol"], resp.text[:80])
                return False
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            logger.warning("  ⚠️ Attempt %d/%d failed for %s: %s", attempt+1, HTTP_RETRIES, signal["symbol"], last_err)
            if attempt < HTTP_RETRIES - 1:
                time.sleep(HTTP_RETRY_DELAY * (attempt + 1))
    
    logger.error("  ❌ All %d attempts failed for %s: %s", HTTP_RETRIES, signal["symbol"], last_err)
    return False


# ── Main ──
def run_bridge(dry_run: bool = False) -> int:
    ensure_webhook_column()
    logger.info("=== Webhook Bridge ===")
    logger.info("Configs: %d | Max batch: %d | Min eff R:R: %.1f | Retries: %d | Mode: %s (%s/%s)",
                len(CONFIG_PRIORITY), MAX_SIGNALS_PER_BATCH, MIN_EFFECTIVE_RR, HTTP_RETRIES,
                _EXECUTION_MODE, ACCOUNT_TYPE, MARKET_TYPE)

    # ── Pass 1: Binance (all configs, existing behavior) ──
    selected = select_signals(bybit=False)
    if not selected:
        logger.info("Nothing to send (Binance)")
    elif dry_run:
        logger.info("=== DRY-RUN — would send %d Binance signals ===", len(selected))
        for sig in selected:
            logger.info("  %s %s @ %.2f (score=%.1f, R:R=%.2f, %s)",
                        sig["direction"], sig["symbol"], sig["entry_price"],
                        sig["composite_score"], sig.get("_effective_rr", 0),
                        sig.get("_priority_label", ""))
    else:
        sent_count = 0
        for sig in selected:
            if send_signal_to_webhook(sig, bybit=False):
                sent_count += 1
        logger.info("Sent %d/%d Binance signals", sent_count, len(selected))

    # ── Pass 2: Bybit / HyroTrader (best configs, skips Binance symbols) ──
    # Runs ALWAYS (parallel to Binance) unless BYBIT_ROUTE=0. The bot
    # enforces the prop-firm risk profile on the Demo challenge account.
    binance_symbols = {sig["symbol"] for sig in selected}
    bybit_selected = []
    if BYBIT_ROUTE:
        bybit_selected = select_signals(bybit=True, skip_symbols=binance_symbols)
    else:
        logger.info("Bybit route DISABLED (BYBIT_ROUTE=0)")
    if not bybit_selected:
        logger.info("Nothing to send (Bybit)")
    elif dry_run:
        logger.info("=== DRY-RUN — would send %d Bybit signals ===", len(bybit_selected))
        for sig in bybit_selected:
            logger.info("  %s %s @ %.2f (score=%.1f, R:R=%.2f, %s) -> Bybit/Demo",
                        sig["direction"], sig["symbol"], sig["entry_price"],
                        sig["composite_score"], sig.get("_effective_rr", 0),
                        sig.get("_priority_label", ""))
    else:
        sent_count = 0
        for sig in bybit_selected:
            if send_signal_to_webhook(sig, bybit=True):
                sent_count += 1
        logger.info("Sent %d/%d Bybit signals", sent_count, len(bybit_selected))

    if dry_run:
        return len(selected) + len(bybit_selected)
    return len(selected) + len(bybit_selected)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    import sys
    dry = "--dry-run" in sys.argv
    count = run_bridge(dry_run=dry)
    print(f"Sent {count} signals{' (dry-run)' if dry else ''}")