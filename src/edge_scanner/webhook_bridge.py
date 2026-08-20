"""
Edge Scanner → Trading-WebHook-Bot bridge.

Multi-config priority system with symbol dedup:

Priority order:
  1. V7.0  (active, 50.0% WR) — quality gate
  2. V6.2  (63.6% WR) — pullback strategy
  3. V4.1  (57.8% WR) — breakout strategy

Rules:
  - One signal per symbol per batch (config priority decides which wins)
  - Max 3 signals per batch (respects bot's 3-5 position limit)
  - 24h cooldown per symbol (no repeat signals for the same symbol)
  - Higher score threshold for V7.0 (8.0) vs others (7.5)
"""

import httpx
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
WEBHOOK_URL = "http://109.123.229.200/webhook"
WEBHOOK_KEY = "6XO7toihtxsSW7s9OgetPwVwjMCNhb4O"
ACCOUNT_TYPE = "TestNet"  # Change to "Standard" for live trading
EXCHANGE = "Binance"
STRATEGY = "EdgeScanner"
DB_PATH = "/home/hermes/BacktestingMCP/data/crypto.db"

# Config priority: (version, min_score, label)
# Reordered by backtested EV (resolved signals in DB, >=20 trades)
#   Top tiers: highest EV configs get priority
#   Bottom tier: rotation round-robin for data collection
CONFIG_PRIORITY = [
    # ── TIER 1: Best EV (backtested) — always checked first ──
    ("3.1", 7.0, "V3.1 ADX Trend"),         # EV=+7.05% 🏆
    ("4.1", 7.0, "V4.1 Breakout+Vol"),       # EV=+5.55%
    ("5.1", 7.0, "V5.1 AI-focused"),         # EV=+5.17%
    ("6.1", 7.0, "V6.1 Breakout Momentum"),  # EV=+4.87%
    ("2.1", 7.0, "V2.1 MT Alignment"),       # EV=+3.65%
    ("4.0", 7.0, "V4.0 TR/ATR Breakout"),    # EV=+2.92%
    # ── TIER 2: Solid EV configs (max 2 V1.x) ──
    ("1.4", 7.0, "V1.4 Scanner-Focused"),    # EV=+2.23% (ACTIVE, proven)
    ("1.5", 7.0, "V1.5 Conservative R:R"),   # EV=+0.86%, best execution EV
    ("2.2", 7.0, "V2.2 Soft MT Alignment"),  # EV=+1.92%
    ("8.0", 7.0, "V8.0 Funding Rate"),       # EV=+1.76%
    ("6.0", 7.0, "V6.0 Pullback"),           # EV=+1.70%
    ("3.2", 7.0, "V3.2 Soft ADX"),           # EV=+1.46%
    ("5.2", 7.0, "V5.2 Balanced DeFi/AI"),   # EV=+1.09%
    ("6.2", 7.0, "V6.2 Pullback Strat"),     # EV=+0.72%
    ("14.0", 7.0, "V14.0 Precursor BTC/ETH"),    # EV=+0.45% (BTC/ETH only)
       ("14.1", 7.0, "V14.1 Precursor SHORT BTC/ETH"), # SHORT-focus variant
    ("5.0", 7.0, "V5.0 DEFI-focused"),       # EV=+0.05%
    # ── TIER 3: Negative EV but keep for rotation / special ──
    # V10.0 (EV=-0.57%), V16.0 (EV=-0.78%), V12.0 (EV=-0.34%)
    ("10.0", 7.0, "V10.0 Chart Patterns"),   # High WR (90.9% executed)
    ("12.0", 7.0, "V12.0 Optimized Pro V2"), # Parameter-optimized
    ("16.0", 7.0, "V16.0 Vol Squeeze"),      # Most execution data
    ("11.0", 7.0, "V11.0 Optimized Pro"),    # 57.5% WR in backtest
    ("13.0", 7.0, "V13.0 Auto-Evolved"),     # LLM-suggested
    # ── TIER 4: Liquidation-driven (new, live data only) ──
    ("22.0", 7.0, "V22.0 Liquidation LONG"),   # Short squeeze detection
    ("22.1", 7.0, "V22.1 Liquidation SHORT"),  # Long squeeze detection
    # ── TIER 5: Rotating configs (for data collection) ──
    # Picked via round-robin rotation based on day of month
    ("1.1", 7.0, "V1.1 Volume-Weighted"),
    ("1.2", 7.0, "V1.2 Signal-Focused"),
    ("1.3", 7.0, "V1.3 On-Chain"),
    ("7.2", 7.0, "V7.2 Filtered Quality"),
    ("7.5", 7.0, "V7.5 LLM Quality Gate"),
    ("7.6", 7.0, "V7.6 LLM Evolved"),
    ("7.8", 7.0, "V7.8 LLM Evolved v2"),
    ("3.3", 7.0, "V3.3 LLM ADX"),
    ("3.5", 7.0, "V3.5 LLM ADX v2"),
    ("6.4", 7.0, "V6.4 Flat Killer"),
    # Special purpose (not in backtest rotation)
    ("20.0", 5.0, "V20.0 Time-of-Day"),
    ("19.0", 5.0, "V19.0 Ratio Arb"),
    ("18.0", 7.0, "V18.0 Mean Reversion"),
    ("17.0", 7.0, "V17.0 Liquidation"),
    ("15.0", 7.0, "V15.0 Multi-TF"),
    ("9.0", 7.0, "V9.0 Vol Imbalance"),
]

MAX_SIGNALS_PER_BATCH = 8       # Matches the 8 concurrent trade slots
MAX_SCORE_CAP = 15.0            # 12+ scores have 57.1% WR, EV=+4.12%
EXCLUDED_SYMBOLS = {"BTWUSDT", "EULUSDT", "EIGENUSDT", "MORPHOUSDT", "DGBUSDT"}  # 0% WR symbols — never profitable
MAX_SLIPPAGE_PCT = 0.5           # Max price difference from entry before skipping signal


# ── DB helpers ──────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_webhook_column():
    """Add webhook_sent_at column if it doesn't exist."""
    db = get_db()
    try:
        db.execute("ALTER TABLE edge_signals ADD COLUMN webhook_sent_at TIMESTAMP")
        db.commit()
        logger.info("Added webhook_sent_at column to edge_signals")
    except sqlite3.OperationalError:
        pass
    finally:
        db.close()


def get_pending_signals_for_config(version: str, min_score: float) -> List[Dict]:
    """Fetch pending signals for a specific config version."""
    db = get_db()
    rows = db.execute("""
        SELECT id, symbol, direction, entry_price, stop_price, target_price,
               composite_score, config_version, created_at
        FROM edge_signals
        WHERE config_version = ?
          AND outcome IS NULL
          AND composite_score >= ?
          AND entry_price > 0
          AND stop_price > 0
          AND target_price > 0
          AND webhook_sent_at IS NULL
          AND created_at > datetime('now', '-2 hours')
        ORDER BY composite_score DESC
        LIMIT 10
    """, (version, min_score)).fetchall()
    db.close()
    return [dict(r) for r in rows]


def get_open_position_symbols() -> set:
    """Get symbols with ACTIVE open positions.

    A position is 'open' if a signal was sent to the webhook (webhook_sent_at)
    but the signal hasn't resolved yet (outcome IS NULL).

    This mirrors the bot's actual open positions. Once a signal resolves
    (WIN/LOSS/FLAT), that symbol's slot is free for any config to use again.
    """
    db = get_db()
    rows = db.execute("""
        SELECT DISTINCT symbol FROM edge_signals
        WHERE webhook_sent_at IS NOT NULL
          AND outcome IS NULL
    """).fetchall()
    db.close()
    open_symbols = {r["symbol"] for r in rows}
    if open_symbols:
        logger.info("Open positions: %s", ", ".join(sorted(open_symbols)))
    return open_symbols


def mark_signal_sent(signal_id: int):
    """Mark a signal as sent to webhook."""
    db = get_db()
    db.execute(
        "UPDATE edge_signals SET webhook_sent_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), signal_id)
    )
    db.commit()
    db.close()


# ── Priority selection ──────────────────────────────────────────────────────

def _validate_signal(sig: Dict) -> Tuple[bool, str]:
    """Validate a signal has valid entry, stop, and target prices.

    Hard rules (never send if violated):
    - Entry price > 0
    - Stop price > 0
    - Target price > 0
    - For LONG: stop < entry < target
    - For SHORT: stop > entry > target
    """
    entry = sig.get("entry_price", 0)
    stop = sig.get("stop_price", 0)
    target = sig.get("target_price", 0)
    direction = sig.get("direction", "").upper()

    if entry <= 0:
        return False, "entry price is 0 or negative"
    if stop <= 0:
        return False, "stop loss is 0 or negative"
    if target <= 0:
        return False, "take profit is 0 or negative"

    if direction == "LONG":
        if stop >= entry:
            return False, f"stop ({stop:.8f}) >= entry ({entry:.8f}) — stop must be below entry"
        if target <= entry:
            return False, f"target ({target:.8f}) <= entry ({entry:.8f}) — target must be above entry"
    elif direction == "SHORT":
        if stop <= entry:
            return False, f"stop ({stop:.8f}) <= entry ({entry:.8f}) — stop must be above entry"
        if target >= entry:
            return False, f"target ({target:.8f}) >= entry ({entry:.8f}) — target must be below entry"
    else:
        return False, f"unknown direction: {direction}"

    return True, "ok"


def _check_slippage(sig: Dict) -> Tuple[bool, str]:
    """Check that the current market price hasn't moved too far from the signal's entry."""
    entry = sig.get("entry_price", 0)
    symbol = sig.get("symbol", "")
    if entry <= 0 or not symbol:
        return True, "no entry or symbol — skip"
    try:
        import httpx
        resp = httpx.get(
            f"https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": f"{symbol}USDT"}, timeout=5,
        )
        if resp.status_code != 200:
            return True, "price fetch failed"
        mark_price = float(resp.json().get("markPrice", 0))
        if mark_price <= 0:
            return True, "invalid mark price"
        slippage_pct = abs(mark_price - entry) / entry * 100
        if slippage_pct > MAX_SLIPPAGE_PCT:
            return False, (
                f"market ${mark_price:.4f} is {slippage_pct:.2f}% away from entry "
                f"${entry:.4f} (max {MAX_SLIPPAGE_PCT}%)"
            )
        return True, f"ok (${mark_price:.4f}, {slippage_pct:.2f}%)"
    except Exception as e:
        return True, f"price check failed ({e})"


def _check_market_regime(direction: str, market: str = "BTC") -> Tuple[bool, str]:
    """Check if the market trend supports the trade direction.
    Uses EMA20 on 4h data — smoother than 1h, catches macro trend.
    Returns (True, '') if OK, (False, reason) if blocked.
    """
    from datetime import datetime, timezone, timedelta
    from src.core.backtesting_engine import BacktestingEngine
    from src.data.timeframe_converter import TimeFrame
    import pandas as pd

    pair = f"{market}USDT"
    try:
        engine = BacktestingEngine()
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=10)
        df = engine.get_data(pair, TimeFrame.H4, start, end)
        if df.empty or len(df) < 20:
            return True, f"no data for {pair}"

        close = df['Close'].values
        last = float(close[-1])
        ema20 = float(pd.Series(close).ewm(span=20).mean().iloc[-1])
        trend = "UP" if last > ema20 else "DOWN"

        if direction == "LONG" and trend == "DOWN":
            return False, f"{market} in downtrend (price {last:.0f} < EMA20 {ema20:.0f}) — LONG blocked"
        elif direction == "SHORT" and trend == "UP":
            return False, f"{market} in uptrend (price {last:.0f} > EMA20 {ema20:.0f}) — SHORT blocked"
        return True, ""
    except Exception as e:
        logger.warning("Market regime check failed: %s", e)
        return True, f"check failed: {e}"


def select_signals() -> List[Dict]:
    """Select the best signals across all configs using priority + dedup.

    Returns at most MAX_SIGNALS_PER_BATCH signals, with no duplicate symbols.
    """
    from src.integrations.binance_symbols import is_on_binance_futures, is_futures_symbol_tradable

    # When using TestNet, validate against TestNet symbols instead of production
    testnet_check = ACCOUNT_TYPE == "TestNet"
    cooldown_symbols = get_open_position_symbols()
    selected = []       # Final selected signals
    selected_symbols = set()  # Symbols already picked in this batch
    selected_configs = set()  # Configs already picked in this batch (diversity)
    sent_count = 0

    # ── ROTATION: rotate Tier-4 configs (lower EV) round-robin by day ──
    # This gives under-tested configs a chance to accumulate execution data.
    from datetime import datetime, timezone
    _rot_offset = datetime.now(timezone.utc).day % 7  # 0-6, changes daily
    _tier4_idx = 0

    for version, min_score, label in CONFIG_PRIORITY:
        if sent_count >= MAX_SIGNALS_PER_BATCH:
            break

        signals = get_pending_signals_for_config(version, min_score)
        if not signals:
            logger.info("  %s: no qualifying signals", label)
            continue

        for sig in signals:
            if sent_count >= MAX_SIGNALS_PER_BATCH:
                break

            sym = sig["symbol"]
            # Skip if symbol already picked by a higher-priority config
            if sym in selected_symbols:
                logger.info(
                    "  %s: skipping %s (already picked by higher priority config)",
                    label, sym,
                )
                continue
            # Skip if symbol is in cooldown
            if sym in cooldown_symbols:
                logger.info(
                    "  %s: skipping %s (open position — waiting for resolution)",
                    label, sym,
                )
                continue

            # HARD VALIDATION: entry, stop, target must be valid
            valid, reason = _validate_signal(sig)
            if not valid:
                logger.info(
                    "  %s: REJECTED %s %s — %s",
                    label, sig["direction"], sym, reason,
                )
                continue

            # SLIPPAGE CHECK: market price must be near entry price
            slip_ok, slip_reason = _check_slippage(sig)
            if not slip_ok:
                logger.info(
                    "  %s: SKIPPED %s %s — %s",
                    label, sig["direction"], sym, slip_reason,
                )
                continue

            # MARKET REGIME CHECK: don't trade against BTC trend
            from src.edge_scanner.scoring_config import ALL_CONFIGS
            _cfg = ALL_CONFIGS.get(version)
            if _cfg and _cfg.status == "disabled":
                logger.info(
                    "  %s: SKIPPED %s %s — config %s is disabled",
                    label, sig["direction"], sym, version,
                )
                continue
            if _cfg and _cfg.symbol_whitelist and sym not in _cfg.symbol_whitelist:
                logger.info(
                    "  %s: SKIPPED %s %s — not in config %s whitelist (%s)",
                    label, sig["direction"], sym, version,
                    ",".join(_cfg.symbol_whitelist),
                )
                continue
            if _cfg and _cfg.market_regime_filter != "OFF":
                regime_ok, regime_reason = _check_market_regime(sig["direction"], _cfg.market_regime_filter)
                if not regime_ok:
                    logger.info(
                        "  %s: REGIME BLOCKED %s %s — %s",
                        label, sig["direction"], sym, regime_reason,
                    )
                    continue

            # TIME-OF-DAY FILTER: only trade during specified hours
            if _cfg and (_cfg.allowed_start_hour > 0 or _cfg.allowed_end_hour < 24):
                from datetime import datetime, timezone
                current_hour = datetime.now(timezone.utc).hour
                if not (_cfg.allowed_start_hour <= current_hour < _cfg.allowed_end_hour):
                    logger.info(
                        "  %s: TIME BLOCKED %s %s — hour %d not in [%d, %d)",
                        label, sig["direction"], sym, current_hour,
                        _cfg.allowed_start_hour, _cfg.allowed_end_hour,
                    )
                    continue

            # Exclude symbols that have never been profitable
            if sym in EXCLUDED_SYMBOLS:
                logger.info("  %s: SKIPPED %s %s — excluded symbol (0%% WR)", label, sym, sig["direction"])
                continue

            # Cap score — extreme scores (12+) mean-revert
            capped_score = min(sig["composite_score"], MAX_SCORE_CAP)
            sig["composite_score"] = capped_score

            # BINANCE FUTURES CHECK: symbol must exist and be actively TRADING
            if not is_on_binance_futures(sym):
                logger.info(
                    "  %s: SKIPPED %s %s — not on Binance Futures (Spot only)",
                    label, sig["direction"], sym,
                )
                continue
            if not is_futures_symbol_tradable(sym):
                logger.info(
                    "  %s: SKIPPED %s %s — not actively TRADING on Futures (PENDING_TRADING)",
                    label, sig["direction"], sym,
                )
                continue
            if testnet_check:
                try:
                    resp = httpx.get(
                        "https://testnet.binancefuture.com/fapi/v1/exchangeInfo",
                        timeout=5,
                    )
                    if resp.status_code == 200:
                        testnet_syms = {s["symbol"][:-4] for s in resp.json()["symbols"]}
                        if sym not in testnet_syms:
                            logger.info(
                                "  %s: SKIPPED %s %s — not on TestNet",
                                label, sig["direction"], sym,
                            )
                            continue
                except Exception:
                    pass  # skip TestNet check if it fails

            # CONFIG DIVERSITY: max 1 signal per config per batch
            if version in selected_configs:
                logger.info(
                    "  %s: SKIPPED %s %s — config already contributed this batch",
                    label, sig["direction"], sym,
                )
                continue

            sig["_priority_label"] = label
            selected.append(sig)
            selected_symbols.add(sym)
            selected_configs.add(version)
            sent_count += 1
            logger.info(
                "  %s: selected %s %s (score=%.1f) [%d/%d]",
                label, sig["direction"], sym, sig["composite_score"],
                sent_count, MAX_SIGNALS_PER_BATCH,
            )

    if not selected:
        logger.info("No signals selected after priority + dedup")
    return selected


# ── Webhook sender ──────────────────────────────────────────────────────────

def _compute_tier_multiplier(signal: Dict) -> float:
    """Compute tier sizing multiplier for a signal using the edge scanner DB.
    Matches the logic in manual_trading.py but runs on this machine (has DB)."""
    try:
        import sqlite3, os
        db_path = "/home/hermes/BacktestingMCP/data/crypto.db"
        if not os.path.exists(db_path):
            return 1.0
        conn = sqlite3.connect(db_path)
        
        config_version = signal.get("config_version", "")
        composite_score = signal.get("composite_score", 0)
        
        # Tier 1: Config WR
        tier1 = 1.0
        wr_row = conn.execute("""
            SELECT (SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END)*1.0/
                   NULLIF(SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END)+
                          SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END), 0)) as wr,
                   SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END)+
                   SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) as n
            FROM edge_signals WHERE config_version=? AND webhook_sent_at IS NOT NULL
              AND outcome IN ('WIN','LOSS')
        """, (config_version,)).fetchone()
        if wr_row and wr_row[1] and wr_row[1] >= 10:
            wr = wr_row[0]
            if wr > 0.60:
                tier1 = 1.2
            elif wr > 0.40:
                tier1 = 1.0
            elif wr > 0.20:
                tier1 = 0.7
            else:
                tier1 = 0.5
        else:
            tier1 = 0.5
        
        # Tier 2: Signal score
        score = float(composite_score)
        if score >= 10.0:
            tier2 = 1.3
        elif score >= 8.0:
            tier2 = 1.0
        elif score >= 5.0:
            tier2 = 0.7
        else:
            tier2 = 0.5
        
        # Tier 3: R:R from config
        tier3 = 1.0
        rr_row = conn.execute("""
            SELECT DISTINCT config_json FROM scoring_configs WHERE version=?
        """, (config_version,)).fetchone()
        if rr_row:
            import json
            cfg_json = json.loads(rr_row[0])
            rr = float(cfg_json.get("rr_ratio", 1.5))
            if rr >= 2.0:
                tier3 = 1.0
            elif rr >= 1.5:
                tier3 = 0.8
            else:
                tier3 = 0.6
        
        conn.close()
        return round(tier1 * tier2 * tier3, 2)
    except Exception:
        return 1.0


def format_webhook_msg(signal: Dict) -> str:
    """Format an edge signal into the webhook's newline-separated message format."""
    action = "OpenLong" if signal["direction"].upper() == "LONG" else "OpenShort"
    side = "BUY" if signal["direction"].upper() == "LONG" else "SELL"

    lines = [
        f"Username: Danyway",
        f"AccountType: {ACCOUNT_TYPE}",
        f"Exchange: {EXCHANGE}",
        f"Strategy: {STRATEGY}",
        f"Action: {action}",
        f"Side: {side}",
        f"Symbol: {signal['symbol']}USDT",
        f"Entry: {signal['entry_price']}",
        f"StopLoss: {signal['stop_price']}",
        f"TakeProfit: {signal['target_price']}",
        f"Score: {signal['composite_score']}",
        f"ConfigVersion: {signal['config_version']}",
        f"TierMultiplier: {_compute_tier_multiplier(signal)}",
    ]
    return "\n".join(lines)


def send_signal_to_webhook(signal: Dict) -> bool:
    """Send a single signal to the webhook. Returns True on success."""
    msg_str = format_webhook_msg(signal)
    payload = {
        "key": WEBHOOK_KEY,
        "telegram_alert_type": "trading_bot",
        "msg": msg_str,
    }

    try:
        resp = httpx.post(WEBHOOK_URL, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info(
                "  ✅ Sent %s %s @ %.2f (score=%.1f, %s)",
                signal["direction"], signal["symbol"],
                signal["entry_price"], signal["composite_score"],
                signal.get("_priority_label", ""),
            )
            mark_signal_sent(signal["id"])
            return True
        else:
            logger.warning(
                "  ❌ Webhook returned %d for %s: %s",
                resp.status_code, signal["symbol"], resp.text[:100],
            )
            return False
    except Exception as e:
        logger.error("  ❌ Failed to send %s: %s", signal["symbol"], e)
        return False


# ── Main ────────────────────────────────────────────────────────────────────

def run_bridge(dry_run: bool = False) -> int:
    """Main bridge function. Returns number of signals sent."""
    ensure_webhook_column()

    logger.info("=== Webhook Bridge ===")
    logger.info("Config priority: %s", ", ".join(f"{l} (≥{s:.1f})" for v, s, l in CONFIG_PRIORITY))
    logger.info("Max per batch: %d | Open-position dedup", MAX_SIGNALS_PER_BATCH)

    selected = select_signals()
    if not selected:
        logger.info("Nothing to send")
        return 0

    if dry_run:
        logger.info("=== DRY-RUN — would send %d signals ===", len(selected))
        for sig in selected:
            logger.info(
                "  %s %s @ %.2f (score=%.1f, %s)",
                sig["direction"], sig["symbol"],
                sig["entry_price"], sig["composite_score"],
                sig.get("_priority_label", ""),
            )
        return 0

    sent_count = 0
    for sig in selected:
        ok = send_signal_to_webhook(sig)
        if ok:
            sent_count += 1

    logger.info("Sent %d/%d signals to webhook", sent_count, len(selected))
    return sent_count


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    import sys
    dry = "--dry-run" in sys.argv
    count = run_bridge(dry_run=dry)
    print(f"Sent {count} signals{' (dry-run)' if dry else ''}")