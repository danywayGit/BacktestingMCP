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
CONFIG_PRIORITY = [
    ("6.2", 7.5, "V6.2 Pullback"),    # 63.6% WR — best performer
    ("3.1", 7.5, "V3.1 ADX Trend"),  # 59.6% WR
    ("4.1", 7.5, "V4.1 Breakout"),    # 58.0% WR
    ("9.0", 7.0, "V9.0 Vol Imbalance"),  # New strategy
    ("1.0", 8.5, "V1.0 Baseline"),    # 52.1% WR, workhorse
]

MAX_SIGNALS_PER_BATCH = 3       # Max positions the bot can handle
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


def select_signals() -> List[Dict]:
    """Select the best signals across all configs using priority + dedup.

    Returns at most MAX_SIGNALS_PER_BATCH signals, with no duplicate symbols.
    """
    from src.integrations.binance_symbols import is_on_binance_futures
    cooldown_symbols = get_open_position_symbols()
    selected = []       # Final selected signals
    selected_symbols = set()  # Symbols already picked in this batch
    sent_count = 0

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

            # BINANCE FUTURES CHECK: symbol must exist on Futures
            if not is_on_binance_futures(sym):
                logger.info(
                    "  %s: SKIPPED %s %s — not on Binance Futures (Spot only)",
                    label, sig["direction"], sym,
                )
                continue

            sig["_priority_label"] = label
            selected.append(sig)
            selected_symbols.add(sym)
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