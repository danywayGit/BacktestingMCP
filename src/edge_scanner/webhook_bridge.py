"""
Edge Scanner → Trading-WebHook-Bot bridge.

Polls the edge_signals database for high-confidence PENDING signals
and sends them to the webhook for execution on Binance TestNet.

Filters:
- Only from ACTIVE_CONFIG (V7.0)
- Score >= 8.0
- Not already sent (webhook_sent_at IS NULL)
- Has valid entry/stop/target prices
"""

import httpx
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
WEBHOOK_URL = "http://109.123.229.200/webhook"
WEBHOOK_KEY = "6XO7toihtxsSW7s9OgetPwVwjMCNhb4O"
ACCOUNT_TYPE = "TestNet"  # Change to "Standard" for live trading
EXCHANGE = "Binance"
STRATEGY = "ManualTrading"
MIN_SCORE = 8.0
DB_PATH = "/home/hermes/BacktestingMCP/data/crypto.db"

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
        pass  # Column already exists
    finally:
        db.close()


def get_pending_webhook_signals() -> List[Dict]:
    """Fetch signals ready to send to webhook.

    Criteria:
    - From the active config version (V7.0)
    - Still PENDING (outcome IS NULL)
    - Score >= MIN_SCORE
    - Valid entry, stop, target prices
    - Not already sent to webhook
    """
    from src.edge_scanner.scoring_config import ACTIVE_CONFIG
    active_ver = ACTIVE_CONFIG.version

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
        LIMIT 5
    """, (active_ver, MIN_SCORE)).fetchall()
    db.close()

    return [dict(r) for r in rows]


def mark_signal_sent(signal_id: int):
    """Mark a signal as sent to webhook."""
    db = get_db()
    db.execute(
        "UPDATE edge_signals SET webhook_sent_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), signal_id)
    )
    db.commit()
    db.close()


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
        f"Symbol: {signal['symbol']}",
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
                "Sent %s %s @ %.2f (score=%.1f) → %s",
                signal["direction"], signal["symbol"],
                signal["entry_price"], signal["composite_score"],
                resp.text[:50],
            )
            mark_signal_sent(signal["id"])
            return True
        else:
            logger.warning(
                "Webhook returned %d for %s: %s",
                resp.status_code, signal["symbol"], resp.text[:100],
            )
            return False
    except Exception as e:
        logger.error("Failed to send %s to webhook: %s", signal["symbol"], e)
        return False


def run_bridge(dry_run: bool = False) -> int:
    """Main bridge function. Returns number of signals sent."""
    ensure_webhook_column()
    signals = get_pending_webhook_signals()

    if not signals:
        logger.info("No qualifying signals to send")
        return 0

    logger.info("Found %d signals to send", len(signals))

    sent_count = 0
    for sig in signals:
        msg_str = format_webhook_msg(sig)
        sym = sig["symbol"]
        direction = sig["direction"]
        score = sig["composite_score"]

        if dry_run:
            logger.info(
                "[DRY-RUN] Would send %s %s @ %.2f (score=%.1f)",
                direction, sym, sig["entry_price"], score,
            )
            continue

        success = send_signal_to_webhook(sig)
        if success:
            sent_count += 1

    if sent_count:
        logger.info("Sent %d/%d signals to webhook", sent_count, len(signals))
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