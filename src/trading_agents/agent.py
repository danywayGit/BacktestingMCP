"""TradingAgent — the main execution loop.

Fetches the best unresolved edge-scanner signals, validates them against
current market conditions, and simulates trade entry / exit.
"""

import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .position_tracker import PositionTracker
from .signal_validator import SignalValidator

logger = logging.getLogger(__name__)


class TradingAgent:
    """Autonomous trading agent that processes edge-scanner signals."""

    def __init__(self, db_path: str = "data/crypto.db", min_score: float = 7.0):
        self.db_path = db_path
        self.min_score = min_score
        self.tracker = PositionTracker(db_path=db_path)
        self.validator = SignalValidator(db_path=db_path)
        self.tracker.create_trades_table()

    # ------------------------------------------------------------------
    # Signal fetching
    # ------------------------------------------------------------------

    def fetch_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Pull PENDING edge signals with |composite_score| >= min_score.

        Returns a list of signal dicts (symbol, direction, score,
        last_close, config_version, created_at).
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, symbol, pair, direction, composite_score,
                       entry_price, config_version, created_at
                FROM edge_signals
                WHERE status = 'PENDING'
                  AND ABS(composite_score) >= ?
                ORDER BY ABS(composite_score) DESC
                LIMIT ?
                """,
                (self.min_score, limit),
            )
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "symbol": r[1],
                    "pair": r[2],
                    "direction": r[3],
                    "score": r[4],
                    "last_close": r[5],
                    "config_version": r[6],
                    "created_at": r[7],
                }
                for r in rows
            ]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Signal validation (placeholder)
    # ------------------------------------------------------------------

    def validate_signal(self, signal: Dict[str, Any]) -> Tuple[bool, str]:
        """Placeholder validation: checks recency, valid price, and direction.

        Returns (valid: bool, reason: str).
        """
        # 1. Age check — signal must be < 24 h old
        created_raw = signal.get("created_at")
        if created_raw:
            try:
                created_dt = datetime.fromisoformat(created_raw)
                age = datetime.now(timezone.utc) - created_dt
                if age > timedelta(hours=24):
                    return False, f"Signal too old ({age.total_seconds() / 3600:.1f}h)"
            except (ValueError, TypeError):
                pass  # malformed timestamp — skip age check

        # 2. Valid last_close / entry_price
        close = signal.get("last_close")
        if close is None or not isinstance(close, (int, float)) or close <= 0:
            return False, f"Invalid entry_price / last_close: {close}"

        # 3. Direction must be LONG or SHORT
        direction = signal.get("direction", "").upper()
        if direction not in ("LONG", "SHORT"):
            return False, f"Unknown direction: {direction}"

        # 4. Let the SignalValidator run market-condition checks
        valid_all = self.validator.validate_all(
            {"symbol": signal.get("symbol"), "direction": direction}
        )
        if not valid_all["trend"]["pass"]:
            return False, f"Trend check failed: {valid_all['trend']['reason']}"
        if not valid_all["volume"]["pass"]:
            return False, f"Volume check failed: {valid_all['volume']['reason']}"
        if not valid_all["volatility"]["pass"]:
            return False, f"Volatility check failed: {valid_all['volatility']['reason']}"

        return True, "Signal passed all validation checks"

    # ------------------------------------------------------------------
    # Trade execution (simulated)
    # ------------------------------------------------------------------

    def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Log a simulated trade based on a validated signal.

        Uses a fixed quantity of 0.001 BTC (or equivalent in quote terms)
        as placeholder position size.
        """
        direction = signal.get("direction", "").upper()
        entry_price = float(signal.get("last_close", 0))
        if entry_price <= 0:
            return {"success": False, "reason": "Invalid entry price"}

        # Fixed placeholder quantity
        quantity = 0.001
        trade_id = self.tracker.open_trade(
            signal_id=signal.get("id", 0),
            symbol=signal.get("symbol", ""),
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
        )
        logger.info(
            "Opened %s trade #%s on %s @ %.2f",
            direction,
            trade_id,
            signal.get("symbol"),
            entry_price,
        )
        return {"success": True, "trade_id": trade_id, "entry_price": entry_price}

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------

    def process_signals(self) -> Dict[str, Any]:
        """Fetch, validate, and execute the best pending signals.

        Returns a summary dict with counts.
        """
        signals = self.fetch_signals(limit=10)
        total = len(signals)
        validated = 0
        executed = 0
        failed = []
        results = []

        for sig in signals:
            valid, reason = self.validate_signal(sig)
            if not valid:
                failed.append({"signal_id": sig.get("id"), "reason": reason})
                continue
            validated += 1

            result = self.execute_trade(sig)
            if result.get("success"):
                executed += 1
                results.append(result)
            else:
                failed.append({"signal_id": sig.get("id"), "reason": result.get("reason")})

        summary = {
            "total": total,
            "validated": validated,
            "executed": executed,
            "failed": len(failed),
            "failures": failed,
            "results": results,
        }
        logger.info("Process signals summary: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Return list of currently OPEN trades."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, signal_id, symbol, direction, entry_price,
                       quantity, entry_time
                FROM trades
                WHERE status = 'OPEN'
                ORDER BY entry_time DESC
                """
            )
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            conn.close()

    def close_position(self, trade_id: int, exit_price: float) -> Dict[str, Any]:
        """Close an OPEN trade, calculate PnL, and return the result."""
        pnl_pct = self.tracker.close_trade(trade_id, exit_price)
        return {
            "trade_id": trade_id,
            "exit_price": exit_price,
            "pnl_pct": round(pnl_pct, 4) if pnl_pct is not None else None,
            "success": pnl_pct is not None,
        }