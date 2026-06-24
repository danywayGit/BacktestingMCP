"""PositionTracker — persistent trade logging and PnL tracking.

Maintains a ``trades`` table in the project SQLite database alongside the
existing ``edge_signals`` table so that backtest and live results live
under the same roof.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TRADES_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    exit_price REAL,
    pnl_pct REAL
)
"""


class PositionTracker:
    """CRUD-style wrapper around the ``trades`` table."""

    def __init__(self, db_path: str = "data/crypto.db"):
        self.db_path = db_path

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def create_trades_table(self) -> None:
        """Create the trades table if it does not already exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(TRADES_DDL)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)"
            )
            conn.commit()
            logger.info("Ensured trades table exists in %s", self.db_path)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Open / close trades
    # ------------------------------------------------------------------

    def open_trade(
        self,
        signal_id: int,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: float,
    ) -> Optional[int]:
        """Insert a new OPEN trade. Returns the new trade id."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO trades
                    (signal_id, symbol, direction, entry_price, quantity,
                     status, entry_time)
                VALUES (?, ?, ?, ?, ?, 'OPEN', ?)
                """,
                (
                    signal_id,
                    symbol,
                    direction.upper(),
                    entry_price,
                    quantity,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            trade_id = cursor.lastrowid
            logger.debug("Opened trade #%s: %s %s @ %.4f", trade_id, direction, symbol, entry_price)
            return trade_id
        finally:
            conn.close()

    def close_trade(self, trade_id: int, exit_price: float) -> Optional[float]:
        """Close an OPEN trade.

        Calculates PnL percent based on direction:
          - LONG:  (exit - entry) / entry * 100
          - SHORT: (entry - exit) / entry * 100

        Returns the PnL percent, or *None* if the trade was not found or
        already closed.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Fetch trade details
            cursor.execute(
                "SELECT direction, entry_price FROM trades WHERE id = ? AND status = 'OPEN'",
                (trade_id,),
            )
            row = cursor.fetchone()
            if row is None:
                logger.warning("Trade #%s not found or already closed", trade_id)
                return None

            direction, entry_price = row
            direction = direction.upper()

            if direction == "LONG":
                pnl_pct = (exit_price - entry_price) / entry_price * 100.0
            elif direction == "SHORT":
                pnl_pct = (entry_price - exit_price) / entry_price * 100.0
            else:
                logger.error("Unknown direction '%s' for trade #%s", direction, trade_id)
                return None

            cursor.execute(
                """
                UPDATE trades
                SET status = 'CLOSED',
                    exit_time = ?,
                    exit_price = ?,
                    pnl_pct = ?
                WHERE id = ?
                """,
                (datetime.now(timezone.utc).isoformat(), exit_price, pnl_pct, trade_id),
            )
            conn.commit()
            logger.info(
                "Closed trade #%s %s @ %.2f — PnL: %.2f%%",
                trade_id,
                direction,
                exit_price,
                pnl_pct,
            )
            return pnl_pct
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Read / stats
    # ------------------------------------------------------------------

    def get_all_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent trades ordered by entry time descending."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, signal_id, symbol, direction, entry_price, quantity,
                       status, entry_time, exit_time, exit_price, pnl_pct
                FROM trades
                ORDER BY entry_time DESC
                LIMIT ?
                """,
                (limit,),
            )
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_trade_stats(self) -> Dict[str, Any]:
        """Return aggregate performance statistics over all CLOSED trades."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*)                                         AS total_trades,
                    COALESCE(SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END), 0) AS win_count,
                    COALESCE(SUM(CASE WHEN pnl_pct <= 0 THEN 1 ELSE 0 END), 0) AS loss_count,
                    ROUND(AVG(pnl_pct), 4)                           AS avg_pnl,
                    ROUND(SUM(pnl_pct), 4)                           AS total_pnl,
                    ROUND(MAX(pnl_pct), 4)                           AS best_trade,
                    ROUND(MIN(pnl_pct), 4)                           AS worst_trade
                FROM trades
                WHERE status = 'CLOSED'
                """
            )
            row = cursor.fetchone()
            if row is None or row[0] == 0:
                return {
                    "total_trades": 0,
                    "win_count": 0,
                    "loss_count": 0,
                    "win_rate": 0.0,
                    "avg_pnl": 0.0,
                    "total_pnl": 0.0,
                    "best_trade": 0.0,
                    "worst_trade": 0.0,
                }

            columns = [d[0] for d in cursor.description]
            stats = dict(zip(columns, row))
            total = stats["total_trades"]
            stats["win_rate"] = round(stats["win_count"] / total * 100, 2) if total else 0.0
            return stats
        finally:
            conn.close()