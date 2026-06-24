"""SignalValidator — market-condition checks for incoming trading signals.

Each validator queries the local market-data cache (``market_data`` table)
to determine whether the current environment is favourable for the signal's
direction.
"""

import logging
import sqlite3
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


class SignalValidator:
    """Validates a signal against trend, volume, and volatility criteria."""

    def __init__(self, db_path: str = "data/crypto.db"):
        self.db_path = db_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_recent_closes(self, symbol: str, n: int = 50) -> list:
        """Return the last *n* closing prices for *symbol*."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT close
                FROM market_data
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (symbol, n),
            )
            return [r[0] for r in cursor.fetchall()]
        finally:
            conn.close()

    def _get_recent_volumes(self, symbol: str, n: int = 20) -> list:
        """Return the last *n* volume values for *symbol*."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT volume
                FROM market_data
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (symbol, n),
            )
            return [r[0] for r in cursor.fetchall()]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def validate_trend(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """Check that current price aligns with EMA20 / EMA50 trend.

        A LONG signal passes when price > EMA20 > EMA50 (uptrend).
        A SHORT signal passes when price < EMA20 < EMA50 (downtrend).

        Returns (pass: bool, reason: str).
        """
        closes = self._get_recent_closes(symbol, n=50)
        if len(closes) < 50:
            return True, "Insufficient data — skipping trend check"

        prices = list(reversed(closes))  # oldest → newest
        current_price = prices[-1]

        ema20 = self._ema(prices, 20)
        ema50 = self._ema(prices, 50)

        direction = direction.upper()
        if direction == "LONG":
            if current_price >= ema20 >= ema50:
                return True, f"Uptrend confirmed (price={current_price:.2f} > EMA20={ema20:.2f} > EMA50={ema50:.2f})"
            return False, f"Not in uptrend (price={current_price:.2f}, EMA20={ema20:.2f}, EMA50={ema50:.2f})"
        elif direction == "SHORT":
            if current_price <= ema20 <= ema50:
                return True, f"Downtrend confirmed (price={current_price:.2f} < EMA20={ema20:.2f} < EMA50={ema50:.2f})"
            return False, f"Not in downtrend (price={current_price:.2f}, EMA20={ema20:.2f}, EMA50={ema50:.2f})"
        else:
            return False, f"Unknown direction: {direction}"

    def validate_volume(self, symbol: str) -> Tuple[bool, str]:
        """Check that current volume > 0.5× the 20-bar average.

        Returns (pass: bool, reason: str).
        """
        volumes = self._get_recent_volumes(symbol, n=20)
        if len(volumes) < 10:
            return True, "Insufficient volume data — skipping check"

        current_vol = volumes[0]  # most recent
        avg_vol = sum(volumes) / len(volumes)

        if avg_vol <= 0:
            return True, "Zero average volume — skipping check"

        ratio = current_vol / avg_vol
        if ratio >= 0.5:
            return True, f"Volume OK (current={current_vol:.0f}, avg={avg_vol:.0f}, ratio={ratio:.2f})"
        return False, f"Volume too low (ratio={ratio:.2f}, need ≥0.5)"

    def validate_volatility(self, symbol: str) -> Tuple[bool, str]:
        """Check that ATR% is within a reasonable range (0.3 % – 5 %).

        Uses a 14-bar ATR normalised by the closing price.

        Returns (pass: bool, reason: str).
        """
        closes = self._get_recent_closes(symbol, n=20)
        if len(closes) < 14:
            return True, "Insufficient data — skipping volatility check"

        prices = list(reversed(closes))  # oldest → newest
        atr = self._atr(prices, period=14)
        atr_pct = atr / prices[-1] * 100.0

        if 0.3 <= atr_pct <= 5.0:
            return True, f"Volatility OK (ATR% = {atr_pct:.2f}%)"
        if atr_pct < 0.3:
            return True, f"Low volatility (ATR% = {atr_pct:.2f}%) — market may be quiet but still tradeable"
        return False, f"Volatility too high (ATR% = {atr_pct:.2f}%, max 5%)"

    # ------------------------------------------------------------------
    # Composite
    # ------------------------------------------------------------------

    def validate_all(self, signal: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Run all three validators and return a dict of per-check results.

        Expected signal keys: ``symbol`` (str), ``direction`` (str).
        """
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "").upper()

        valid_trend, reason_trend = self.validate_trend(symbol, direction)
        valid_vol, reason_vol = self.validate_volume(symbol)
        valid_voly, reason_voly = self.validate_volatility(symbol)

        return {
            "trend": {"pass": valid_trend, "reason": reason_trend},
            "volume": {"pass": valid_vol, "reason": reason_vol},
            "volatility": {"pass": valid_voly, "reason": reason_voly},
        }

    # ------------------------------------------------------------------
    # Technical helpers (pure Python)
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(prices: list, period: int) -> float:
        """Exponential moving average (last value)."""
        if len(prices) < period:
            return prices[-1]
        multiplier = 2.0 / (period + 1)
        ema = sum(prices[:period]) / period
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    @staticmethod
    def _atr(prices: list, period: int = 14) -> float:
        """Average True Range as a simple mean of bar-to-bar ranges.

        Because we only have close prices here, we approximate ATR using
        the average absolute price change over the window.
        """
        if len(prices) < period + 1:
            return 0.0
        changes = [abs(prices[i] - prices[i - 1]) for i in range(1, period + 1)]
        return sum(changes) / len(changes)