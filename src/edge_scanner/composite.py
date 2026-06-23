"""
Composite edge scoring.

Pipeline per scan cycle:
  1. Discover candidates from altFINS' screener (bullish + bearish
     SHORT_TERM_TREND filters) across its 2000+ symbol universe.
  2. Cross-check each candidate against altFINS' direct signal feed
     (the API equivalent of the altFINS VIP Telegram signal channel).
  3. Add on-chain exchange-flow bias from Santiment (net outflow reads
     bullish/accumulation, net inflow reads bearish/sell-pressure).
  4. Confirm with BacktestingMCP's own price/volume breakout scanners on
     real OHLCV (so the score isn't just trusting a third party).
  5. Combine into one composite score; symbols crossing the threshold in
     either direction become tracked signals (see store.py).

If ALTFINS_API_KEY isn't configured, falls back to a static symbol
universe and TA-only scoring so the scanner still works. SANTIMENT_API_KEY
is optional too -- on-chain component degrades to None per-symbol if
unset, unmapped (small-cap symbols), or unreachable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from config.settings import CRYPTO_PAIRS, TimeFrame
from ..core.backtesting_engine import engine
from ..strategies.scanner import evaluate_scan
from ..integrations import altfins_client
from ..integrations.altfins_client import AltfinsError, parse_trend_score
from ..integrations import santiment_client
from ..integrations.santiment_client import SantimentError

logger = logging.getLogger(__name__)

# Scoring weights. Kept as module constants (not config) until backtested
# evidence justifies tuning them per-symbol/timeframe.
TREND_WEIGHT = 0.4          # altFINS SHORT_TERM_TREND, raw range -10..10
VOLUME_RELATIVE_WEIGHT = 2.0  # per unit of (volume_relative - 1), capped
VOLUME_RELATIVE_CAP = 3.0
SCANNER_HIT_WEIGHT = 2.5    # per triggered BacktestingMCP breakout scan
SIGNAL_FEED_WEIGHT = 3.0    # altFINS direct signal feed agreement
ONCHAIN_NETFLOW_WEIGHT = 2.0  # per unit of net exchange outflow ratio, -1..1
ONCHAIN_LOOKBACK_DAYS = 3

DEFAULT_MIN_ABS_SCORE = 3.0

# ── Symbol blocklist ──────────────────────────────────────────────────────────
# Stablecoins and tokenized stocks produce meaningless trend/volume signals
# and must never be logged as tradeable candidates.
#
# Stablecoins: their "uptrend" is just peg maintenance noise.
# Tokenized stocks (xStock platform, suffix X/XUSDT or known tickers):
#   they are not available on Binance Futures and have no crypto liquidity.
_STABLECOINS: frozenset[str] = frozenset({
    "USDT", "USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP", "GUSD",
    "FRAX", "LUSD", "SUSD", "USDD", "USDG", "PYUSD", "USDB", "CRVUSD",
    "USDE", "SUSDE", "STABLE", "FDUSD", "LISUSD", "USD0",
})

# Tokenized-stock symbols from xStock/tokenized platforms.
# These end in X on altFINS (INTCX, HOODX, ABBVX, QQQX, SPXX, TQQQX, etc.)
# or match known stock tickers repurposed as crypto (US, IN, AT, BR, etc.)
_TOKENIZED_STOCK_SUFFIXES: tuple[str, ...] = ("X",)  # e.g. INTCX, HOODX, QQQX
_TOKENIZED_STOCK_EXACT: frozenset[str] = frozenset({
    # Short single/double-letter "country" or stock tracker symbols
    "US", "IN", "AT", "BR", "VX", "VR", "MET", "MY", "EV", "B2",
    # Known xStock tickers seen in first scan
    "INTCX", "HOODX", "QQQX", "SPXX", "TQQQX", "ABBVX", "CSCOX",
    "JPMX", "UNHX", "MRVLX", "EDGEX", "STRCX", "BACX", "BPUSDT",
})


def _is_blocked(symbol: str) -> bool:
    """Return True if symbol should never be scored/logged as a candidate."""
    s = symbol.upper()
    if s in _STABLECOINS:
        return True
    if s in _TOKENIZED_STOCK_EXACT:
        return True
    # xStock tokens: length > 3, end in X, and the base (without X) looks like
    # a stock ticker (all-caps, 2-5 chars). We also accept known crypto tokens
    # ending in X (e.g. OX, WX) so only block if length >= 4.
    if len(s) >= 4 and s.endswith("X") and s[:-1].isalpha():
        base = s[:-1]
        # Crypto exceptions that legitimately end in X
        _CRYPTO_X_EXCEPTIONS = {"OX", "INX", "KSX", "HEX", "LEX", "LUMX"}
        if base not in _CRYPTO_X_EXCEPTIONS:
            return True
    return False


@dataclass
class CandidateScore:
    symbol: str               # altFINS-style base symbol, e.g. "BTC"
    pair: str                 # exchange pair, e.g. "BTCUSDT"
    composite_score: float
    direction: Optional[str]  # "LONG" / "SHORT" / None
    last_close: Optional[float]
    components: Dict[str, Any] = field(default_factory=dict)


def _altfins_to_pair(symbol: str) -> str:
    return f"{symbol.upper()}USDT"


def _pair_to_altfins(pair: str) -> str:
    return pair[:-4] if pair.upper().endswith("USDT") else pair


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def _signal_feed_index(entries: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map symbol -> 'BULLISH'/'BEARISH' from the raw signal feed payload.

    Verified against live altFINS API (Jun 2026). Real response fields:
      symbol, name, timestamp, direction (BULLISH/BEARISH),
      signalKey, signalName, lastPrice, priceChange, marketCap.

    A symbol may appear multiple times (different signal types); last entry
    wins, which is fine since we only need the directional bias.
    """
    index: Dict[str, str] = {}
    for entry in entries:
        symbol = entry.get("symbol")
        if not symbol:
            continue
        direction_raw = str(entry.get("direction", "")).upper()
        if "BULL" in direction_raw:
            index[str(symbol).upper()] = "BULLISH"
        elif "BEAR" in direction_raw:
            index[str(symbol).upper()] = "BEARISH"
    return index


def discover_candidates(per_side_size: int = 20) -> Dict[str, Dict[str, Any]]:
    """Returns {altfins_symbol: screener_row} for bullish + bearish screens.

    Filters out stablecoins and tokenized stocks — they produce meaningless
    trend/volume signals and are not tradeable on Binance Futures.
    Falls back to the static CRYPTO_PAIRS universe (with no screener row)
    if altFINS isn't configured.
    """
    candidates: Dict[str, Dict[str, Any]] = {}
    blocked: List[str] = []
    try:
        for direction in ("UP", "DOWN"):
            rows = altfins_client.get_screener_data(signal_filter_value=direction, size=per_side_size)
            for row in rows:
                symbol = row.get("symbol")
                if not symbol:
                    continue
                if _is_blocked(symbol):
                    blocked.append(symbol)
                else:
                    candidates[symbol.upper()] = row
    except AltfinsError as exc:
        logger.warning("altFINS unavailable (%s); falling back to static universe, TA-only.", exc)
        for pair in CRYPTO_PAIRS:
            sym = _pair_to_altfins(pair)
            if not _is_blocked(sym):
                candidates[sym] = {}
    if blocked:
        logger.info("discover_candidates: blocked %d stablecoin/stock symbols: %s", len(blocked), blocked)
    logger.info("discover_candidates: %d tradeable candidates", len(candidates))
    return candidates


def score_symbol(
    symbol: str,
    screener_row: Dict[str, Any],
    signal_feed_index: Dict[str, str],
    timeframe: TimeFrame,
    lookback_days: int,
) -> CandidateScore:
    pair = _altfins_to_pair(symbol)
    components: Dict[str, Any] = {}
    score = 0.0

    additional = screener_row.get("additionalData", {}) if screener_row else {}
    trend_score = parse_trend_score(additional.get("SHORT_TERM_TREND"))
    volume_relative = _safe_float(additional.get("VOLUME_RELATIVE"), default=1.0)

    score += trend_score * TREND_WEIGHT
    components["altfins_trend_score"] = trend_score

    vol_excess = min(max(volume_relative - 1.0, 0.0), VOLUME_RELATIVE_CAP)
    if trend_score != 0:
        score += vol_excess * VOLUME_RELATIVE_WEIGHT * (1 if trend_score > 0 else -1)
    components["altfins_volume_relative"] = volume_relative

    feed_direction = signal_feed_index.get(symbol.upper())
    if feed_direction == "BULLISH":
        score += SIGNAL_FEED_WEIGHT
    elif feed_direction == "BEARISH":
        score -= SIGNAL_FEED_WEIGHT
    components["altfins_signal_feed"] = feed_direction

    netflow_ratio: Optional[float] = None
    # Only call Santiment if the symbol has a slug mapping AND the altFINS
    # score is already directionally meaningful — conserves the free-tier budget
    # (~33 calls/day) by skipping weak/neutral candidates.
    altfins_score_so_far = score  # trend + volume + signal feed already applied
    if santiment_client.slug_for(symbol) and abs(altfins_score_so_far) >= 2.0:
        try:
            onchain = santiment_client.get_onchain_snapshot(symbol, lookback_days=ONCHAIN_LOOKBACK_DAYS)
            inflow = onchain.get("exchange_inflow_usd", 0.0)
            outflow = onchain.get("exchange_outflow_usd", 0.0)
            total = inflow + outflow
            if total > 0:
                # Net outflow (coins leaving exchanges) reads bullish/accumulation;
                # net inflow (coins arriving, available to sell) reads bearish.
                netflow_ratio = (outflow - inflow) / total
                score += netflow_ratio * ONCHAIN_NETFLOW_WEIGHT
        except SantimentError as exc:
            logger.debug("No on-chain data for %s: %s", symbol, exc)
    components["onchain_netflow_ratio"] = netflow_ratio

    last_close: Optional[float] = None
    # Use screener lastPrice as fallback before attempting OHLCV fetch
    try:
        raw_price = screener_row.get("lastPrice")
        if raw_price is not None:
            last_close = float(str(raw_price).replace(",", ""))
    except (TypeError, ValueError):
        pass

    triggered_scans: List[str] = []
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days)
        data = engine.get_data(pair, timeframe, start_date, end_date)
        if not data.empty:
            last_close = float(data["Close"].iloc[-1])  # prefer OHLCV close over screener price
            scan_result = evaluate_scan(data, "all")
            triggered_scans = [name for name, details in scan_result.items() if details.get("triggered")]
            # BacktestingMCP's current scanners are all upside-breakout
            # patterns, so a hit only ever adds to the bullish side.
            score += len(triggered_scans) * SCANNER_HIT_WEIGHT
    except Exception as exc:  # noqa: BLE001 - data gaps/network issues shouldn't kill the whole scan
        logger.warning("Could not fetch/scan OHLCV for %s: %s", pair, exc)
    components["backtestingmcp_scanner_hits"] = triggered_scans

    direction: Optional[str] = None
    if score >= DEFAULT_MIN_ABS_SCORE:
        direction = "LONG"
    elif score <= -DEFAULT_MIN_ABS_SCORE:
        direction = "SHORT"

    return CandidateScore(
        symbol=symbol,
        pair=pair,
        composite_score=round(score, 2),
        direction=direction,
        last_close=last_close,
        components=components,
    )


def run_composite_scan(
    timeframe: TimeFrame = TimeFrame.H1,
    lookback_days: int = 30,
    per_side_size: int = 20,
) -> List[CandidateScore]:
    candidates = discover_candidates(per_side_size=per_side_size)

    try:
        feed_index = _signal_feed_index(altfins_client.get_signal_feed(size=100, lookback="last 7 days"))
    except AltfinsError:
        feed_index = {}

    results = [
        score_symbol(symbol, row, feed_index, timeframe, lookback_days)
        for symbol, row in candidates.items()
    ]
    results.sort(key=lambda c: abs(c.composite_score), reverse=True)
    return results
