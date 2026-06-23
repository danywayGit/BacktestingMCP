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
from .scoring_config import (
    ScoringConfig, ACTIVE_CONFIG, get_coin_type,
    is_stablecoin_or_stock, parse_trend_score_extended,
    parse_float as _safe_float_sc,
)

logger = logging.getLogger(__name__)

# Legacy module-level constants — kept for backward compat, sourced from ACTIVE_CONFIG.
TREND_WEIGHT = ACTIVE_CONFIG.trend_weight
VOLUME_RELATIVE_WEIGHT = ACTIVE_CONFIG.volume_relative_weight
VOLUME_RELATIVE_CAP = ACTIVE_CONFIG.volume_relative_cap
SCANNER_HIT_WEIGHT = ACTIVE_CONFIG.scanner_hit_weight
SIGNAL_FEED_WEIGHT = ACTIVE_CONFIG.signal_feed_weight
ONCHAIN_NETFLOW_WEIGHT = ACTIVE_CONFIG.onchain_netflow_weight
ONCHAIN_LOOKBACK_DAYS = 3
DEFAULT_MIN_ABS_SCORE = ACTIVE_CONFIG.min_abs_score

# ── Symbol blocklist ──────────────────────────────────────────────────────────
# Stablecoins and tokenized stocks produce meaningless trend/volume signals
# and must never be logged as tradeable candidates.
#
# Stablecoins: their "uptrend" is just peg maintenance noise.
# Tokenized stocks (xStock platform, suffix X/XUSDT or known tickers):
#   they are not available on Binance Futures and have no crypto liquidity.
def _is_blocked(symbol: str) -> bool:
    """Return True if symbol should never be scored — delegates to global blocklist in scoring_config."""
    return is_stablecoin_or_stock(symbol)


@dataclass
class CandidateScore:
    symbol: str               # altFINS-style base symbol, e.g. "BTC"
    pair: str                 # exchange pair, e.g. "BTCUSDT"
    composite_score: float
    direction: Optional[str]  # "LONG" / "SHORT" / None
    last_close: Optional[float]
    components: Dict[str, Any] = field(default_factory=dict)
    config_version: str = "v1.0"   # which ScoringConfig produced this signal
    coin_type: str = "OTHER"        # LAYER1 / LAYER2 / DEFI / MEME / AI / etc.


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


def discover_candidates(per_side_size: int = 20, config: Optional[ScoringConfig] = None) -> Dict[str, Dict[str, Any]]:
    """Returns {altfins_symbol: screener_row} for bullish + bearish screens.

    Fetches the display_types required by the given config (defaults to ACTIVE_CONFIG).
    Filters out stablecoins and tokenized stocks globally — these are never tradeable.
    """
    cfg = config or ACTIVE_CONFIG
    display_types = cfg.get_required_display_types()

    candidates: Dict[str, Dict[str, Any]] = {}
    blocked: List[str] = []
    try:
        for direction in ("UP", "DOWN"):
            rows = altfins_client.get_screener_data(
                display_types=display_types,
                signal_filter_value=direction,
                size=per_side_size,
            )
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
    logger.info("discover_candidates: %d tradeable candidates (config=%s)", len(candidates), cfg.version)
    return candidates


def score_symbol(
    symbol: str,
    screener_row: Dict[str, Any],
    signal_feed_index: Dict[str, str],
    timeframe: TimeFrame,
    lookback_days: int,
    config: Optional[ScoringConfig] = None,
) -> CandidateScore:
    """Score a single symbol using the given ScoringConfig (defaults to ACTIVE_CONFIG)."""
    cfg = config or ACTIVE_CONFIG
    pair = _altfins_to_pair(symbol)
    coin_type = get_coin_type(symbol)
    components: Dict[str, Any] = {}
    score = 0.0

    # Apply config filters before scoring — fail fast
    passes, filter_reason = cfg.passes_filters(symbol, screener_row)
    if not passes:
        components["filtered_out"] = filter_reason
        return CandidateScore(
            symbol=symbol, pair=pair, composite_score=0.0, direction=None,
            last_close=None, components=components,
            config_version=cfg.version, coin_type=coin_type,
        )

    additional = screener_row.get("additionalData", {}) if screener_row else {}
    trend_score = parse_trend_score(additional.get("SHORT_TERM_TREND"))
    volume_relative = _safe_float(additional.get("VOLUME_RELATIVE"), default=1.0)

    score += trend_score * cfg.trend_weight
    components["altfins_trend_score"] = trend_score

    vol_excess = min(max(volume_relative - 1.0, 0.0), cfg.volume_relative_cap)
    if trend_score != 0:
        score += vol_excess * cfg.volume_relative_weight * (1 if trend_score > 0 else -1)
    components["altfins_volume_relative"] = volume_relative

    feed_direction = signal_feed_index.get(symbol.upper())
    if feed_direction == "BULLISH":
        score += cfg.signal_feed_weight
    elif feed_direction == "BEARISH":
        score -= cfg.signal_feed_weight
    components["altfins_signal_feed"] = feed_direction

    netflow_ratio: Optional[float] = None
    altfins_score_so_far = score
    if santiment_client.slug_for(symbol) and abs(altfins_score_so_far) >= 2.0:
        try:
            onchain = santiment_client.get_onchain_snapshot(symbol, lookback_days=ONCHAIN_LOOKBACK_DAYS)
            inflow = onchain.get("exchange_inflow_usd", 0.0)
            outflow = onchain.get("exchange_outflow_usd", 0.0)
            total = inflow + outflow
            if total > 0:
                netflow_ratio = (outflow - inflow) / total
                score += netflow_ratio * cfg.onchain_netflow_weight
        except SantimentError as exc:
            logger.debug("No on-chain data for %s: %s", symbol, exc)
    components["onchain_netflow_ratio"] = netflow_ratio

    # Extended scoring signals (ADX, OBV, RSI, TR/ATR, price momentum)
    direction_hint = score  # positive = bullish context so far
    extra_score, extra_components = cfg.compute_extended_score(additional, score, direction_hint)
    score += extra_score
    components.update(extra_components)

    last_close: Optional[float] = None
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
            last_close = float(data["Close"].iloc[-1])
            scan_result = evaluate_scan(data, "all")
            triggered_scans = [name for name, details in scan_result.items() if details.get("triggered")]
            score += len(triggered_scans) * cfg.scanner_hit_weight
    except Exception as exc:
        logger.warning("Could not fetch/scan OHLCV for %s: %s", pair, exc)
    components["backtestingmcp_scanner_hits"] = triggered_scans
    components["coin_type"] = coin_type

    direction: Optional[str] = None
    if score >= cfg.min_abs_score:
        direction = "LONG"
    elif score <= -cfg.min_abs_score:
        direction = "SHORT"

    return CandidateScore(
        symbol=symbol,
        pair=pair,
        composite_score=round(score, 2),
        direction=direction,
        last_close=last_close,
        components=components,
        config_version=cfg.version,
        coin_type=coin_type,
    )


def run_composite_scan(
    timeframe: TimeFrame = TimeFrame.H1,
    lookback_days: int = 30,
    per_side_size: int = 20,
    config: Optional[ScoringConfig] = None,
) -> List[CandidateScore]:
    """Run a full scan cycle using the given config (defaults to ACTIVE_CONFIG)."""
    cfg = config or ACTIVE_CONFIG
    candidates = discover_candidates(per_side_size=per_side_size, config=cfg)

    try:
        feed_index = _signal_feed_index(altfins_client.get_signal_feed(size=100, lookback="last 7 days"))
    except AltfinsError:
        feed_index = {}

    results = [
        score_symbol(symbol, row, feed_index, timeframe, lookback_days, config=cfg)
        for symbol, row in candidates.items()
    ]
    results.sort(key=lambda c: abs(c.composite_score), reverse=True)
    return results
