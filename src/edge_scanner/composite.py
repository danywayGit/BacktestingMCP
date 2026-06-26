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
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

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
from ..integrations.binance_symbols import is_on_binance_futures

# ── Regime detection cache ─────────────────────────────────────────────────
_regime_cache: dict = {}
_regime_cache_time: datetime | None = None
REGIME_CACHE_TTL_SEC = 900  # 15 min cache lifetime

def _detect_regime_cached() -> dict:
    """Get the current market regime, with a 15-minute cache to avoid
    redundant OHLCV fetches during multi-symbol scans."""
    global _regime_cache, _regime_cache_time
    now = datetime.now(timezone.utc)
    if _regime_cache_time is not None and (now - _regime_cache_time).total_seconds() < REGIME_CACHE_TTL_SEC:
        return _regime_cache
    try:
        from .regime_detector import detect_regime
        _regime_cache = detect_regime('BTC/USDT', 'H1', 30)
        _regime_cache_time = now
    except Exception as exc:
        logger.warning("Regime detection failed: %s", exc)
        _regime_cache = {"regime": "UNKNOWN", "adx": 0.0, "volatility": 0.0,
                         "price_trend": 0.0, "confidence": 0.0}
        _regime_cache_time = now
    return _regime_cache


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
    if is_stablecoin_or_stock(symbol):
        return True
    if not is_on_binance_futures(symbol):
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


def _filtered_result(
    symbol: str, pair: str, coin_type: str, config_version: str,
    components: Dict[str, Any],
) -> CandidateScore:
    """Build a filtered-out CandidateScore (score=0, direction=None)."""
    return CandidateScore(
        symbol=symbol, pair=pair, composite_score=0.0, direction=None,
        last_close=None, components=components,
        config_version=config_version, coin_type=coin_type,
    )


def _compute_atr_pct(data: pd.DataFrame, period: int = 14) -> float:
    """Compute ATR as a percentage of the latest close price.
    
    Returns ATR% (e.g. 0.5 means 0.5% of close price). Returns 0.0 if
    insufficient data.
    """
    if len(data) < period + 1:
        return 0.0
    high = data["High"].values
    low = data["Low"].values
    close = data["Close"].values
    tr = np.maximum(high[1:] - low[1:],
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1]))
    if len(tr) < period:
        return 0.0
    atr = np.mean(tr[-period:])
    last_close = float(close[-1])
    if last_close == 0:
        return 0.0
    return (atr / last_close) * 100.0


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

            # ATR volatility filter — reject low-volatility setups
            if cfg.min_atr_pct > 0:
                atr_val = _compute_atr_pct(data, period=14)
                if atr_val < cfg.min_atr_pct:
                    components["filtered_out"] = f"ATR%={atr_val:.2f}% < min={cfg.min_atr_pct}%"
                    return _filtered_result(symbol, pair, coin_type, cfg.version, components)

            # Volume divergence — leading indicator (catches moves before trend)
            div_adj = 0.0
            if cfg.volume_divergence_weight > 0 or cfg.smart_money_index_weight > 0 or cfg.low_float_squeeze_weight > 0:
                try:
                    from .volume_divergence import compute_divergence_score, compute_smart_money_index, is_low_float_squeeze

                    if cfg.volume_divergence_weight > 0:
                        d_adj, d_details = compute_divergence_score(data)
                        div_adj += d_adj * cfg.volume_divergence_weight
                        components.update(d_details)

                    if cfg.smart_money_index_weight > 0:
                        sm_adj, sm_details = compute_smart_money_index(data)
                        div_adj += sm_adj * cfg.smart_money_index_weight
                        components.update(sm_details)

                    if cfg.low_float_squeeze_weight > 0:
                        is_sqz, sqz_score = is_low_float_squeeze(data, coin_type)
                        if is_sqz:
                            div_adj += sqz_score * cfg.low_float_squeeze_weight
                            components["low_float_squeeze"] = sqz_score

                    if div_adj != 0:
                        score += div_adj
                        components["volume_divergence_total"] = round(div_adj, 2)
                except Exception as exc:
                    logger.debug("Volume divergence computation failed: %s", exc)
    except Exception as exc:
        logger.warning("Could not fetch/scan OHLCV for %s: %s", pair, exc)
    components["backtestingmcp_scanner_hits"] = triggered_scans
    components["coin_type"] = coin_type

    direction: Optional[str] = None

    # Apply config filters at scoring stage
    trend = components.get("altfins_trend_score", 0)
    if cfg.min_trend_abs_score > 0 and abs(trend) < cfg.min_trend_abs_score:
        components["failed_trend_min"] = f"|trend|={abs(trend):.0f} < min={cfg.min_trend_abs_score}"
    elif cfg.require_non_trend_confirmation and abs(trend) > 0:
        # At least one non-trend source must confirm
        vol = components.get("altfins_volume_relative", 0)
        feed = components.get("altfins_signal_feed")
        scanner = components.get("backtestingmcp_scanner_hits", [])
        netflow = components.get("onchain_netflow_ratio")
        has_confirmation = (
            vol > 1.0
            or (feed == "BULLISH" and trend > 0)
            or (feed == "BEARISH" and trend < 0)
            or len(scanner) > 0
            or (netflow is not None and abs(netflow) > 0.05)
        )
        if not has_confirmation:
            components["failed_confirmation"] = "no non-trend source confirms"

    # Direction assignment with separate SHORT threshold
    short_min = cfg.short_min_abs_score if cfg.short_min_abs_score is not None else cfg.min_abs_score
    if score >= cfg.min_abs_score:
        direction = "LONG"
    elif score <= -short_min:
        direction = "SHORT"

    # ── Regime-aware direction bias ──────────────────────────────────────
    # In BEAR_TRENDING, SHORT signals get a bonus and LONG signals get a penalty
    # In BULL_TRENDING, the reverse applies.
    regime_info = _detect_regime_cached()
    regime = regime_info.get("regime", "UNKNOWN")
    if direction == "SHORT" and regime == "BEAR_TRENDING" and cfg.regime_dir_bear_short_bonus > 0:
        score += cfg.regime_dir_bear_short_bonus
        components["regime_bias"] = f"bear+short={cfg.regime_dir_bear_short_bonus:+.0f}"
    elif direction == "LONG" and regime == "BEAR_TRENDING" and cfg.regime_dir_bear_long_penalty > 0:
        score -= cfg.regime_dir_bear_long_penalty
        components["regime_bias"] = f"bear+long={-cfg.regime_dir_bear_long_penalty:+.0f}"
    elif direction == "LONG" and regime == "BULL_TRENDING" and cfg.regime_dir_bull_long_bonus > 0:
        score += cfg.regime_dir_bull_long_bonus
        components["regime_bias"] = f"bull+long={cfg.regime_dir_bull_long_bonus:+.0f}"
    elif direction == "SHORT" and regime == "BULL_TRENDING" and cfg.regime_dir_bull_short_penalty > 0:
        score -= cfg.regime_dir_bull_short_penalty
        components["regime_bias"] = f"bull+short={-cfg.regime_dir_bull_short_penalty:+.0f}"
    components["regime"] = regime

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


def run_all_versions(
    timeframe: TimeFrame = TimeFrame.H1,
    lookback_days: int = 30,
    per_side_size: int = 20,
    configs: Optional[List[ScoringConfig]] = None,
) -> Dict[str, List[CandidateScore]]:
    """Score candidates against ALL config versions in a single API round-trip.

    This is the parallel multi-version scan:
      1. One altFINS screener call fetches candidates with the union of all
         required display_types across all configs.
      2. One altFINS signal feed call fetches the feed index.
      3. Each config independently scores the same candidate pool.
      4. Returns {version: [CandidateScore, ...]} for every config.

    This is far more efficient than calling run_composite_scan() 14 times
    (which would make 14× altFINS API calls and 14× OHLCV downloads).
    OHLCV data is cached in SQLite after the first download, so only new
    symbols pay a download cost on the first cycle.
    """
    from .scoring_config import ALL_CONFIGS

    all_cfgs: List[ScoringConfig] = configs or list(ALL_CONFIGS.values())

    # ── Step 1: union of all display_types needed across all configs ──────
    all_display_types: List[str] = []
    for cfg in all_cfgs:
        for dt in cfg.get_required_display_types():
            if dt not in all_display_types:
                all_display_types.append(dt)

    # Use the baseline config as the "driver" for discovery (broadest universe)
    # but pass the union of display_types so every config has its fields.
    logger.info("run_all_versions: %d configs, %d display_types, fetching candidates...",
                len(all_cfgs), len(all_display_types))

    # Fetch candidates once with the full display_type union
    candidates: Dict[str, Dict[str, Any]] = {}
    blocked: List[str] = []
    try:
        for direction in ("UP", "DOWN"):
            rows = altfins_client.get_screener_data(
                display_types=all_display_types,
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
        logger.warning("altFINS unavailable (%s); falling back to static universe.", exc)
        for pair in CRYPTO_PAIRS:
            sym = _pair_to_altfins(pair)
            if not _is_blocked(sym):
                candidates[sym] = {}

    if blocked:
        logger.info("run_all_versions: blocked %d stablecoin/stock symbols", len(blocked))
    logger.info("run_all_versions: %d candidates → scoring across %d versions",
                len(candidates), len(all_cfgs))

    # ── Step 2: one signal feed fetch shared by all configs ───────────────
    try:
        feed_index = _signal_feed_index(
            altfins_client.get_signal_feed(size=100, lookback="last 7 days")
        )
    except AltfinsError:
        feed_index = {}

    # ── Step 3: score each config against the shared candidate pool ───────
    results: Dict[str, List[CandidateScore]] = {}
    for cfg in all_cfgs:
        version_scores = [
            score_symbol(symbol, row, feed_index, timeframe, lookback_days, config=cfg)
            for symbol, row in candidates.items()
        ]
        version_scores.sort(key=lambda c: abs(c.composite_score), reverse=True)
        results[cfg.version] = version_scores
        actionable = sum(1 for s in version_scores if s.direction is not None)
        logger.info("  %s → %d actionable signals", cfg.version, actionable)

    return results
