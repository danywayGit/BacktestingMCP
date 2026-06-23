"""
Multi-version parallel scanning.

Runs all registered scoring configs in a single scan cycle using:
  - ONE altFINS screener call (fetches union of all required display_types)
  - ONE altFINS signal feed call (shared across all configs)
  - N × M scoring passes (N configs × M candidates) — pure CPU, fast

Each signal is tagged with its config_version so win-rate reports can
compare performance across versions once signals resolve.

Only configs that are "enabled for parallel" are included. By default
all configs in ALL_CONFIGS are included. Configs can be disabled via
the DB (is_active=0 means retired — still used for historical data,
but excluded from new parallel scans).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from config.settings import TimeFrame
from ..data.database import db
from ..integrations import altfins_client
from ..integrations.altfins_client import AltfinsError, parse_trend_score
from ..integrations import santiment_client
from ..integrations.santiment_client import SantimentError
from .scoring_config import (
    ScoringConfig, ACTIVE_CONFIG, ALL_CONFIGS,
    is_stablecoin_or_stock, get_coin_type,
)
from .composite import (
    _altfins_to_pair, _safe_float, _signal_feed_index,
    score_symbol, CandidateScore,
)
from . import store as edge_store

logger = logging.getLogger(__name__)


@dataclass
class MultiVersionScanResult:
    """Aggregated results from one parallel multi-version scan cycle."""
    scan_time: str
    timeframe: str
    configs_run: List[str]                          # version strings
    # signals_by_version[version] = list of CandidateScore with direction != None
    signals_by_version: Dict[str, List[CandidateScore]] = field(default_factory=dict)
    total_logged: int = 0
    total_alerts_sent: int = 0
    candidates_count: int = 0


def _get_union_display_types(configs: List[ScoringConfig]) -> List[str]:
    """Return the union of all display_types needed across all given configs."""
    seen: Set[str] = set()
    result: List[str] = []
    for cfg in configs:
        for dt in cfg.get_required_display_types():
            if dt not in seen:
                seen.add(dt)
                result.append(dt)
    return result


def _discover_candidates_multi(
    configs: List[ScoringConfig],
    per_side_size: int,
) -> Dict[str, Dict]:
    """Single altFINS screener call for the union of display_types across all configs.

    Returns {symbol: screener_row} for all non-stablecoin/stock candidates.
    """
    display_types = _get_union_display_types(configs)
    candidates: Dict[str, Dict] = {}
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
                if is_stablecoin_or_stock(symbol):
                    blocked.append(symbol)
                else:
                    candidates[symbol.upper()] = row
    except AltfinsError as exc:
        logger.warning("altFINS screener unavailable: %s", exc)

    if blocked:
        logger.info("Blocked %d stablecoins/stocks: %s", len(blocked), blocked[:10])
    logger.info("Multi-version scan: %d candidates, %d configs, display_types=%s",
                len(candidates), len(configs), display_types)
    return candidates


def run_parallel_scan(
    timeframe: TimeFrame = TimeFrame.H1,
    lookback_days: int = 30,
    per_side_size: int = 20,
    horizon_hours: int = 24,
    configs: Optional[List[ScoringConfig]] = None,
    log_signals: bool = True,
    send_alerts: bool = True,
) -> MultiVersionScanResult:
    """Run all configs in one scan cycle — single API call, multiple scorings.

    Args:
        configs: List of ScoringConfig to run. Defaults to ALL_CONFIGS values.
                 Pass a subset to run only specific versions.
        log_signals: If True, persist all actionable signals to edge_signals DB.
        send_alerts: If True, send Telegram alerts for high-score signals.

    Returns:
        MultiVersionScanResult with per-version signal lists and counts.
    """
    active_configs = configs or list(ALL_CONFIGS.values())
    result = MultiVersionScanResult(
        scan_time=datetime.now(timezone.utc).isoformat(),
        timeframe=timeframe.value,
        configs_run=[c.version for c in active_configs],
    )

    # ── Step 1: Single screener call for union of display_types ──────────
    candidates = _discover_candidates_multi(active_configs, per_side_size)
    result.candidates_count = len(candidates)

    if not candidates:
        logger.warning("No candidates returned from altFINS screener.")
        return result

    # ── Step 2: Single signal feed call (shared across all versions) ──────
    try:
        feed_index = _signal_feed_index(
            altfins_client.get_signal_feed(size=100, lookback="last 7 days")
        )
        logger.info("Signal feed: %d symbols indexed", len(feed_index))
    except AltfinsError as exc:
        logger.warning("Signal feed unavailable: %s", exc)
        feed_index = {}

    # ── Step 3: Score each symbol against each config ─────────────────────
    # Group by symbol first to deduplicate Santiment calls across configs
    # (Santiment is already cached 6h in santiment_client, so this is safe)
    for cfg in active_configs:
        scored: List[CandidateScore] = []
        for symbol, row in candidates.items():
            candidate = score_symbol(
                symbol, row, feed_index, timeframe, lookback_days, config=cfg
            )
            scored.append(candidate)

        # Sort by abs score descending
        scored.sort(key=lambda c: abs(c.composite_score), reverse=True)
        actionable = [c for c in scored if c.direction is not None]
        result.signals_by_version[cfg.version] = actionable

        logger.info("Config %s: %d actionable signals (LONG=%d SHORT=%d)",
                    cfg.version,
                    len(actionable),
                    sum(1 for c in actionable if c.direction == "LONG"),
                    sum(1 for c in actionable if c.direction == "SHORT"))

    # ── Step 4: Log all signals to DB tagged by version ───────────────────
    if log_signals:
        total = 0
        for version, signals in result.signals_by_version.items():
            cfg = ALL_CONFIGS.get(version)
            if cfg is None:
                continue
            logged = edge_store.log_signals(signals, timeframe, horizon_hours=horizon_hours)
            total += logged
        result.total_logged = total
        logger.info("Logged %d total signals across %d configs", total, len(active_configs))

    # ── Step 5: Alerts — only from ACTIVE_CONFIG to avoid spam ───────────
    # Alerts fire only for the active production config, not all 14 versions
    if send_alerts:
        from . import alerts as edge_alerts
        active_signals = result.signals_by_version.get(ACTIVE_CONFIG.version, [])
        if active_signals:
            sent = edge_alerts.send_alerts(active_signals)
            result.total_alerts_sent = sent

    return result


def format_summary(result: MultiVersionScanResult, top_n: int = 5) -> str:
    """Format a human-readable summary of a multi-version scan result."""
    lines = [
        f"📊 *Multi-version scan* — {result.timeframe} — {result.candidates_count} candidates",
        f"Configs: {len(result.configs_run)} | Logged: {result.total_logged} signals",
        "",
    ]

    # Show top signals from ACTIVE_CONFIG
    active_signals = result.signals_by_version.get(ACTIVE_CONFIG.version, [])
    if active_signals:
        lines.append(f"*Top signals (v{ACTIVE_CONFIG.version} active):*")
        for s in active_signals[:top_n]:
            emoji = "🟢" if s.direction == "LONG" else "🔴"
            lines.append(f"  {emoji} {s.symbol} {s.direction} score={s.composite_score:+.2f}")

    lines.append("")

    # Signal count comparison across versions
    lines.append("*Signals per version:*")
    # Sort by signal count descending
    version_counts = sorted(
        [(v, len(sigs)) for v, sigs in result.signals_by_version.items()],
        key=lambda x: x[1], reverse=True
    )
    for version, count in version_counts[:8]:  # top 8
        active_marker = " ✅" if version == ACTIVE_CONFIG.version else ""
        lines.append(f"  {version}{active_marker}: {count} signals")

    if result.total_alerts_sent:
        lines.append(f"\n🔔 {result.total_alerts_sent} alert(s) sent to @CryptoAlertsTradingView")

    return "\n".join(lines)
