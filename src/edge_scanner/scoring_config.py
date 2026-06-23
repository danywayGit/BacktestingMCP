"""
Scoring configuration versioning for the edge scanner.

Every scan cycle uses a named, versioned ScoringConfig. The version is
stored alongside each signal in edge_signals so that win-rate reports can
attribute results to the exact config that generated them.

This is the foundation for Phase 5 (Hermes self-evolution): once enough
resolved signals exist, the weights and filters in each version can be
compared objectively and the best-performing config promoted.

Design
------
- ScoringConfig is a frozen dataclass — immutable once created.
- Configs are stored in the SQLite `scoring_configs` table.
- The "active" config is flagged in the DB; only one can be active at a time.
- Changing a weight or filter creates a NEW version (bumps the minor version).
  The old version is retired but kept for historical win-rate attribution.
- The DB is the single source of truth; module constants are the defaults for
  bootstrapping when the DB is empty.

Versioning scheme:  v<major>.<minor>
  major — breaking change (e.g. new signal source added)
  minor — weight/filter tuning within the same signal sources

Coin type categories (for coin_type_filter):
  "LAYER1"  — L1 chains: BTC, ETH, SOL, AVAX, ADA, ...
  "LAYER2"  — L2 rollups: ARB, OP, MATIC, STRKUSDT, ...
  "DEFI"    — DeFi protocols: UNI, AAVE, CRV, JUP, ...
  "MEME"    — Meme coins: DOGE, SHIB, PEPE, FLOKI, ...
  "AI"      — AI/data tokens: FET, AGIX, RENDER, TAO, ...
  "INFRA"   — Infrastructure: LINK, GRT, OCEAN, ...
  "ANY"     — No filter (default)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coin type classification map
# Covers the most common Binance Futures symbols.
# Unclassified symbols default to "OTHER".
# ---------------------------------------------------------------------------
COIN_TYPE_MAP: dict[str, str] = {
    # Layer 1
    "BTC": "LAYER1", "ETH": "LAYER1", "SOL": "LAYER1", "AVAX": "LAYER1",
    "ADA": "LAYER1", "DOT": "LAYER1", "ATOM": "LAYER1", "NEAR": "LAYER1",
    "APT": "LAYER1", "SUI": "LAYER1", "SEI": "LAYER1", "TIA": "LAYER1",
    "INJ": "LAYER1", "TRX": "LAYER1", "XRP": "LAYER1", "BNB": "LAYER1",
    "ALGO": "LAYER1", "XLM": "LAYER1", "VET": "LAYER1", "ETC": "LAYER1",
    "FIL": "LAYER1", "ICP": "LAYER1", "HBAR": "LAYER1", "XTZ": "LAYER1",
    "EOS": "LAYER1", "FLOW": "LAYER1", "EGLD": "LAYER1", "ZIL": "LAYER1",
    # Layer 2 / Scaling
    "ARB": "LAYER2", "OP": "LAYER2", "MATIC": "LAYER2", "STRK": "LAYER2",
    "IMX": "LAYER2", "METIS": "LAYER2", "BOBA": "LAYER2", "ZKJ": "LAYER2",
    "MANTA": "LAYER2", "SCROLL": "LAYER2", "ZKEVM": "LAYER2",
    # DeFi
    "UNI": "DEFI", "AAVE": "DEFI", "CRV": "DEFI", "MKR": "DEFI",
    "SNX": "DEFI", "COMP": "DEFI", "YFI": "DEFI", "SUSHI": "DEFI",
    "JUP": "DEFI", "ORCA": "DEFI", "RAY": "DEFI", "CAKE": "DEFI",
    "GMX": "DEFI", "DYDX": "DEFI", "PENDLE": "DEFI", "LDO": "DEFI",
    "EIGEN": "DEFI", "ENA": "DEFI", "ETHFI": "DEFI", "RUNE": "DEFI",
    "WLD": "DEFI", "JTO": "DEFI", "TIA": "DEFI", "KMNO": "DEFI",
    # Meme
    "DOGE": "MEME", "SHIB": "MEME", "PEPE": "MEME", "FLOKI": "MEME",
    "BONK": "MEME", "WIF": "MEME", "BOME": "MEME", "MEW": "MEME",
    "NEIRO": "MEME", "TURBO": "MEME", "MOG": "MEME", "BRETT": "MEME",
    "APE": "MEME", "BABYDOGE": "MEME", "LADYS": "MEME",
    # AI / Data
    "FET": "AI", "AGIX": "AI", "RENDER": "AI", "TAO": "AI",
    "OCEAN": "AI", "NMR": "AI", "GRT": "AI", "CTXC": "AI",
    "AIOZ": "AI", "RSS3": "AI", "VOXEL": "AI",
    # Infrastructure / Oracle / Bridge
    "LINK": "INFRA", "DOT": "INFRA", "KSM": "INFRA", "API3": "INFRA",
    "BAND": "INFRA", "TRB": "INFRA", "ZRO": "INFRA", "PYTH": "INFRA",
    "W": "INFRA", "WBTC": "INFRA",
    # Gaming / NFT / Metaverse
    "AXS": "GAMING", "SAND": "GAMING", "MANA": "GAMING", "ENJ": "GAMING",
    "GALA": "GAMING", "IMX": "GAMING", "PIXEL": "GAMING", "PRIME": "GAMING",
    "MAGIC": "GAMING", "YGG": "GAMING", "PYR": "GAMING",
}


def get_coin_type(symbol: str) -> str:
    """Return the coin type category for a symbol, defaulting to 'OTHER'."""
    return COIN_TYPE_MAP.get(symbol.upper(), "OTHER")


# ---------------------------------------------------------------------------
# ScoringConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringConfig:
    """Immutable scoring configuration snapshot.

    Every field that influences which signals get logged or how they're
    scored should live here — so win-rate attribution is always exact.
    """
    # Version identifier
    version: str = "v1.0"
    description: str = "Initial hand-tuned config"

    # --- Scoring weights ---
    trend_weight: float = 0.4
    """altFINS SHORT_TERM_TREND score weight (raw range -10..+10)."""

    volume_relative_weight: float = 2.0
    """Per unit of (relative_volume - 1), directionally signed."""

    volume_relative_cap: float = 3.0
    """Max excess volume contribution (prevents single spike dominating)."""

    scanner_hit_weight: float = 2.5
    """Per triggered BacktestingMCP TA breakout scanner."""

    signal_feed_weight: float = 3.0
    """altFINS direct signal feed confirmation."""

    onchain_netflow_weight: float = 2.0
    """Santiment exchange-flow net ratio in [-1, 1]."""

    # --- Entry threshold ---
    min_abs_score: float = 3.0
    """Minimum |composite_score| to log a signal for tracking."""

    alert_min_score: float = 7.0
    """Minimum |composite_score| to send a Telegram alert."""

    alert_require_multi_source: bool = True
    """Telegram alerts require >= 2 independent sources confirming."""

    # --- Filters ---
    min_market_cap_usd: float = 0.0
    """Minimum market cap in USD. 0 = no filter.
    Suggested values: 50_000_000 (micro), 500_000_000 (mid), 1_000_000_000 (large)."""

    min_volume_relative: float = 0.0
    """Minimum relative volume (vs 10-bar avg). 0 = no filter.
    1.0 = at least average volume. 1.5 = at least 50% above average."""

    coin_type_filter: List[str] = field(default_factory=lambda: ["ANY"])
    """Coin type whitelist. ['ANY'] = no filter. Otherwise only these types pass.
    Options: LAYER1, LAYER2, DEFI, MEME, AI, INFRA, GAMING, OTHER."""

    exclude_coin_types: List[str] = field(default_factory=list)
    """Coin types to always exclude (e.g. ['MEME'] for conservative scanning)."""

    # --- Metadata ---
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ScoringConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def passes_filters(self, symbol: str, screener_row: dict) -> tuple[bool, str]:
        """Check if a symbol passes this config's filters.

        Returns (passes: bool, reason: str).
        reason is empty string if passes=True.
        """
        additional = screener_row.get("additionalData", {}) if screener_row else {}

        # Volume filter
        if self.min_volume_relative > 0:
            try:
                vol = float(str(additional.get("VOLUME_RELATIVE", "0")).replace(",", ""))
                if vol < self.min_volume_relative:
                    return False, f"vol_relative={vol:.2f} < min={self.min_volume_relative}"
            except (ValueError, TypeError):
                pass

        # Market cap filter
        if self.min_market_cap_usd > 0:
            mc_raw = screener_row.get("marketCap") if screener_row else None
            if mc_raw is not None:
                try:
                    mc = float(str(mc_raw).replace(",", "").replace("$", ""))
                    if mc < self.min_market_cap_usd:
                        return False, f"market_cap=${mc:,.0f} < min=${self.min_market_cap_usd:,.0f}"
                except (ValueError, TypeError):
                    pass

        # Coin type filter
        coin_type = get_coin_type(symbol)
        if "ANY" not in self.coin_type_filter and coin_type not in self.coin_type_filter:
            return False, f"coin_type={coin_type} not in {self.coin_type_filter}"
        if coin_type in self.exclude_coin_types:
            return False, f"coin_type={coin_type} excluded"

        return True, ""


# ---------------------------------------------------------------------------
# Pre-defined config variants
# Add new variants here when you want to A/B test a different hypothesis.
# ---------------------------------------------------------------------------

# v1.0 — baseline, hand-tuned, no extra filters
CONFIG_V1_0 = ScoringConfig(
    version="v1.0",
    description="Baseline: altFINS trend+volume+signal+on-chain. No market cap or coin type filter.",
)

# v1.1 — same weights, adds minimum relative volume filter
CONFIG_V1_1 = ScoringConfig(
    version="v1.1",
    description="v1.0 + min volume_relative >= 1.0 (at least average volume required).",
    min_volume_relative=1.0,
)

# v1.2 — large cap only (>$500M market cap), excludes memes
CONFIG_V1_2 = ScoringConfig(
    version="v1.2",
    description="Large cap filter: market_cap > $500M, excludes MEME coins.",
    min_market_cap_usd=500_000_000,
    exclude_coin_types=["MEME"],
)

# v1.3 — DeFi + Layer1/2 focus with volume filter
CONFIG_V1_3 = ScoringConfig(
    version="v1.3",
    description="DeFi+L1+L2 focus: coin_type in [LAYER1, LAYER2, DEFI], vol >= 1.0.",
    coin_type_filter=["LAYER1", "LAYER2", "DEFI"],
    min_volume_relative=1.0,
)

# v1.4 — higher signal feed weight (hypothesis: signal feed is more predictive)
CONFIG_V1_4 = ScoringConfig(
    version="v1.4",
    description="Higher signal feed weight (4.0 vs 3.0). Tests if signal feed confirmation predicts better.",
    signal_feed_weight=4.0,
    trend_weight=0.3,  # slightly reduce trend to compensate
)

# Active config — change this to switch which version runs in production.
# All signals logged will carry this version in config_version column.
ACTIVE_CONFIG: ScoringConfig = CONFIG_V1_0

ALL_CONFIGS: dict[str, ScoringConfig] = {
    c.version: c for c in [CONFIG_V1_0, CONFIG_V1_1, CONFIG_V1_2, CONFIG_V1_3, CONFIG_V1_4]
}
