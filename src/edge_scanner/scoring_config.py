"""
Scoring configuration versioning for the edge scanner.

Every scan cycle uses a named, versioned ScoringConfig. The version is
stored alongside each signal in edge_signals so that win-rate reports can
attribute results to the exact config that generated them.

This is the foundation for Phase 5 (self-evolution): once enough resolved
signals exist, the weights and filters in each version can be compared
objectively and the best-performing config promoted automatically.

Design
------
- ScoringConfig is a frozen dataclass — immutable once created.
- Configs are stored in the SQLite `scoring_configs` table.
- The "active" config is flagged in the DB; only one can be active at a time.
- Changing a weight or filter creates a NEW version. The old version is
  retired but kept for historical win-rate attribution.
- The DB is the single source of truth; module constants bootstrap when empty.

Versioning scheme:  v<major>.<minor>
  major — structural change (new signal source, new scoring formula type)
  minor — weight/filter tuning within the same signal sources

Coin type categories:
  LAYER1, LAYER2, DEFI, MEME, AI, INFRA, GAMING, OTHER

Available altFINS data fields (121 total, key ones used here):
  SHORT_TERM_TREND, MEDIUM_TERM_TREND, LONG_TERM_TREND
  VOLUME_RELATIVE, MARKET_CAP, RSI14, ADX, ATR, TR_VS_ATR
  MACD, MACD_SIGNAL, OBV_TREND, STOCH_RSI
  PRICE_CHANGE_1D/1W/1M, SHORT_TERM_TREND_CHANGE
  EMA50, EMA200, TVL, TOTAL_REVENUE_1W
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global stablecoin blocklist — ALWAYS excluded regardless of config version.
# These tokens have no tradeable edge — their "trend" is just peg maintenance.
# ---------------------------------------------------------------------------
STABLECOIN_SYMBOLS: frozenset[str] = frozenset({
    # USD-pegged
    "USDT", "USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP", "GUSD",
    "FRAX", "LUSD", "SUSD", "USDD", "USDG", "PYUSD", "USDB", "CRVUSD",
    "USDE", "SUSDE", "USDC0", "LISUSD", "USD0", "FDUSD", "STABLE",
    "USDBR", "ZUSD", "EURS", "EURT", "USDBC", "USDS", "USDR",
    # Euro/other pegged
    "EURC", "AGEUR", "JEUR",
    # Wrapped/LST (not tradeable on futures as trend signals)
    "WBTC", "WETH", "WBNB", "STETH", "WSTETH", "RETH", "CBETH",
    "LSETH", "METH", "WEETH", "RSETH", "EZETH",
    # Algorithmic / rebasing
    "AMPL", "ESD", "BAC",
})

# Tokenized stocks — not on Binance Futures, produce meaningless signals
# Pattern: 4+ chars ending in X where base looks like a stock ticker
_TOKENIZED_STOCK_EXACT: frozenset[str] = frozenset({
    "INTCX", "HOODX", "QQQX", "SPXX", "TQQQX", "ABBVX", "CSCOX",
    "JPMX", "UNHX", "MRVLX", "EDGEX", "STRCX", "BACX", "VTIX",
    "CRWDX", "TBLLX", "DEGENX", "NVOX", "APEX",
    # Short ambiguous tickers that are not real crypto
    "US", "IN", "AT", "BR", "VX", "VR", "MET", "MY", "EV", "B2",
    "VR", "M", "H", "B", "CC",
})

def is_stablecoin_or_stock(symbol: str) -> bool:
    """Return True if symbol should ALWAYS be excluded (stablecoin or tokenized stock)."""
    s = symbol.upper()
    if s in STABLECOIN_SYMBOLS:
        return True
    if s in _TOKENIZED_STOCK_EXACT:
        return True
    # xStock pattern: 4+ chars, ends in X, base is alphabetic (e.g. INTCX, QQQX)
    # Exceptions: legitimate crypto tokens ending in X
    _CRYPTO_X_OK = {"OX", "INX", "KSX", "HEX", "LEX", "LUMX", "AX", "MX", "SX"}
    if len(s) >= 4 and s.endswith("X") and s[:-1].isalpha() and s not in _CRYPTO_X_OK:
        return True
    return False

# ---------------------------------------------------------------------------
# Coin type classification
# ---------------------------------------------------------------------------
COIN_TYPE_MAP: dict[str, str] = {
    # Layer 1 blockchains
    "BTC": "LAYER1", "ETH": "LAYER1", "SOL": "LAYER1", "AVAX": "LAYER1",
    "ADA": "LAYER1", "DOT": "LAYER1", "ATOM": "LAYER1", "NEAR": "LAYER1",
    "APT": "LAYER1", "SUI": "LAYER1", "SEI": "LAYER1", "TIA": "LAYER1",
    "INJ": "LAYER1", "TRX": "LAYER1", "XRP": "LAYER1", "BNB": "LAYER1",
    "ALGO": "LAYER1", "XLM": "LAYER1", "VET": "LAYER1", "ETC": "LAYER1",
    "FIL": "LAYER1", "ICP": "LAYER1", "HBAR": "LAYER1", "XTZ": "LAYER1",
    "EOS": "LAYER1", "FLOW": "LAYER1", "EGLD": "LAYER1", "ZIL": "LAYER1",
    "TON": "LAYER1", "KAS": "LAYER1", "XMR": "LAYER1", "ZEC": "LAYER1",
    "KAIA": "LAYER1", "CFX": "LAYER1", "BCH": "LAYER1", "LTC": "LAYER1",
    "BSV": "LAYER1", "DASH": "LAYER1", "XDC": "LAYER1",
    "CLO": "LAYER1",  # Callisto Network
    # Layer 2 / Scaling
    "ARB": "LAYER2", "OP": "LAYER2", "MATIC": "LAYER2", "STRK": "LAYER2",
    "IMX": "LAYER2", "METIS": "LAYER2", "ZRO": "LAYER2", "STX": "LAYER2",
    "MANTA": "LAYER2", "BLAST": "LAYER2", "ZKEVM": "LAYER2",
    "MNT": "LAYER2",  # Mantle
    # DeFi protocols
    "UNI": "DEFI", "AAVE": "DEFI", "CRV": "DEFI", "MKR": "DEFI",
    "SNX": "DEFI", "COMP": "DEFI", "YFI": "DEFI", "SUSHI": "DEFI",
    "JUP": "DEFI", "ORCA": "DEFI", "RAY": "DEFI", "CAKE": "DEFI",
    "GMX": "DEFI", "DYDX": "DEFI", "PENDLE": "DEFI", "LDO": "DEFI",
    "EIGEN": "DEFI", "ENA": "DEFI", "ETHFI": "DEFI", "RUNE": "DEFI",
    "WLD": "DEFI", "JTO": "DEFI", "KMNO": "DEFI", "ONDO": "DEFI",
    "ZRX": "DEFI", "BAL": "DEFI", "1INCH": "DEFI", "PERP": "DEFI",
    "OKB": "DEFI",  # OKX exchange token
    "DEXE": "DEFI",  # DeXe Protocol
    "AERO": "DEFI",  # Aerodrome DEX
    "XVS": "DEFI",  # Venus Protocol (lending)
    "VELODROME": "DEFI",  # Velodrome DEX
    "HYPE": "DEFI",  # Hyperliquid
    "SUN": "DEFI",  # Sun Token (TRON DeFi)
    "JITOSOL": "DEFI",  # Liquid staking on Solana
    "ALLO": "DEFI",  # Allo protocol
    "BLESS": "DEFI",  # Bless ecosystem
    # Meme coins
    "DOGE": "MEME", "SHIB": "MEME", "PEPE": "MEME", "FLOKI": "MEME",
    "BONK": "MEME", "WIF": "MEME", "BOME": "MEME", "MEW": "MEME",
    "NEIRO": "MEME", "TURBO": "MEME", "MOG": "MEME", "BRETT": "MEME",
    "APE": "MEME", "BABYDOGE": "MEME", "LADYS": "MEME", "CHEEMS": "MEME",
    "MEME": "MEME", "COQ": "MEME",
    "POPCAT": "MEME", "TRUTH": "MEME", "FOLKS": "MEME",
    "CATI": "MEME", "GSX": "MEME",
    "GPS": "MEME",
    # AI / Data tokens
    "FET": "AI", "AGIX": "AI", "RENDER": "AI", "TAO": "AI",
    "OCEAN": "AI", "NMR": "AI", "GRT": "AI", "CTXC": "AI",
    "AIOZ": "AI", "RSS3": "AI", "VOXEL": "AI", "ARKM": "AI",
    "MYRIA": "AI", "ALT": "AI",
    "IO": "AI",  # io.net (decentralized GPU)
    "COAI": "AI", "SKYAI": "AI", "UAI": "AI", "AIO": "AI",
    "GRASS": "AI",  # Grass Foundation (data scraping for AI)
    "HANA": "AI",
    # Infrastructure / Oracle / Bridge / Interop
    "LINK": "INFRA", "KSM": "INFRA", "API3": "INFRA", "BAND": "INFRA",
    "TRB": "INFRA", "PYTH": "INFRA", "W": "INFRA", "AXL": "INFRA",
    "CELR": "INFRA", "HOP": "INFRA", "ACX": "INFRA",
    "SYN": "INFRA",  # Synapse (cross-chain bridge)
    "GEOD": "INFRA",  # Geodnet (decentralized GPS)
    "BICO": "INFRA",  # Biconomy (relayer infra)
    "DBR": "INFRA",  # deBridge (cross-chain)
    "SQD": "INFRA",  # Subsquid (indexing)
    "PHA": "INFRA",  # Phala Network (privacy cloud)
    "AWE": "INFRA",
    "BP": "INFRA",
    "BTT": "INFRA",  # BitTorrent (file sharing network)
    # Gaming / NFT / Metaverse
    "AXS": "GAMING", "SAND": "GAMING", "MANA": "GAMING", "ENJ": "GAMING",
    "GALA": "GAMING", "PIXEL": "GAMING", "PRIME": "GAMING", "ILV": "GAMING",
    "BEAM": "GAMING", "RON": "GAMING",
    "ZEREBRO": "GAMING",  # Zerebro (gaming ecosystem)
    "CARDS": "GAMING",
    # RWA / Tokenized Assets
    "ONDO": "DEFI",  # ONDO Finance (RWA)
}

def get_coin_type(symbol: str) -> str:
    """Return coin type category, defaulting to 'OTHER'."""
    return COIN_TYPE_MAP.get(symbol.upper(), "OTHER")

# ---------------------------------------------------------------------------
# Helper parsers for altFINS data fields
# ---------------------------------------------------------------------------

_TREND_RE = re.compile(r"(Strong\s+)?(Up|Down)\s*\((\d+)/10\)", re.IGNORECASE)

def parse_trend_score_extended(label: str) -> float:
    """Parse any altFINS trend label into a signed -10..+10 float."""
    if not isinstance(label, str):
        return 0.0
    m = _TREND_RE.search(label)
    if not m:
        return 0.0
    _, direction, magnitude = m.groups()
    score = float(magnitude)
    return score if direction.lower() == "up" else -score

def parse_float(value, default: float = 0.0) -> float:
    """Safe float parse for altFINS values that may contain commas or %."""
    try:
        return float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return default

def parse_adx(value) -> float:
    """ADX is always positive (0-100). Returns 0 if missing."""
    return max(0.0, parse_float(value, 0.0))

def parse_obv_trend(value) -> float:
    """OBV_TREND comes as a percentage string like '104.45%' or '-12.3%'.
    Returns a signed float: positive = rising OBV (accumulation)."""
    return parse_float(value, 0.0)

def parse_rsi(value) -> float:
    """RSI 0-100. Returns 50 (neutral) if missing."""
    v = parse_float(value, 50.0)
    return max(0.0, min(100.0, v))

def parse_market_cap(value) -> float:
    """Parse market cap which may come as '1,250,963,823,334' or similar."""
    return parse_float(value, 0.0)

def parse_price_change_pct(value) -> float:
    """Parse price change like '-0.06%' or '12.5%'."""
    s = str(value).replace("%", "").replace(",", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0

# ---------------------------------------------------------------------------
# ScoringConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringConfig:
    """Immutable scoring configuration snapshot.

    All fields that influence which signals get logged or how they're
    scored live here — so win-rate attribution is always exact and
    reproducible for any past version.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    version: str = "v1.0"
    description: str = ""

    # ── Scoring formula ───────────────────────────────────────────────────
    # Core weights
    trend_weight: float = 0.4
    """Weight for SHORT_TERM_TREND score (-10..+10 → multiplied)."""

    volume_relative_weight: float = 2.0
    """Per unit of (relative_volume - 1), directionally signed."""

    volume_relative_cap: float = 3.0
    """Cap on volume excess contribution (prevents single spike dominating)."""

    signal_feed_weight: float = 3.0
    """altFINS analyst signal feed: +/- this value."""

    scanner_hit_weight: float = 2.5
    """Per triggered BacktestingMCP TA breakout scanner."""

    onchain_netflow_weight: float = 2.0
    """Santiment exchange-flow net ratio in [-1, 1]."""

    # Extended scoring signals (0.0 = disabled)
    adx_weight: float = 0.0
    """ADX trend strength bonus. ADX > 25 = trending market.
    Score contribution: (adx - 25) * adx_weight if trending, else 0."""

    obv_trend_weight: float = 0.0
    """OBV trend % change. Positive = accumulation bias (bullish).
    Score contribution: obv_trend_pct * obv_trend_weight (capped ±1.0)."""

    medium_term_trend_weight: float = 0.0
    """MEDIUM_TERM_TREND score (-10..+10) extra multiplier.
    Adds directional confirmation across timeframes."""

    rsi_momentum_weight: float = 0.0
    """RSI momentum bonus. RSI 50-70 (bullish momentum zone) adds score.
    RSI 30-50 (bearish momentum zone) subtracts. Extremes (overbought/oversold) ignored."""

    price_change_1w_weight: float = 0.0
    """1-week price change % contribution. Captures recent price momentum.
    Score = clamp(price_change_1w_pct / 10, -1, 1) * weight."""

    tr_vs_atr_weight: float = 0.0
    """TR/ATR ratio — measures if current volatility exceeds average range.
    TR/ATR > 1.5 signals a breakout candle. Score = (tr_vs_atr - 1) * weight."""

    # ── Entry thresholds ──────────────────────────────────────────────────
    min_abs_score: float = 3.0
    """Minimum |composite_score| to log a signal for tracking."""

    short_min_abs_score: Optional[float] = None
    """Separate lower threshold for SHORT signals. None = uses min_abs_score.
    Set to e.g. 3.5 to make SHORT signals easier to fire (fewer SHORT sources)."""

    min_trend_abs_score: float = 0.0
    """Minimum absolute SHORT_TERM_TREND score required. 0 = no filter.
    Set to 5.0 to require at least "Up (5/10)" for LONG or "Down (5/10)" for SHORT."""

    require_non_trend_confirmation: bool = False
    """If True, at least one non-trend source must confirm the direction:
    volume_relative > 1.0 OR signal feed match OR scanner hit OR on-chain match.
    Prevents signals based on trend alone."""

    # ── Volume Divergence (leading indicator) ──────────────────────────────
    volume_divergence_weight: float = 0.0
    """Weight for volume-price divergence score. Non-zero enables early-entry
    divergence detection that catches moves before the consensus trend forms."""

    smart_money_index_weight: float = 0.0
    """Weight for smart-money accumulation index. Based on up-volume vs
    down-volume imbalance. Detects buy/sell pressure."""

    low_float_squeeze_weight: float = 0.0
    """Weight for low-float volume squeeze detection. High scores for
    AI/AGENT/MEME coins with sudden volume spikes."""

    # ── Funding Rate Mean-Reversion (leading indicator) ─────────────────────
    funding_rate_weight: float = 0.0
    """Weight for extreme funding rate signal. Funding rate < -0.006 (LONG)
    or > +0.006 (SHORT) adds score proportional to extremity.
    Score contribution: |funding_rate| * 100 * weight (e.g., 0.008 * 100 * 5 = +4)."""

    funding_momentum_weight: float = 0.0
    """Weight for funding rate momentum (rate of change). When funding is
    extreme and normalizing, adds bonus: +score for LONG if funding rising,
    +score for SHORT if funding falling."""

    oi_change_weight: float = 0.0
    """Weight for open interest change confirmation. OI declining = short
    covering (bullish for LONG). OI rising = short adding (bearish)."""

    min_abs_funding_rate: float = 0.0
    """Minimum absolute funding rate to activate funding scoring.
    0.006 = 0.6% extreme threshold. 0 = no filter."""

    min_funding_momentum: float = 0.0
    """Minimum funding rate change per hour to contribute score.
    0.00003 = +0.003%/hour. 0 = no filter."""

    max_oi_change: float = 0.0
    """Maximum absolute OI change fraction to allow funding scoring.
    0.02 = 2% max OI increase. 0 = no filter."""

    # ── Funding Interval & Pre-Funding Dip (added Jun 2026) ──────────────
    funding_interval_weight: float = 0.0
    """Weight for funding interval bonus. Shorter intervals (1h) are more
    expensive than longer (8h). Scaled: score = (8/interval) × weight.
    8h = 1×, 4h = 2×, 2h = 4×, 1h = 8×."""

    pre_funding_dip_weight: float = 0.0
    """Weight for pre-funding dip entry bonus. When funding is extreme
    and next funding is 15-45 min away, adds score bonus for the expected
    dip/settlement bounce. Higher = more aggressive dip capture."""

    interval_switch_weight: float = 0.0
    """Weight for funding interval switch signal. When a symbol's funding
    interval decreases (e.g., 4h→1h), it signals a volatility event.
    Adds score bonus proportional to the interval change magnitude."""

    # ── Risk management (target/stop computation) ──────────────────────────
    atr_stop_mult: float = 1.5
    """ATR multiplier for stop loss placement. stop = entry ± (ATR × mult).
    SWING strategies optimized to 1.5-2.0. Higher = wider stop, more room."""

    rr_ratio: float = 2.0
    """Risk-to-reward ratio. target = entry ± (stop_distance × RR).
    SWING strategies optimized to 1.5-2.5. Higher = more ambitious targets."""

    # ── Alert thresholds ────────────────────────────────────────────────────
    alert_min_score: float = 7.0
    """Minimum absolute composite score to trigger a Telegram alert."""

    alert_require_multi_source: bool = True
    """Alerts require >= 2 independent sources confirming direction."""

    # ── Filters ───────────────────────────────────────────────────────────
    min_market_cap_usd: float = 0.0
    """Minimum market cap in USD. 0 = no filter.
    Bands: micro=50M, small=200M, mid=500M, large=1B, mega=10B."""

    max_market_cap_usd: float = 0.0
    """Maximum market cap in USD. 0 = no filter.
    Use to target small/mid cap only (e.g. max=2B)."""

    min_volume_relative: float = 0.0
    """Minimum relative volume vs 10-bar average. 0 = no filter.
    1.0 = at least average. 1.5 = 50% above average. 2.0 = double average."""

    min_atr_pct: float = 0.0
    """Minimum ATR% (14-period ATR / close). 0 = no filter.
    Filters out low-volatility symbols that tend to produce FLAT resolutions.
    0.3 for 1h, 0.5 for active pairs, 1.0+ for volatile pairs."""

    min_adx: float = 0.0
    """Minimum ADX for trending market confirmation. 0 = no filter.
    25 = weakly trending. 40 = strongly trending."""

    # ── Regime-aware direction bias ──────────────────────────────────────
    regime_dir_bear_short_bonus: float = 0.0
    """Bonus score added to SHORT signals when market regime is BEAR_TRENDING.
    In a bear market short trades naturally perform better. 2.0 = +2 score.
    0.0 = disabled."""

    regime_dir_bear_long_penalty: float = 0.0
    """Penalty subtracted from LONG signals when market regime is BEAR_TRENDING.
    In a bear market long trades are risky. 2.0 = -2 score.
    0.0 = disabled."""

    regime_dir_bull_long_bonus: float = 0.0
    """Bonus score added to LONG signals when market regime is BULL_TRENDING.
    0.0 = disabled."""

    regime_dir_bull_short_penalty: float = 0.0
    """Penalty subtracted from SHORT signals when market regime is BULL_TRENDING.
    0.0 = disabled."""

    min_rsi: float = 0.0
    """Minimum RSI14. 0 = no filter."""

    max_rsi: float = 0.0
    """Maximum RSI14. 0 = no filter. Use 70 to avoid overbought entries."""

    coin_type_filter: List[str] = field(default_factory=lambda: ["ANY"])
    """Coin type whitelist. ['ANY'] = no filter.
    Options: LAYER1, LAYER2, DEFI, MEME, AI, INFRA, GAMING, OTHER."""

    exclude_coin_types: List[str] = field(default_factory=list)
    """Coin types to always exclude (applied on top of coin_type_filter)."""

    require_multi_timeframe_alignment: bool = False
    """If True, require SHORT_TERM and MEDIUM_TERM trend to agree in direction.
    Eliminates counter-trend entries but reduces signal count significantly."""

    # ── Metadata ──────────────────────────────────────────────────────────
    display_types_extra: List[str] = field(default_factory=list)
    """Additional altFINS display_types to fetch for this config's scoring.
    These are fetched alongside the base fields in each scan cycle."""

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    status: str = "enabled"
    """Config lifecycle status:
       'active'   — Currently the ACTIVE_CONFIG (generates Telegram alerts)
       'enabled'  — Runs in scan, logs to DB, no Telegram alerts
       'disabled' — Not scanned, kept for historical records only
    """

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_enabled(self) -> bool:
        return self.status in ("active", "enabled")

    @property
    def is_disabled(self) -> bool:
        return self.status == "disabled"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScoringConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def get_required_display_types(self) -> List[str]:
        """Return the complete list of altFINS display_types needed for this config."""
        base = ["SHORT_TERM_TREND", "VOLUME_RELATIVE", "MARKET_CAP"]
        if self.adx_weight > 0 or self.min_adx > 0:
            base.append("ADX")
        if self.obv_trend_weight > 0:
            base.append("OBV_TREND")
        if self.medium_term_trend_weight > 0 or self.require_multi_timeframe_alignment:
            base.append("MEDIUM_TERM_TREND")
        if self.rsi_momentum_weight > 0 or self.min_rsi > 0 or self.max_rsi > 0:
            base.append("RSI14")
        if self.price_change_1w_weight > 0:
            base.append("PRICE_CHANGE_1W")
        if self.tr_vs_atr_weight > 0:
            base += ["TR_VS_ATR", "ATR"]
        # display_types_extra skipped here — use only known altFINS display types
        base += [dt for dt in self.display_types_extra
                 if dt not in ("VOLUME_DIVERGENCE", "SMART_MONEY_INDEX", "LOW_FLOAT_SQUEEZE",
                              "CAPITAL_GAINS", "FUNDING_RATE")]
        return list(dict.fromkeys(base))  # deduplicate, preserve order

    def passes_filters(self, symbol: str, screener_row: dict) -> tuple[bool, str]:
        """Check if a symbol passes this config's filters.
        Returns (passes: bool, reason: str). reason is '' if passes=True."""
        additional = screener_row.get("additionalData", {}) if screener_row else {}

        # Market cap filters
        if self.min_market_cap_usd > 0 or self.max_market_cap_usd > 0:
            mc = parse_market_cap(additional.get("MARKET_CAP", 0))
            if mc > 0:
                if self.min_market_cap_usd > 0 and mc < self.min_market_cap_usd:
                    return False, f"mcap=${mc/1e6:.0f}M < min=${self.min_market_cap_usd/1e6:.0f}M"
                if self.max_market_cap_usd > 0 and mc > self.max_market_cap_usd:
                    return False, f"mcap=${mc/1e6:.0f}M > max=${self.max_market_cap_usd/1e6:.0f}M"

        # Volume filter
        if self.min_volume_relative > 0:
            vol = parse_float(additional.get("VOLUME_RELATIVE", 0))
            if vol < self.min_volume_relative:
                return False, f"vol_rel={vol:.2f} < min={self.min_volume_relative}"

        # ADX filter
        if self.min_adx > 0:
            adx = parse_adx(additional.get("ADX", 0))
            if adx < self.min_adx:
                return False, f"ADX={adx:.1f} < min={self.min_adx}"

        # RSI filters
        rsi = parse_rsi(additional.get("RSI14", 50))
        if self.min_rsi > 0 and rsi < self.min_rsi:
            return False, f"RSI={rsi:.1f} < min={self.min_rsi}"
        if self.max_rsi > 0 and rsi > self.max_rsi:
            return False, f"RSI={rsi:.1f} > max={self.max_rsi}"

        # Multi-timeframe alignment
        if self.require_multi_timeframe_alignment:
            st = parse_trend_score_extended(additional.get("SHORT_TERM_TREND", ""))
            mt = parse_trend_score_extended(additional.get("MEDIUM_TERM_TREND", ""))
            if st != 0 and mt != 0 and (st > 0) != (mt > 0):
                return False, f"ST/MT trend misaligned (st={st:.0f} mt={mt:.0f})"

        # Coin type filters
        coin_type = get_coin_type(symbol)
        if "ANY" not in self.coin_type_filter and coin_type not in self.coin_type_filter:
            return False, f"coin_type={coin_type} not in {self.coin_type_filter}"
        if coin_type in self.exclude_coin_types:
            return False, f"coin_type={coin_type} excluded"

        return True, ""

    def compute_extended_score(self, additional: dict, base_score: float, direction_hint: float) -> tuple[float, dict]:
        """Compute extended signal contributions beyond the base 5 sources.

        Returns (extra_score, extra_components).
        direction_hint: positive = bullish context, negative = bearish.
        """
        extra = 0.0
        components: dict = {}

        # ADX momentum bonus — rewards strongly trending markets
        if self.adx_weight > 0:
            adx = parse_adx(additional.get("ADX", 0))
            if adx > 25:
                adx_contrib = min((adx - 25) / 75, 1.0) * self.adx_weight
                extra += adx_contrib if direction_hint >= 0 else -adx_contrib
            components["adx"] = adx

        # OBV trend — on-balance volume confirms price direction
        if self.obv_trend_weight > 0:
            obv_pct = parse_obv_trend(additional.get("OBV_TREND", 0))
            obv_contrib = max(-1.0, min(1.0, obv_pct / 100)) * self.obv_trend_weight
            extra += obv_contrib
            components["obv_trend_pct"] = obv_pct

        # Medium-term trend alignment bonus
        if self.medium_term_trend_weight > 0:
            mt = parse_trend_score_extended(additional.get("MEDIUM_TERM_TREND", ""))
            extra += mt * self.medium_term_trend_weight
            components["medium_term_trend"] = mt

        # RSI momentum zone (not overbought/oversold extremes)
        if self.rsi_momentum_weight > 0:
            rsi = parse_rsi(additional.get("RSI14", 50))
            if 45 <= rsi <= 70:
                # Bullish momentum zone — score proportional to how far above 50
                extra += ((rsi - 50) / 20) * self.rsi_momentum_weight
            elif 30 <= rsi < 45:
                # Bearish momentum zone
                extra -= ((50 - rsi) / 20) * self.rsi_momentum_weight
            # RSI < 30 or > 70 (extremes): no contribution — don't chase
            components["rsi14"] = rsi

        # 1-week price change momentum
        if self.price_change_1w_weight > 0:
            pct = parse_price_change_pct(additional.get("PRICE_CHANGE_1W", 0))
            contribution = max(-1.0, min(1.0, pct / 10)) * self.price_change_1w_weight
            extra += contribution
            components["price_change_1w_pct"] = pct

        # TR/ATR ratio — breakout detection
        if self.tr_vs_atr_weight > 0:
            ratio = parse_float(additional.get("TR_VS_ATR", 0))
            if ratio > 0:
                contrib = max(0.0, min(1.0, (ratio - 1) / 2)) * self.tr_vs_atr_weight  # assume max 3.0 for normalization
                extra += contrib
            components["tr_vs_atr"] = ratio

        return extra, components

# ---------------------------------------------------------------------------
# Configuration definitions
# ---------------------------------------------------------------------------

CONFIG_V1_0 = ScoringConfig(
    version="1.0",
    description="Baseline: Original scoring formula with equal weights",
    trend_weight=1.0,
    volume_relative_weight=1.0,
    signal_feed_weight=1.0,
    scanner_hit_weight=1.0,
    onchain_netflow_weight=1.0,
    # Alert thresholds
    alert_min_score=4.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=[],
)

CONFIG_V1_1 = ScoringConfig(
    version="1.1",
    description="Volume-weighted: Higher weight for volume relative (2.0)",
    trend_weight=1.0,
    volume_relative_weight=2.0,
    signal_feed_weight=1.0,
    scanner_hit_weight=1.0,
    onchain_netflow_weight=1.0,
    # Alert thresholds
    alert_min_score=4.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=[],
)

CONFIG_V1_2 = ScoringConfig(
    version="1.2",
    description="Signal-focused: Higher weight for signal feed (2.0)",
    trend_weight=1.0,
    volume_relative_weight=1.0,
    signal_feed_weight=2.0,
    scanner_hit_weight=1.0,
    onchain_netflow_weight=1.0,
    # Alert thresholds
    alert_min_score=4.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=[],
)

CONFIG_V1_3 = ScoringConfig(
    version="1.3",
    description="On-chain focused: Higher weight for on-chain netflow (2.0)",
    trend_weight=1.0,
    volume_relative_weight=1.0,
    signal_feed_weight=1.0,
    scanner_hit_weight=1.0,
    onchain_netflow_weight=2.0,
    # Alert thresholds
    alert_min_score=4.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=[],
)

CONFIG_V1_4 = ScoringConfig(
    version="1.4",
    description="Scanner-focused: Higher weight for scanner hits (2.0)",
    trend_weight=1.0,
    volume_relative_weight=1.0,
    signal_feed_weight=1.0,
    scanner_hit_weight=2.0,
    onchain_netflow_weight=1.0,
    # Alert thresholds
    alert_min_score=4.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=[],
)

# Multi-timeframe alignment configs
CONFIG_V2_0 = ScoringConfig(
    version="2.0",
    description="Multi-timeframe: Requires ST and MT trend alignment [DISABLED — replaced by V2.2]",
    status="disabled",
    trend_weight=0.4,
    volume_relative_weight=0.2,
    signal_feed_weight=0.3,
    scanner_hit_weight=0.2,
    onchain_netflow_weight=0.1,
    medium_term_trend_weight=0.3,
    require_multi_timeframe_alignment=True,
    # Alert thresholds
    alert_min_score=5.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=["MEDIUM_TERM_TREND"],
)

CONFIG_V2_1 = ScoringConfig(
    version="2.1",
    description="Multi-timeframe + Volume: MT alignment + higher volume weight",
    trend_weight=0.3,
    volume_relative_weight=0.3,
    signal_feed_weight=0.2,
    scanner_hit_weight=0.1,
    onchain_netflow_weight=0.1,
    medium_term_trend_weight=0.3,
    require_multi_timeframe_alignment=True,
    # Alert thresholds
    alert_min_score=5.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=["MEDIUM_TERM_TREND"],
)

# ── CONFIG_V2_2 — Soft Multi-Timeframe (bonus, not gate) ──
# Replaces V2.0: removes the binary `require_multi_timeframe_alignment` gate
# that caused 90.1% flat rate. MT alignment is still rewarded via
# medium_term_trend_weight but no longer required.
CONFIG_V2_2 = ScoringConfig(
    version="2.2",
    description="Soft MT Alignment: MT trend bonus (no binary gate) + volatility filter + regime bias",
    trend_weight=0.35,
    volume_relative_weight=0.2,
    signal_feed_weight=0.25,
    scanner_hit_weight=0.15,
    onchain_netflow_weight=0.05,
    medium_term_trend_weight=0.3,
    require_multi_timeframe_alignment=False,
    # Quality filters from V7.x
    min_atr_pct=0.2,
    min_adx=18,
    min_volume_relative=0.5,
    # Volume divergence for early entry
    volume_divergence_weight=2.0,
    # Regime-aware direction bias
    regime_dir_bear_short_bonus=2.0,
    regime_dir_bear_long_penalty=1.5,
    regime_dir_bull_long_bonus=2.0,
    regime_dir_bull_short_penalty=1.5,
    # Alert thresholds
    alert_min_score=5.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    display_types_extra=["MEDIUM_TERM_TREND", "VOLUME_DIVERGENCE"],
)

# ADX momentum configs
CONFIG_V3_0 = ScoringConfig(
    version="3.0",
    description="ADX filter: Requires ADX >= 25 for trending market [DISABLED — replaced by V3.2]",
    status="disabled",
    trend_weight=0.4,
    volume_relative_weight=0.2,
    signal_feed_weight=0.3,
    scanner_hit_weight=0.2,
    onchain_netflow_weight=0.1,
    adx_weight=0.3,
    min_adx=25,
    # Alert thresholds
    alert_min_score=5.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=["ADX"],
)

CONFIG_V3_1 = ScoringConfig(
    version="3.1",
    description="Strong ADX filter: Requires ADX >= 40 for strong trend",
    trend_weight=0.4,
    volume_relative_weight=0.2,
    signal_feed_weight=0.3,
    scanner_hit_weight=0.2,
    onchain_netflow_weight=0.1,
    adx_weight=0.3,
    min_adx=40,
    # Alert thresholds
    alert_min_score=5.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=["ADX"],
)

# ── CONFIG_V3_2 — Soft ADX Trend (bonus, not gate) ──
# Replaces V3.0: lowers min_adx from 25 -> 18 (soft filter, not hard gate).
# ADX above threshold still adds score via adx_weight but no longer blocks.
# Adds volume divergence confirmation + regime-aware direction bias.
CONFIG_V3_2 = ScoringConfig(
    version="3.2",
    description="Soft ADX Trend: ADX bonus (no hard gate) + volume divergence + regime bias",
    trend_weight=0.35,
    volume_relative_weight=0.2,
    signal_feed_weight=0.25,
    scanner_hit_weight=0.15,
    onchain_netflow_weight=0.05,
    adx_weight=0.3,
    min_adx=18,
    # Non-trend confirmation: ADX alone isn't enough
    require_non_trend_confirmation=True,
    # Quality filters from V7.x
    min_atr_pct=0.2,
    min_volume_relative=0.5,
    # Volume divergence for confirmation
    volume_divergence_weight=2.0,
    # Regime-aware direction bias
    regime_dir_bear_short_bonus=2.0,
    regime_dir_bear_long_penalty=1.5,
    regime_dir_bull_long_bonus=2.0,
    regime_dir_bull_short_penalty=1.5,
    # Alert thresholds
    alert_min_score=5.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    display_types_extra=["ADX", "VOLUME_DIVERGENCE"],
)

# Breakout intensity configs
CONFIG_V4_0 = ScoringConfig(
    version="4.0",
    description="TR/ATR breakout: Rewards high volatility expansion",
    trend_weight=0.4,
    volume_relative_weight=0.2,
    signal_feed_weight=0.3,
    scanner_hit_weight=0.2,
    onchain_netflow_weight=0.1,
    tr_vs_atr_weight=0.3,
    # Alert thresholds
    alert_min_score=5.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=["TR_VS_ATR", "ATR"],
)

CONFIG_V4_1 = ScoringConfig(
    version="4.1",
    description="TR/ATR + Volume: Breakout with volume confirmation",
    trend_weight=0.3,
    volume_relative_weight=0.3,
    signal_feed_weight=0.2,
    scanner_hit_weight=0.1,
    onchain_netflow_weight=0.1,
    tr_vs_atr_weight=0.3,
    # Alert thresholds
    alert_min_score=5.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=["TR_VS_ATR", "ATR"],
)

# Coin-type specific configs
CONFIG_V5_0 = ScoringConfig(
    version="5.0",
    description="DEFI-focused: Filters to DEFI coins only",
    trend_weight=0.4,
    volume_relative_weight=0.2,
    signal_feed_weight=0.3,
    scanner_hit_weight=0.2,
    onchain_netflow_weight=0.1,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["DEFI"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=[],
)

CONFIG_V5_1 = ScoringConfig(
    version="5.1",
    description="AI-focused: Filters to AI coins only",
    trend_weight=0.4,
    volume_relative_weight=0.2,
    signal_feed_weight=0.3,
    scanner_hit_weight=0.2,
    onchain_netflow_weight=0.1,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["AI"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=[],
)

CONFIG_V5_2 = ScoringConfig(
    version="5.2",
    description="Balanced DeFi/AI: 50/50 split between DEFI and AI",
    trend_weight=0.4,
    volume_relative_weight=0.2,
    signal_feed_weight=0.3,
    scanner_hit_weight=0.2,
    onchain_netflow_weight=0.1,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["DEFI", "AI"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=[],
)

# CEO suggested pattern configs
CONFIG_V6_0 = ScoringConfig(
    version="6.0",
    description="Pullback opportunity: Looks for pullbacks in uptrends",
    trend_weight=0.3,
    volume_relative_weight=0.2,
    signal_feed_weight=10.0,  # Heavy weight on signal feed
    scanner_hit_weight=0.1,
    onchain_netflow_weight=0.1,
    max_rsi=70.0,
    min_volume_relative=1.0,
    min_market_cap_usd=50_000_000,
    medium_term_trend_weight=0.3,
    # Alert thresholds
    alert_min_score=7.0,
    alert_require_multi_source=True,
    # Filters
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=["MEDIUM_TERM_TREND", "RSI14"],
)

CONFIG_V6_1 = ScoringConfig(
    version="6.1",
    description="Breakout momentum: Looks for breakouts with volume confirmation",
    trend_weight=0.3,
    volume_relative_weight=0.2,
    signal_feed_weight=0.3,
    scanner_hit_weight=10.0,  # Heavy weight on scanner hits
    onchain_netflow_weight=0.1,
    min_volume_relative=1.5,
    min_market_cap_usd=100_000_000,
    # Alert thresholds
    alert_min_score=7.0,
    alert_require_multi_source=True,
    # Filters
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=[],
)

# Quality Gate configs - NEW in v7.0
CONFIG_V7_0 = ScoringConfig(
    version="7.0",
    description="Filtered_Quality_Gate: Increased min_abs_score, required moderate trend (>=5), volume filter (>=0.5x), ADX>=20, RSI 30-70 to avoid extremes",
    status="active",
    trend_weight=0.4,
    volume_relative_weight=0.2,
    signal_feed_weight=0.3,
    scanner_hit_weight=0.2,
    onchain_netflow_weight=0.1,
    # Quality filters - NEW in v7.0
    min_abs_score=5.0,           # Increased from 3.0 - require stronger signal
    min_trend_abs_score=5,                 # Require at least moderate trend strength (5/10)
    require_non_trend_confirmation=True,   # Already True in v6.x but explicit
    min_volume_relative=0.5,     # Require at least 0.5x average volume
    min_adx=20,                  # Require ADX >= 20 for trend strength
    # RSI range for momentum - avoid extreme overbought/oversold
    min_rsi=30,
    max_rsi=70,
    # Volatility filter — reject low-vol setups to avoid FLAT resolutions
    min_atr_pct=0.3,
    # Regime-aware direction bias — SHORT favored in bear, LONG favored in bull
    regime_dir_bear_short_bonus=2.0,
    regime_dir_bear_long_penalty=2.0,
    regime_dir_bull_long_bonus=2.0,
    regime_dir_bull_short_penalty=2.0,
    # Alert thresholds
    alert_min_score=7.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    # Coin type filters (defaults)
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=[],
)

CONFIG_V7_2 = ScoringConfig(
    version="7.2",
    description="V7.0_Filtered_Quality_Gate + Volume Divergence Early Entry: Adds volume-price divergence detection as a leading indicator. Adds smart-money index and low-float squeeze detection.",
    trend_weight=0.4,
    volume_relative_weight=0.2,
    signal_feed_weight=0.3,
    onchain_netflow_weight=0.1,
    # Extended scoring signals (0.0 = disabled)
    volume_divergence_weight=3.0,
    smart_money_index_weight=2.0,
    low_float_squeeze_weight=1.5,
    # Risk management (target/stop computation)
    atr_stop_mult=1.5,
    rr_ratio=2.0,
    # Quality filters - NEW in v7.0
    min_abs_score=5.0,
    min_trend_abs_score=5,
    require_non_trend_confirmation=True,
    min_volume_relative=0.5,
    min_adx=20,
    # RSI range for momentum - avoid extreme overbought/oversold
    min_rsi=30,
    max_rsi=70,
    # Volatility filter — reject low-vol setups to avoid FLAT resolutions
    min_atr_pct=0.3,
    # Regime-aware direction bias — SHORT favored in bear, LONG favored in bull
    regime_dir_bear_short_bonus=2.0,
    regime_dir_bear_long_penalty=2.0,
    regime_dir_bull_long_bonus=2.0,
    regime_dir_bull_short_penalty=2.0,
    # Alert thresholds
    alert_min_score=7.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    # Coin type filters (defaults)
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=[],
)

CONFIG_V7_3 = ScoringConfig(
    version="7.3",
    description="Wide Stop, Wide Target: Higher ATR stop multiplier (2.0) and RR (2.5) for more room in volatile moves",
    trend_weight=0.4,
    volume_relative_weight=0.2,
    signal_feed_weight=0.3,
    scanner_hit_weight=0.2,
    onchain_netflow_weight=0.1,
    # Extended scoring signals (0.0 = disabled)
    volume_divergence_weight=0.0,
    smart_money_index_weight=0.0,
    low_float_squeeze_weight=0.0,
    # Risk management (target/stop computation) - WIDER STOP/TARGET
    atr_stop_mult=2.0,
    rr_ratio=2.5,
    # Quality filters - NEW in v7.0
    min_abs_score=5.0,
    min_trend_abs_score=5,
    require_non_trend_confirmation=True,
    min_volume_relative=0.5,
    min_adx=20,
    # RSI range for momentum - avoid extreme overbought/oversold
    min_rsi=30,
    max_rsi=70,
    # Volatility filter — reject low-vol setups to avoid FLAT resolutions
    min_atr_pct=0.3,
    # Regime-aware direction bias — SHORT favored in bear, LONG favored in bull
    regime_dir_bear_short_bonus=2.0,
    regime_dir_bear_long_penalty=2.0,
    regime_dir_bull_long_bonus=2.0,
    regime_dir_bull_short_penalty=2.0,
    # Alert thresholds
    alert_min_score=7.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    # Coin type filters (defaults)
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=[],
)

CONFIG_V7_4 = ScoringConfig(
    version="7.4",
    description="Wide Stop, Tight Target: Higher ATR stop multiplier (2.0) but tighter RR (1.5) for higher win rate, lower reward",
    trend_weight=0.4,
    volume_relative_weight=0.2,
    signal_feed_weight=0.3,
    scanner_hit_weight=0.2,
    onchain_netflow_weight=0.1,
    # Extended scoring signals (0.0 = disabled)
    volume_divergence_weight=0.0,
    smart_money_index_weight=0.0,
    low_float_squeeze_weight=0.0,
    # Risk management (target/stop computation) - WIDE STOP, TIGHT TARGET
    atr_stop_mult=2.0,
    rr_ratio=1.5,
    # Quality filters - NEW in v7.0
    min_abs_score=5.0,
    min_trend_abs_score=5,
    require_non_trend_confirmation=True,
    min_volume_relative=0.5,
    min_adx=20,
    # RSI range for momentum - avoid extreme overbought/oversold
    min_rsi=30,
    max_rsi=70,
    # Volatility filter — reject low-vol setups to avoid FLAT resolutions
    min_atr_pct=0.3,
    # Regime-aware direction bias — SHORT favored in bear, LONG favored in bull
    regime_dir_bear_short_bonus=2.0,
    regime_dir_bear_long_penalty=2.0,
    regime_dir_bull_long_bonus=2.0,
    regime_dir_bull_short_penalty=2.0,
    # Alert thresholds
    alert_min_score=7.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    # Coin type filters (defaults)
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=[],
)

CONFIG_V8_0 = ScoringConfig(
    version="8.0",
    description="Funding Rate Mean-Reversion: Uses extreme funding rates as primary entry signal. Requires |funding| > 0.006 + rising momentum + OI confirmation. Tighter stops and faster exits.",
    # Core weights — reduced reliance on altFINS, funding is primary
    trend_weight=0.3,
    volume_relative_weight=0.2,
    signal_feed_weight=0.2,
    scanner_hit_weight=0.1,
    onchain_netflow_weight=0.1,
    # Extended scoring signals (0.0 = disabled)
    volume_divergence_weight=0.0,
    smart_money_index_weight=0.0,
    low_float_squeeze_weight=0.0,
    # Funding Rate Mean-Reversion — PRIMARY signal source
    funding_rate_weight=5.0,
    funding_momentum_weight=5.0,  # Increased from 3.0 per paper: momentum explains 50%+ of gap
    oi_change_weight=2.0,
    min_abs_funding_rate=0.006,
    min_funding_momentum=0.0,
    max_oi_change=0.02,
    # Funding interval & pre-funding dip (added Jun 2026)
    funding_interval_weight=2.0,
    pre_funding_dip_weight=3.0,
    interval_switch_weight=5.0,
    # Risk management — tighter, faster for funding-driven moves
    atr_stop_mult=1.0,
    rr_ratio=1.5,
    # Quality filters — lighter, funding rate is the confirmation
    min_abs_score=5.0,
    min_trend_abs_score=3,
    require_non_trend_confirmation=False,
    min_volume_relative=0.0,
    min_adx=0,
    # RSI range — no filter (funding works in all regimes)
    min_rsi=0,
    max_rsi=0,
    # Volatility filter — no ATR filter (funding can work in low vol)
    min_atr_pct=0.0,
    # Regime-aware direction bias
    regime_dir_bear_short_bonus=2.0,
    regime_dir_bear_long_penalty=2.0,
    regime_dir_bull_long_bonus=2.0,
    regime_dir_bull_short_penalty=2.0,
    # Alert thresholds
    alert_min_score=7.0,
    alert_require_multi_source=False,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    # Metadata
    display_types_extra=[],
)

# Active config — change via CLI: python -m src.cli.main edge activate-config --version v1.1
# All signals logged will carry this version in the config_version column.

# ── CONFIG_V7_5 — Auto-generated 2026-07-01 19:47 ──
CONFIG_V7_5 = ScoringConfig(
    version="7.5",
    description="LLM-generated: relax V7.0 filters for more signals. Lower min_abs_score (6.5), wider RSI, lower min_atr_pct, increased vol-div weight.",
    min_abs_score=6.5,
    min_adx=18,
    min_rsi=25,
    max_rsi=75,
    min_atr_pct=0.2,
    atr_stop_mult=1.5,
    rr_ratio=2.0,
    trend_weight=0.4,
    volume_relative_weight=0.2,
    signal_feed_weight=0.35,
    scanner_hit_weight=0.2,
    onchain_netflow_weight=0.1,
    # Extended scoring
    volume_divergence_weight=3.5,
    smart_money_index_weight=2.0,
    low_float_squeeze_weight=1.5,
    # Regime bias
    regime_dir_bear_short_bonus=2.0,
    regime_dir_bear_long_penalty=2.0,
    regime_dir_bull_long_bonus=2.0,
    regime_dir_bull_short_penalty=2.0,
    # Alert thresholds
    alert_min_score=7.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    display_types_extra=[],
)



# ── CONFIG_V7_6 — Auto-generated 2026-07-08 19:08 ──
CONFIG_V7_6 = ScoringConfig(
    version="7.6",
    description="LLM-evolved: relaxed ADX/ATR/RSI filters, enabled multi-factor weights, boosted trend weight to 0.5",
    min_abs_score=6.5,
    min_adx=18,
    min_rsi=28,
    max_rsi=72,
    min_atr_pct=0.2,
    atr_stop_mult=1.5,
    rr_ratio=2.0,
    trend_weight=0.35,
    volume_relative_weight=0.25,
    signal_feed_weight=0.3,
    onchain_netflow_weight=0.1,
    volume_divergence_weight=3.0,
    smart_money_index_weight=2.0,
    low_float_squeeze_weight=1.5,
    regime_dir_bear_short_bonus=2.0,
    regime_dir_bear_long_penalty=2.0,
    regime_dir_bull_long_bonus=2.0,
    regime_dir_bull_short_penalty=2.0,
)



# ── CONFIG_V7_7 — Auto-generated 2026-07-12 14:00 ──
CONFIG_V7_7 = ScoringConfig(
    version="7.7",
    description="LLM-evolved: relaxed ADX/ATR/RSI filters, enabled multi-factor weights, boosted trend weight to 0.5",
    min_abs_score=6.5,
    min_adx=18,
    min_rsi=25,
    max_rsi=75,
    min_atr_pct=0.25,
    atr_stop_mult=1.5,
    rr_ratio=2.0,
    trend_weight=0.35,
    volume_relative_weight=0.25,
    signal_feed_weight=0.3,
    onchain_netflow_weight=0.1,
    volume_divergence_weight=3.0,
    smart_money_index_weight=2.0,
    low_float_squeeze_weight=1.5,
    regime_dir_bear_short_bonus=2.0,
    regime_dir_bear_long_penalty=2.0,
    regime_dir_bull_long_bonus=2.0,
    regime_dir_bull_short_penalty=2.0,
)

ACTIVE_CONFIG = CONFIG_V7_0

ALL_CONFIGS: dict[str, ScoringConfig] = {
    c.version: c for c in [
        # Baseline variants
        CONFIG_V1_0, CONFIG_V1_1, CONFIG_V1_2, CONFIG_V1_3, CONFIG_V1_4,
        # Multi-timeframe (all kept for records)
        CONFIG_V2_0, CONFIG_V2_1, CONFIG_V2_2,
        # ADX momentum (all kept for records)
        CONFIG_V3_0, CONFIG_V3_1, CONFIG_V3_2,
        # Breakout intensity
        CONFIG_V4_0, CONFIG_V4_1,
        # Coin-type specific
        CONFIG_V5_0, CONFIG_V5_1, CONFIG_V5_2,
        # CEO suggested patterns
        CONFIG_V6_0, CONFIG_V6_1,
        # Quality Gate (LLM-evolved series)
        CONFIG_V7_0, CONFIG_V7_2, CONFIG_V7_3, CONFIG_V7_4, CONFIG_V7_5, CONFIG_V7_6, CONFIG_V7_7,
        # Funding Rate Mean-Reversion
        CONFIG_V8_0,
    ]
}


def get_enabled_configs() -> dict[str, ScoringConfig]:
    """Return only configs that should actively run in scans (excludes disabled)."""
    return {v: c for v, c in ALL_CONFIGS.items() if c.is_enabled}


def get_disabled_configs() -> dict[str, ScoringConfig]:
    """Return only disabled configs (kept for historical records)."""
    return {v: c for v, c in ALL_CONFIGS.items() if c.is_disabled}


def get_active_config() -> ScoringConfig:
    """Return the currently active config (generates Telegram alerts)."""
    for c in ALL_CONFIGS.values():
        if c.is_active:
            return c
    return CONFIG_V7_0  # fallback