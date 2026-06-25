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
    "GALA": "GAMING", "PIXEL": "GAMING", "PRIME": "GAMING",
    "MAGIC": "GAMING", "YGG": "GAMING", "PYR": "GAMING", "ILV": "GAMING",
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
    Set to 5.0 to require at least \"Up (5/10)\" for LONG or \"Down (5/10)\" for SHORT."""

    require_non_trend_confirmation: bool = False
    """If True, at least one non-trend source must confirm the direction:
    volume_relative > 1.0 OR signal feed match OR scanner hit OR on-chain match.
    Prevents signals based on trend alone."""

    alert_min_score: float = 7.0
    """Minimum |composite_score| to send a Telegram alert."""

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

    min_rsi: float = 0.0
    """Minimum RSI14. 0 = no filter. Use 45+ to require bullish momentum."""

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
        base += self.display_types_extra
        return list(dict.fromkeys(base))  # deduplicate, preserve order

    def passes_filters(self, symbol: str, screener_row: dict) -> tuple[bool, str]:
        """Check if a symbol passes this config's filters.
        Returns (passes: bool, reason: str). reason is '' if passes=True.
        """
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
            # Normalize: ±10% weekly change → ±1.0 contribution
            contrib = max(-1.0, min(1.0, pct / 10)) * self.price_change_1w_weight
            extra += contrib
            components["price_change_1w_pct"] = pct

        # TR/ATR breakout intensity
        if self.tr_vs_atr_weight > 0:
            tr_vs_atr = parse_float(additional.get("TR_VS_ATR", 1.0), 1.0)
            if tr_vs_atr > 1.0:
                contrib = min(tr_vs_atr - 1.0, 2.0) * self.tr_vs_atr_weight
                extra += contrib if direction_hint >= 0 else -contrib
            components["tr_vs_atr"] = tr_vs_atr

        return extra, components


# ---------------------------------------------------------------------------
# Scoring Config Library
#
# DESIGN PHILOSOPHY:
# Each version tests ONE clear hypothesis about what predicts positive
# forward returns in crypto swing trading. The description states:
#   1. THE THEORY — what market behaviour it believes in
#   2. THE CRITERIA — exact combination of filters + weights used
#   3. THE PREDICTION — what kind of signals it produces vs v1.0
#
# Versions are grouped by major theme:
#   v1.x — Baseline variants (filter / weight changes on base formula)
#   v2.x — Multi-timeframe alignment (adds MEDIUM_TERM_TREND as filter/signal)
#   v3.x — Momentum-quality (RSI + ADX confirm genuine momentum)
#   v4.x — Volume-breakout intensity (TR/ATR + OBV as primary signals)
#   v5.x — Coin-type-specific (custom weights tuned per asset class)
# ---------------------------------------------------------------------------

# ── v1.x — Baseline variants ─────────────────────────────────────────────

CONFIG_V1_0 = ScoringConfig(
    version="v1.0",
    description=(
        "BASELINE — Broad universe, standard weights, stablecoins/stocks always excluded. "
        "Theory: the raw combination of altFINS trend score + unusual volume + signal feed "
        "confirmation + on-chain exchange flow has positive expectancy across any liquid crypto. "
        "No market cap, RSI, ADX or coin type restrictions — maximum signal count. "
        "Serves as the performance benchmark every other version is measured against. "
        "Criteria: SHORT_TERM_TREND×0.4 + vol_excess×2.0 (cap 3x) + signal_feed±3.0 "
        "+ onchain_netflow×2.0 + scanner_hits×2.5. Threshold: |score| >= 3.0."
    ),
)

CONFIG_V1_1 = ScoringConfig(
    version="v1.1",
    description=(
        "VOLUME GATE — Adds minimum relative volume >= 1.2x. "
        "Theory: signals firing on below-average volume are prone to wash trading and "
        "illiquid manipulation, especially on small-cap altcoins. Requiring 20% above-average "
        "volume ensures genuine market participation is behind the move — not just a "
        "single whale or bot farm. Expected to produce ~25% fewer signals than v1.0 "
        "but with cleaner entries on real liquidity. "
        "Criteria: v1.0 weights + min_volume_relative=1.2. No coin type restriction."
    ),
    min_volume_relative=1.2,
)

CONFIG_V1_2 = ScoringConfig(
    version="v1.2",
    description=(
        "LARGE CAP QUALITY FILTER — Market cap $500M+, excludes MEME coins. "
        "Theory: large-cap assets ($500M+) have institutional presence, tighter bid-ask "
        "spreads, and more reliable technical patterns because they are less susceptible "
        "to social-media pump cycles. Meme coins (DOGE, SHIB, PEPE, FLOKI etc.) have "
        "volatility driven by influencers and Reddit not fundamentals, making trend "
        "signals unreliable for systematic following. "
        "Expected: conservative, institutional-quality signals only (~30-40% of v1.0 count). "
        "Criteria: v1.0 weights + min_mcap=$500M + exclude=[MEME]."
    ),
    min_market_cap_usd=500_000_000,
    exclude_coin_types=["MEME"],
)

CONFIG_V1_3 = ScoringConfig(
    version="v1.3",
    description=(
        "SMALL/MID CAP MOMENTUM — Market cap $20M-$500M, high relative volume >= 1.5x. "
        "Theory: the highest percentage gains in crypto come from smaller coins that "
        "institutional players haven't yet discovered. A strong trend signal + unusually "
        "high volume on a small-cap coin is often the early sign of a genuine breakout "
        "before it reaches mainstream attention. High risk, high reward profile. "
        "The volume gate (1.5x) is critical here to filter manipulation on thin order books. "
        "Expected: fewer signals than v1.0 but targets the highest upside opportunities. "
        "Criteria: v1.0 weights + mcap=$20M-$500M + min_vol=1.5x."
    ),
    min_market_cap_usd=20_000_000,
    max_market_cap_usd=500_000_000,
    min_volume_relative=1.5,
)

CONFIG_V1_4 = ScoringConfig(
    version="v1.4",
    description=(
        "ANALYST SIGNAL DOMINANT — Boosts altFINS signal feed weight from 3.0 to 4.5, "
        "reduces trend weight from 0.4 to 0.25. "
        "Theory: the altFINS signal feed is the output of professional analyst review "
        "(equivalent to their VIP Telegram channel) and is inherently forward-looking — "
        "analysts anticipate moves before the lagging trend score catches up. "
        "The screener's SHORT_TERM_TREND is a 10-period look-back average, making it "
        "a lagging indicator. Reducing its weight and boosting analyst signals tests "
        "whether human curation outperforms pure quant momentum. "
        "Isolates signal feed quality as the sole variable — no additional filters. "
        "Criteria: signal_feed_weight=4.5, trend_weight=0.25, all other weights unchanged."
    ),
    signal_feed_weight=4.5,
    trend_weight=0.25,
)

# ── v2.x — Multi-timeframe alignment ─────────────────────────────────────

CONFIG_V2_0 = ScoringConfig(
    version="v2.0",
    description=(
        "MULTI-TIMEFRAME TREND ALIGNMENT — Requires SHORT_TERM and MEDIUM_TERM trend "
        "to agree in direction. Adds medium-term trend as a scoring signal (weight=0.2). "
        "Theory: the highest-probability trend trades occur when multiple timeframes "
        "are aligned — short-term momentum in the direction of the medium-term trend "
        "has a much higher continuation rate than counter-trend bounces. "
        "Eliminating conflicting timeframes removes a large class of false signals "
        "(short-term spike against medium-term downtrend). "
        "Expected: ~40% fewer signals vs v1.0 but with significantly higher "
        "directional conviction per signal. "
        "Criteria: require_multi_timeframe_alignment=True + medium_term_trend_weight=0.2 "
        "+ v1.0 base weights."
    ),
    require_multi_timeframe_alignment=True,
    medium_term_trend_weight=0.2,
    display_types_extra=["MEDIUM_TERM_TREND"],
)

CONFIG_V2_1 = ScoringConfig(
    version="v2.1",
    description=(
        "MULTI-TIMEFRAME + VOLUME + LARGE CAP — Combines multi-timeframe alignment, "
        "volume gate (>=1.2x), and large cap filter ($300M+). "
        "Theory: the combination of timeframe alignment + real volume + institutional-grade "
        "market cap creates a triple-quality gate that should produce the most reliable "
        "signals overall. Each gate eliminates a different category of noise: "
        "timeframe alignment removes counter-trend entries, volume removes thin-liquidity "
        "manipulation, and market cap removes micro-cap pump schemes. "
        "Expected: fewest signals of any v2.x config, but highest per-signal quality. "
        "Designed as a 'trade this live' conservative filter. "
        "Criteria: MTF alignment + min_vol=1.2x + min_mcap=$300M + medium_term_weight=0.2."
    ),
    require_multi_timeframe_alignment=True,
    medium_term_trend_weight=0.2,
    min_volume_relative=1.2,
    min_market_cap_usd=300_000_000,
    display_types_extra=["MEDIUM_TERM_TREND"],
)

# ── v3.x — Momentum quality (RSI + ADX confirmation) ─────────────────────

CONFIG_V3_0 = ScoringConfig(
    version="v3.0",
    description=(
        "MOMENTUM QUALITY GATE — Requires ADX >= 25 (confirms trending market), "
        "RSI between 45-72 (momentum zone, not overbought). Adds ADX and RSI "
        "as scoring signals. "
        "Theory: the two biggest failure modes of trend-following are: "
        "(1) entering in a ranging/sideways market (ADX < 25 means no trend), "
        "(2) entering at overbought extremes (RSI > 72) where mean-reversion "
        "is more likely than continuation. ADX >= 25 ensures we only trade "
        "genuine trends. RSI 45-72 is the 'sweet spot' — momentum confirmed "
        "but not exhausted. "
        "Expected: ~35% fewer signals but much cleaner trend quality per signal. "
        "Criteria: min_ADX=25 + RSI 45-72 gate + adx_weight=0.5 + rsi_momentum_weight=0.3."
    ),
    min_adx=25.0,
    min_rsi=45.0,
    max_rsi=72.0,
    adx_weight=0.5,
    rsi_momentum_weight=0.3,
    display_types_extra=["ADX", "RSI14"],
)

CONFIG_V3_1 = ScoringConfig(
    version="v3.1",
    description=(
        "STRONG TREND ONLY — ADX >= 40 (strong trending market requirement), "
        "volume >= 1.2x, market cap >= $100M. "
        "Theory: ADX above 40 represents a strongly trending market where momentum "
        "continuation is statistically more likely than in weak trends (ADX 25-40). "
        "Strong trends also have lower whipsaw risk and larger average moves per signal, "
        "making them ideal for swing trading with 24-72h horizons. "
        "The volume and market cap gates prevent acting on strong-trend signals "
        "in illiquid micro-caps where ADX can be artificially high. "
        "Expected: fewest signals of all v3.x but highest trend strength per signal. "
        "Criteria: min_ADX=40 + min_vol=1.2x + min_mcap=$100M + adx_weight=0.8."
    ),
    min_adx=40.0,
    min_volume_relative=1.2,
    min_market_cap_usd=100_000_000,
    adx_weight=0.8,
    display_types_extra=["ADX"],
)

# ── v4.x — Volume-breakout intensity (TR/ATR + OBV) ──────────────────────

CONFIG_V4_0 = ScoringConfig(
    version="v4.0",
    description=(
        "BREAKOUT INTENSITY — Adds TR/ATR ratio (current range vs average range) "
        "and OBV trend as primary scoring signals. "
        "Theory: the most powerful breakouts are characterised by TWO simultaneous "
        "confirmations: (1) current candle range exceeds average range (TR/ATR > 1.5 "
        "means price is moving more than usual = high-energy breakout), "
        "(2) On-Balance Volume is rising (OBV trend > 0 means volume is confirming "
        "the price move, not diverging). When both fire together, the move has "
        "both price momentum AND volume conviction behind it. "
        "Expected: roughly same signal count as v1.0 but higher average move per signal. "
        "Criteria: tr_vs_atr_weight=1.0 + obv_trend_weight=0.5 + v1.0 base weights."
    ),
    tr_vs_atr_weight=1.0,
    obv_trend_weight=0.5,
    display_types_extra=["TR_VS_ATR", "ATR", "OBV_TREND"],
)

CONFIG_V4_1 = ScoringConfig(
    version="v4.1",
    description=(
        "BREAKOUT INTENSITY + PRICE MOMENTUM — Adds 1-week price change as a "
        "momentum confirmation signal, combined with TR/ATR breakout detection. "
        "Theory: breakouts that occur after a strong recent week (price_change_1w > +5%) "
        "have established prior momentum, reducing the chance that the current breakout "
        "is a false-start reversal spike. Conversely, breakouts from oversold after a "
        "bad week can be mean-reversion traps. Adding weekly price momentum as a signal "
        "tests whether recent price context improves breakout signal quality. "
        "Also adds volume minimum (1.2x) to ensure breakouts have genuine participation. "
        "Criteria: tr_vs_atr_weight=0.8 + price_change_1w_weight=0.4 + "
        "obv_trend_weight=0.3 + min_vol=1.2x."
    ),
    tr_vs_atr_weight=0.8,
    obv_trend_weight=0.3,
    price_change_1w_weight=0.4,
    min_volume_relative=1.2,
    display_types_extra=["TR_VS_ATR", "ATR", "OBV_TREND", "PRICE_CHANGE_1W"],
)

# ── v5.x — Coin-type-specific configs ────────────────────────────────────

CONFIG_V5_0 = ScoringConfig(
    version="v5.0",
    description=(
        "DEFI + L1/L2 ECOSYSTEM — Universe restricted to LAYER1, LAYER2, DEFI. "
        "Higher on-chain weight (3.0 vs 2.0) since DeFi/L1 protocols have "
        "meaningful on-chain flow data. Adds multi-timeframe alignment. "
        "Theory: DeFi protocols and L1/L2 chains have real revenue, TVL, and "
        "on-chain activity that drives price — unlike meme/gaming tokens driven "
        "purely by narrative. Exchange outflow (coins leaving exchanges = accumulation) "
        "is especially meaningful for these assets because long-term holders actively "
        "move them to DeFi protocols. MTF alignment ensures we follow the medium-term "
        "ecosystem trend, not just short-term noise. "
        "Criteria: LAYER1+LAYER2+DEFI only + onchain_weight=3.0 + MTF alignment "
        "+ medium_term_weight=0.2 + min_vol=1.0x."
    ),
    coin_type_filter=["LAYER1", "LAYER2", "DEFI"],
    onchain_netflow_weight=3.0,
    medium_term_trend_weight=0.2,
    require_multi_timeframe_alignment=True,
    min_volume_relative=1.0,
    display_types_extra=["MEDIUM_TERM_TREND"],
)

CONFIG_V5_1 = ScoringConfig(
    version="v5.1",
    description=(
        "AI + INFRASTRUCTURE SECTOR — Universe restricted to AI and INFRA coin types. "
        "Requires volume >= 1.5x and market cap >= $50M. "
        "Theory: AI and infrastructure tokens (FET, RENDER, TAO, LINK, GRT, PYTH, OCEAN) "
        "are driven by a multi-year structural narrative (AI adoption, oracle demand, "
        "data monetization). Their trend signals have a stronger fundamental backing "
        "than pure speculative tokens. Requiring 1.5x volume ensures only genuine "
        "breakouts fire — AI tokens can have sudden spikes on news that fade quickly "
        "without follow-through volume. The market cap floor ($50M) removes micro-projects "
        "with single-digit liquidity. "
        "Expected: fewest signals of all v5.x — high conviction, long-horizon plays. "
        "Criteria: AI+INFRA only + min_vol=1.5x + min_mcap=$50M + signal_feed_weight=3.5."
    ),
    coin_type_filter=["AI", "INFRA"],
    min_volume_relative=1.5,
    min_market_cap_usd=50_000_000,
    signal_feed_weight=3.5,
)

CONFIG_V5_2 = ScoringConfig(
    version="v5.2",
    description=(
        "BALANCED DEPLOY-READY — Designed as a practical live-trading filter. "
        "Combines: large cap ($200M+) + volume gate (1.2x) + no MEME + ADX >= 20 "
        "(at least weakly trending) + RSI cap at 75 (not overbought) + "
        "higher scanner weight (3.0) to reward TA-confirmed breakouts. "
        "Theory: for actual signal following in live trading, you want signals that "
        "pass multiple independent quality gates simultaneously — not just one strong "
        "signal. This config requires: (1) real market cap behind the asset, "
        "(2) real volume participation, (3) no meme coin narrative risk, "
        "(4) a trending market (not sideways), (5) not overbought, "
        "(6) ideally TA-scanner confirmed on real Binance OHLCV data. "
        "The scanner weight increase rewards the one signal that uses our own data, "
        "not third-party APIs. This is the config most suitable for live signal alerts. "
        "Criteria: min_mcap=$200M + min_vol=1.2x + exclude=MEME + min_ADX=20 "
        "+ max_RSI=75 + scanner_weight=3.0."
    ),
    min_market_cap_usd=200_000_000,
    min_volume_relative=1.2,
    exclude_coin_types=["MEME"],
    min_adx=20.0,
    scanner_hit_weight=3.0,
    display_types_extra=["ADX", "RSI14"],
)


# ── v6.x — CEO altFINS suggested patterns ──────────────────────────────
# Three dedicated versions, each isolating ONE of the CEO's suggestions:
#   v6.0: Uptrend + Pullback (buy the dip in a trend)
#   v6.1: Resistance Breakout (breakout from resistance levels)
#   v6.2: Bullish MACD Crossover (momentum shift confirmation)
#
# Each version:
#   - Only fires signals when the signal_feed matches its specific pattern
#   - Uses the signal_feed_weight as the primary signal (maximized to 10.0)
#   - Reduces other weights so only that pattern dominates
#   - Includes volume + cap quality gates to filter fakeouts

CONFIG_V6_0 = ScoringConfig(
    version="v6.0",
    description=(
        "CEO PATTERN 1 — UPTREND PULLBACK. Detects assets in an established "
        "uptrend that have pulled back to support (buy-the-dip in a trend). "
        "Theory: the highest-probability swing trades are entries in the direction "
        "of the dominant trend after a temporary counter-trend move — buying "
        "the dip in a bull trend. The altFINS 'PULLBACK_UP_DOWN_TREND' signal "
        "type flags exactly this pattern. "
        "Maximizes signal_feed_weight (10.0) so only pullback signals dominate. "
        "Criteria: signal_feed_weight=10.0 + min_vol=1.0x + min_mcap=$100M "
        "to ensure quality pullbacks in liquid markets."
    ),
    signal_feed_weight=10.0,
    trend_weight=0.1,
    volume_relative_weight=0.5,
    min_volume_relative=1.0,
    min_market_cap_usd=100_000_000,
    exclude_coin_types=["MEME"],
)

CONFIG_V6_1 = ScoringConfig(
    version="v6.1",
    description=(
        "CEO PATTERN 2 — RESISTANCE BREAKOUT. Detects assets breaking through "
        "key resistance levels with volume confirmation. "
        "Theory: breakouts above resistance with above-average volume have the "
        "highest continuation rates in crypto — the breakout creates a new "
        "support level and attracts momentum traders. "
        "The altFINS 'SUPPORT_RESISTANCE_BREAKOUT' signal type flags exactly this. "
        "Adds scanner_hit_weight bonus for TA-confirmed breakouts on real OHLCV. "
        "Criteria: signal_feed_weight=10.0 + scanner_hit_weight=4.0 + "
        "min_vol=1.2x + min_mcap=$200M + no MEME."
    ),
    signal_feed_weight=10.0,
    scanner_hit_weight=4.0,
    trend_weight=0.1,
    volume_relative_weight=0.5,
    min_volume_relative=1.2,
    min_market_cap_usd=200_000_000,
    exclude_coin_types=["MEME"],
)

CONFIG_V6_2 = ScoringConfig(
    version="v6.2",
    description=(
        "CEO PATTERN 3 — BULLISH MACD CROSSOVER. Detects assets where the "
        "MACD line has just crossed above the signal line, indicating fresh "
        "bullish momentum. "
        "Theory: the MACD crossover is one of the most widely tracked technical "
        "signals — a fresh bullish cross at the start of an uptrend has "
        "statistically significant continuation probability. "
        "The altFINS 'FRESH_MOMENTUM_MACD_SIGNAL_LINE_CROSSOVER' signal type "
        "flags exactly this. RSI is capped at 70 to avoid catching overbought "
        "crosses that already exhausted. "
        "Criteria: signal_feed_weight=10.0 + max_rsi=70 + "
        "min_vol=1.0x + medium_term_trend_weight=0.3 to confirm MTF alignment."
    ),
    signal_feed_weight=10.0,
    trend_weight=0.1,
    medium_term_trend_weight=0.3,
    max_rsi=70.0,
    min_volume_relative=1.0,
    min_market_cap_usd=50_000_000,
    display_types_extra=["MEDIUM_TERM_TREND", "RSI14"],
)
CONFIG_V7_0 = ScoringConfig(
    version="7.0",
    description=(
        "Filtered_Quality_Gate: "
        "Increased min_abs_score to 5.0, required moderate trend (>=5), "
        "volume filter (>=0.5x), ADX>=20, RSI 30-70 to avoid extremes. "
        "Core weights same as v6.x."
    ),
    # Core weights
    trend_weight=0.4,
    volume_relative_weight=0.2,
    signal_feed_weight=0.3,
    onchain_netflow_weight=0.1,
    # Extended scoring signals (0.0 = disabled)
    adx_weight=0.1,
    obv_trend_weight=0.05,
    medium_term_trend_weight=0.1,
    rsi_momentum_weight=0.05,
    price_change_1w_weight=0.0,
    tr_vs_atr_weight=0.0,
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
    # Alert thresholds
    alert_min_score=7.0,
    alert_require_multi_source=True,
    # Filters
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    # Coin type filters (defaults)
    coin_type_filter=["ANY"],    # No filter on coin type
    exclude_coin_types=[],       # No additional exclusions
    # Metadata
    display_types_extra=[],
)



# ---------------------------------------------------------------------------
# Active config + registry
# ---------------------------------------------------------------------------

# Active config — change via CLI: python -m src.cli.main edge activate-config --version v1.1
# All signals logged will carry this version in the config_version column.
ACTIVE_CONFIG = CONFIG_V7_0

ALL_CONFIGS: dict[str, ScoringConfig] = {
    c.version: c for c in [
        # Baseline variants
        CONFIG_V1_0, CONFIG_V1_1, CONFIG_V1_2, CONFIG_V1_3, CONFIG_V1_4,
        # Multi-timeframe
        CONFIG_V2_0, CONFIG_V2_1,
        # Momentum quality
        CONFIG_V3_0, CONFIG_V3_1,
        # Breakout intensity
        CONFIG_V4_0, CONFIG_V4_1,
        # Coin-type specific
        CONFIG_V5_0, CONFIG_V5_1, CONFIG_V5_2,
        # CEO suggested patterns
        CONFIG_V6_0, CONFIG_V6_1, CONFIG_V6_2,
        CONFIG_V7_0,
    ]
}
