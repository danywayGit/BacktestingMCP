"""
Spot Gem Scanner — finds undervalued coins on Binance with strong tokenomics.

Scores coins on a multi-factor model:
  - Market cap (small-mid = room to grow)
  - Volume / MCap ratio (liquidity check)
  - Distance from ATH (reversion potential)
  - Supply dynamics (not too inflationary)
  - Binance listing (must-have)
  - Holder concentration proxy (FDV/MCap ratio)
  - Recent price action (not already pumping)
"""

import httpx
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# ── Scoring weights ────────────────────────────────────────────────────────

WEIGHTS = {
    "market_cap": 0.15,         # Small-mid cap = more room to grow
    "vol_mcap_ratio": 0.15,     # Liquidity relative to size
    "distance_from_ath": 0.15,  # Down from ATH = reversion potential
    "supply_dilution": 0.10,    # Low FDV/MCap = less future dilution
    "circulating_ratio": 0.10,  # 20-80% circulating = sweet spot
    "binance_futures": 0.10,    # Listed on Binance Futures = bonus
    "price_momentum": 0.10,     # Slight negative or neutral = entry opportunity
    "coin_age": 0.15,           # Under 2 years = room to grow
}

# Scoring thresholds
MCAP_MIN = 10_000_000       # $10M minimum
MCAP_MAX = 300_000_000      # $300M maximum (sweet spot for gems)
VOL_MCAP_MIN = 0.03         # 3% minimum volume/mcap ratio
ATH_DROP_MIN = 30           # At least 30% down from ATH
FDV_MCAP_MAX = 10.0         # Max 10x FDV/MCap (dilution cap)
CIRCULATING_MIN = 0.10      # At least 10% circulating
CIRCULATING_MAX = 0.90      # At most 90% circulating


@dataclass
class GemCandidate:
    symbol: str
    name: str
    price: float
    market_cap: float
    volume_24h: float
    circulating_supply: float
    max_supply: Optional[float]
    total_supply: Optional[float]
    fdv: Optional[float]
    ath: float
    ath_change_pct: float
    price_change_7d: Optional[float]
    price_change_30d: Optional[float]
    price_change_1y: Optional[float]
    market_cap_rank: Optional[int]
    on_binance_futures: bool
    coin_gecko_id: str
    atl_date: Optional[str] = None        # CoinGecko ATL date — proxy for coin age
    score: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)


def _score_gem(c: GemCandidate) -> GemCandidate:
    """Score a gem candidate on all factors. Returns updated candidate."""
    score = 0.0
    breakdown = {}

    # 1. Market cap score (inverse: smaller = better)
    mcap = c.market_cap
    if MCAP_MIN <= mcap <= 50_000_000:
        mcap_score = 1.0  # Sweet spot
    elif mcap <= 150_000_000:
        mcap_score = 0.7
    elif mcap <= MCAP_MAX:
        mcap_score = 0.4
    else:
        mcap_score = 0.0
    score += mcap_score * WEIGHTS["market_cap"]
    breakdown["market_cap"] = round(mcap_score * WEIGHTS["market_cap"], 3)

    # 2. Volume/MCap ratio (higher = better liquidity)
    vol_mcap = c.volume_24h / mcap if mcap > 0 else 0
    if vol_mcap >= 0.20:
        vol_score = 1.0
    elif vol_mcap >= 0.10:
        vol_score = 0.8
    elif vol_mcap >= 0.05:
        vol_score = 0.5
    elif vol_mcap >= 0.03:
        vol_score = 0.3
    else:
        vol_score = 0.0
    score += vol_score * WEIGHTS["vol_mcap_ratio"]
    breakdown["vol_mcap"] = round(vol_score * WEIGHTS["vol_mcap_ratio"], 3)

    # 3. Distance from ATH (more negative = more room to recover)
    ath_drop = abs(c.ath_change_pct) if c.ath_change_pct < 0 else 0
    if ath_drop >= 90:
        ath_score = 1.0  # 90%+ down = ultimate recovery play
    elif ath_drop >= 70:
        ath_score = 0.8
    elif ath_drop >= 50:
        ath_score = 0.6
    elif ath_drop >= 30:
        ath_score = 0.3
    else:
        ath_score = 0.0
    score += ath_score * WEIGHTS["distance_from_ath"]
    breakdown["ath_distance"] = round(ath_score * WEIGHTS["distance_from_ath"], 3)

    # 4. Supply dilution (FDV/MCap ratio — lower = better)
    if c.fdv and c.fdv > 0:
        dilution = c.fdv / mcap if mcap > 0 else 999
        if dilution <= 2:
            dil_score = 1.0  # Almost no dilution
        elif dilution <= 5:
            dil_score = 0.7
        elif dilution <= 10:
            dil_score = 0.4
        else:
            dil_score = 0.1  # Massive dilution ahead
        score += dil_score * WEIGHTS["supply_dilution"]
        breakdown["dilution"] = round(dil_score * WEIGHTS["supply_dilution"], 3)

    # 5. Circulating ratio (20-80% = sweet spot)
    if c.total_supply and c.total_supply > 0:
        circ_ratio = c.circulating_supply / c.total_supply
        if 0.30 <= circ_ratio <= 0.70:
            circ_score = 1.0
        elif 0.20 <= circ_ratio <= 0.80:
            circ_score = 0.7
        elif circ_ratio >= 0.10 and circ_ratio <= 0.90:
            circ_score = 0.4
        else:
            circ_score = 0.0
        score += circ_score * WEIGHTS["circulating_ratio"]
        breakdown["circulating"] = round(circ_score * WEIGHTS["circulating_ratio"], 3)

    # 6. Binance Futures bonus
    if c.on_binance_futures:
        score += 1.0 * WEIGHTS["binance_futures"]
        breakdown["binance_futures"] = round(WEIGHTS["binance_futures"], 3)

    # 7. Price momentum (slightly negative = entry opportunity)
    # 30d change: -10% to -50% is ideal (not crashing, but available cheap)
    p30 = c.price_change_30d or 0
    if -50 <= p30 <= -10:
        mom_score = 0.8
    elif p30 < -50:
        mom_score = 0.4  # Too far down, might be dead
    elif -10 < p30 <= 5:
        mom_score = 0.5  # Neutral / slightly positive
    elif p30 > 5:
        mom_score = 0.0  # Already pumping — missed the entry
    else:
        mom_score = 0.0
    score += mom_score * WEIGHTS["price_momentum"]
    breakdown["momentum"] = round(mom_score * WEIGHTS["price_momentum"], 3)

    # 8. Coin age — young coins (< 2 years) have more growth potential
    age_years = None
    if c.atl_date:
        try:
            atl_dt = datetime.fromisoformat(c.atl_date.replace("Z", "+00:00"))
            age_years = (datetime.now(timezone.utc) - atl_dt).total_seconds() / (365.25 * 86400)
        except (ValueError, TypeError):
            pass
    # Also use 1y price data as proxy: None means no 1y data = young coin
    if c.price_change_1y is None:
        age_score = 0.8  # Less than 1 year of tracked data = very young
    elif age_years is not None:
        if age_years <= 1:
            age_score = 0.9  # Under 1 year
        elif age_years <= 2:
            age_score = 0.7  # 1-2 years
        elif age_years <= 3:
            age_score = 0.3  # 2-3 years
        else:
            age_score = 0.0  # > 3 years — too old
    elif c.price_change_1y is not None:
        # Has 1y data, no ATL date — at least 1 year old, check if under 2
        # Use 1y change as rough proxy
        age_score = 0.3  # Known to be at least 1 year
    else:
        age_score = 0.2
    score += age_score * WEIGHTS["coin_age"]
    breakdown["coin_age"] = round(age_score * WEIGHTS["coin_age"], 3)

    c.score = round(score, 4)
    c.breakdown = breakdown
    return c


def scan_gems(pages: int = 5, start_page: int = 3) -> List[GemCandidate]:
    """
    Scan CoinGecko for gem candidates across multiple pages.

    Each page = 250 coins. Pages 3-20 cover MCap ranks ~500-5000.
    Default: pages 3-7 (ranks ~500-1750).
    """
    from src.integrations.binance_symbols import is_on_binance_futures

    candidates: List[GemCandidate] = []
    total_scanned = 0

    for page in range(start_page, start_page + pages):
        logger.info("Scanning CoinGecko page %d...", page)
        try:
            resp = httpx.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 250,
                    "page": page,
                    "sparkline": False,
                    "price_change_percentage": "7d,30d,1y",
                },
                timeout=15.0,
            )
            if resp.status_code != 200:
                logger.warning("CoinGecko page %d: HTTP %d", page, resp.status_code)
                time.sleep(5)
                continue

            coins = resp.json()
            for c in coins:
                symbol = c.get("symbol", "").upper()
                mcap = c.get("market_cap") or 0
                vol = c.get("total_volume") or 0
                ath = c.get("ath") or 0
                current = c.get("current_price") or 0
                ath_change = c.get("ath_change_percentage") or 0
                max_supply = c.get("max_supply")
                total_supply = c.get("total_supply")
                circ_supply = c.get("circulating_supply") or 0
                fdv = c.get("fully_diluted_valuation")

                # Quick filter: must be on Binance
                if not is_on_binance_futures(symbol):
                    continue

                total_scanned += 1

                # Size filter
                if mcap < MCAP_MIN or mcap > MCAP_MAX:
                    continue

                # Volume filter
                vol_mcap = vol / mcap if mcap > 0 else 0
                if vol_mcap < VOL_MCAP_MIN:
                    continue

                # ATH drop filter
                if ath_change >= -ATH_DROP_MIN:
                    continue

                # Basic supply sanity
                if total_supply and total_supply > 0:
                    circ_ratio = circ_supply / total_supply
                    if circ_ratio < CIRCULATING_MIN or circ_ratio > CIRCULATING_MAX:
                        continue

                # Dilution check
                if fdv and fdv > 0:
                    dilution = fdv / mcap
                    if dilution > FDV_MCAP_MAX:
                        continue

                gem = GemCandidate(
                    symbol=symbol,
                    name=c.get("name", ""),
                    price=current,
                    market_cap=mcap,
                    volume_24h=vol,
                    circulating_supply=circ_supply,
                    max_supply=max_supply,
                    total_supply=total_supply,
                    fdv=fdv,
                    ath=ath,
                    ath_change_pct=ath_change,
                    price_change_7d=c.get("price_change_percentage_7d_in_currency"),
                    price_change_30d=c.get("price_change_percentage_30d_in_currency"),
                    price_change_1y=c.get("price_change_percentage_1y_in_currency"),
                    market_cap_rank=c.get("market_cap_rank"),
                    on_binance_futures=is_on_binance_futures(symbol),
                    coin_gecko_id=c.get("id", ""),
                )
                gem = _score_gem(gem)
                candidates.append(gem)

            # Rate limit: CoinGecko free = 10-30 calls/min
            time.sleep(3.5)

        except Exception as e:
            logger.error("Error scanning page %d: %s", page, e)
            time.sleep(5)
            continue

    candidates.sort(key=lambda x: x.score, reverse=True)
    logger.info("Scanned %d Binance coins, found %d gem candidates", total_scanned, len(candidates))

    # Enrich top 30 candidates with ATL dates for better age scoring
    import time as _time2
    for i, gem in enumerate(candidates[:30]):
        if gem.price_change_1y is not None or gem.atl_date:
            continue  # Already has age data
        try:
            time.sleep(0.5)  # Rate limit
            resp = httpx.get(
                f"https://api.coingecko.com/api/v3/coins/{gem.coin_gecko_id}",
                params={"localization": "false", "tickers": "false", "community_data": "false", "sparkline": "false"},
                timeout=10.0,
            )
            if resp.status_code == 200:
                coin_data = resp.json()
                md = coin_data.get("market_data", {})
                atl_dt = md.get("atl_date", {}).get("usd")
                if atl_dt:
                    gem.atl_date = atl_dt
                    gem = _score_gem(gem)  # Re-score with better age data
                    candidates[i] = gem
        except Exception:
            pass

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates


def format_gem_report(candidates: List[GemCandidate], top_n: int = 20) -> str:
    """Format gem candidates for Telegram."""
    if not candidates:
        return "No gem candidates found on this scan."

    lines = [
        "🔍 *Spot Gem Scan — {}*".format(datetime.now().strftime("%Y-%m-%d")),
        "Coins on Binance with strong tokenomics for 3-6 month holds",
        "",
        f"`{'Rank':<4} {'Symbol':<8} {'Score':<6} {'MCap':<10} {'Vol/MCap':<9} {'ATH%':<7} {'FDVx':<6} {'Circ%':<6} {'30d%':<7}`",
        "`" + "-" * 68 + "`",
    ]

    for i, c in enumerate(candidates[:top_n], 1):
        vol_mcap = c.volume_24h / c.market_cap * 100 if c.market_cap > 0 else 0
        dilution = c.fdv / c.market_cap if c.fdv and c.market_cap > 0 else 0
        circ_pct = c.circulating_supply / c.total_supply * 100 if c.total_supply and c.total_supply > 0 else 0
        ath_str = f"{c.ath_change_pct:+.0f}%" if c.ath_change_pct else "N/A"
        p30 = f"{c.price_change_30d:+.0f}%" if c.price_change_30d else "N/A"

        # Emoji based on score
        if c.score >= 0.7:
            emoji = "🟢"
        elif c.score >= 0.5:
            emoji = "🟡"
        else:
            emoji = "⚪"

        fut = "📍" if c.on_binance_futures else "  "

        lines.append(
            f"`{i:<4} {c.symbol:<8} {c.score:<6.2f} ${c.market_cap/1e6:<7.0f}M {vol_mcap:<7.1f}% {ath_str:<7} {dilution:<5.1f}x {circ_pct:<5.0f}% {p30:<7}`{emoji}{fut}"
        )

    lines.append("")
    lines.append("_MCap: Market Cap | FDVx: FDV/MCap ratio | Circ%: Circulating %_")
    lines.append("_🟢 Strong | 🟡 Moderate | ⚪ Weak | 📍 Binance Futures_")
    return "\n".join(lines)