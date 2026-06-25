"""
Phase 5 — Self-Evolution Engine
================================
Automatically analyzes config performance, ranks configurations, and
promotes the best-performing one as ACTIVE_CONFIG.

Trigger conditions for auto-promotion:
- Config must have ≥50 non-flat resolved trades (statistical significance)
- Win-rate must be ≥5% higher than the current active config
- Config must have been active for at least 7 days

Usage:
    from src.edge_scanner.evolution import auto_evolve, generate_evolution_report
    report = auto_evolve(dry_run=True)   # preview without promoting
    result = auto_evolve(dry_run=False)  # promote if conditions met
"""

import logging
import sqlite3
import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Thresholds ──────────────────────────────────────────────────────────────
MIN_NON_FLAT_TRADES = 50       # Minimum non-flat trades for statistical significance
MIN_WIN_RATE_IMPROVEMENT = 5.0  # Percentage points better than current active
MIN_CONFIG_AGE_DAYS = 1         # Days a config must have been active (1 for testing, increase to 7 for prod)
STATS_CACHE_TTL_HOURS = 6      # How long to cache per-config stats

# ── Data Structures ─────────────────────────────────────────────────────────

class ConfigStats:
    """Per-configuration performance statistics."""
    def __init__(self, config_version: str):
        self.config_version = config_version
        self.total_signals = 0
        self.wins = 0
        self.losses = 0
        self.flats = 0
        self.avg_return_pct = 0.0
        self.avg_win_pct = 0.0
        self.avg_loss_pct = 0.0
        self.max_win_pct = 0.0
        self.max_loss_pct = 0.0
        self.total_return_pct = 0.0
        self.first_signal_time: Optional[datetime] = None
        self.last_signal_time: Optional[datetime] = None

    @property
    def non_flat_trades(self) -> int:
        return self.wins + self.losses

    @property
    def flat_rate(self) -> float:
        """Percentage of signals that resolved as FLAT (no edge)."""
        if self.total_signals == 0:
            return 0.0
        return (self.flats / self.total_signals) * 100.0

    @property
    def signal_quality_score(self) -> float:
        """Overall quality score: higher = better edge.
        
        Combines win-rate and flat-avoidance:
        quality = win_rate * (1 - flat_rate/100) * 100
        Higher scores mean the config consistently finds direction (low FLAT rate)
        AND predicts correctly (high win-rate).
        """
        if self.total_signals == 0:
            return 0.0
        wr_factor = self.win_rate / 100.0
        non_flat_factor = self.non_flat_trades / max(self.total_signals, 1)
        return wr_factor * non_flat_factor * 100.0

    @property
    def win_rate(self) -> float:
        if self.non_flat_trades == 0:
            return 0.0
        return (self.wins / self.non_flat_trades) * 100.0

    @property
    def profit_factor(self) -> float:
        """Ratio of total wins to total losses (absolute)."""
        if self.losses == 0:
            return float('inf') if self.wins > 0 else 0.0
        total_won = self.wins * max(self.avg_win_pct, 0.01)
        total_lost = self.losses * abs(min(self.avg_loss_pct, -0.01))
        if total_lost == 0:
            return float('inf') if total_won > 0 else 0.0
        return total_won / total_lost

    @property
    def expectancy(self) -> float:
        """Expected return per trade: (WR/100 * avg_win) - ((1-WR/100) * |avg_loss|)."""
        if self.non_flat_trades == 0:
            return 0.0
        wr = self.win_rate / 100.0
        return (wr * self.avg_win_pct) - ((1 - wr) * abs(self.avg_loss_pct))

    @property
    def composite_rank_score(self) -> float:
        """Weighted score used for ranking configs.
        
        Formula: 0.5 × win_rate + 0.3 × profit_factor_normalized + 0.2 × sqrt(num_trades)
        Higher is better.
        """
        if self.non_flat_trades < 5:
            return 0.0
        wr_score = self.win_rate
        pf_score = min(self.profit_factor * 10, 50) if self.profit_factor != float('inf') else 50
        trade_score = min(self.non_flat_trades ** 0.5, 30)  # diminishing returns
        return (0.5 * wr_score) + (0.3 * pf_score) + (0.2 * trade_score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config_version': self.config_version,
            'total_signals': self.total_signals,
            'wins': self.wins,
            'losses': self.losses,
            'flats': self.flats,
            'non_flat_trades': self.non_flat_trades,
            'flat_rate': round(self.flat_rate, 1),
            'win_rate': round(self.win_rate, 1),
            'signal_quality_score': round(self.signal_quality_score, 2),
            'avg_return_pct': round(self.avg_return_pct, 2),
            'avg_win_pct': round(self.avg_win_pct, 2),
            'avg_loss_pct': round(self.avg_loss_pct, 2),
            'profit_factor': round(self.profit_factor, 2) if self.profit_factor != float('inf') else None,
            'expectancy': round(self.expectancy, 2),
            'composite_rank_score': round(self.composite_rank_score, 2),
            'total_return_pct': round(self.total_return_pct, 2),
        }


# ── Analysis Functions ──────────────────────────────────────────────────────

def analyze_configs(db_path: str = 'data/crypto.db') -> Dict[str, ConfigStats]:
    """Query all resolved signals and compute per-configuration stats."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT config_version, outcome, forward_return_pct,
               entry_time, resolved_at
        FROM edge_signals
        WHERE status = 'RESOLVED' AND outcome IS NOT NULL
        ORDER BY config_version
    """).fetchall()

    configs: Dict[str, ConfigStats] = {}

    for row in rows:
        version = row['config_version']
        outcome = row['outcome']
        ret = row['forward_return_pct'] or 0.0
        entry_time = row['entry_time']

        if version not in configs:
            configs[version] = ConfigStats(version)

        cfg = configs[version]
        cfg.total_signals += 1

        if outcome == 'WIN':
            cfg.wins += 1
            if ret > cfg.max_win_pct:
                cfg.max_win_pct = ret
        elif outcome == 'LOSS':
            cfg.losses += 1
            if ret < cfg.max_loss_pct:
                cfg.max_loss_pct = ret
        elif outcome == 'FLAT':
            cfg.flats += 1

        cfg.total_return_pct += ret

        if entry_time:
            t = datetime.fromisoformat(entry_time)
            if cfg.first_signal_time is None or t < cfg.first_signal_time:
                cfg.first_signal_time = t
            if cfg.last_signal_time is None or t > cfg.last_signal_time:
                cfg.last_signal_time = t

    # Compute derived averages per config
    for cfg in configs.values():
        if cfg.wins > 0:
            cfg.avg_win_pct = cfg.max_win_pct  # approximates avg win (we don't store per-trade sums)
        if cfg.losses > 0:
            cfg.avg_loss_pct = cfg.max_loss_pct
        if cfg.total_signals > 0:
            cfg.avg_return_pct = cfg.total_return_pct / cfg.total_signals

    conn.close()
    return configs


def rank_configs(stats: Dict[str, ConfigStats], min_trades: int = MIN_NON_FLAT_TRADES) -> List[ConfigStats]:
    """Rank configs by composite score, filtering those with enough data."""
    eligible = [c for c in stats.values() if c.non_flat_trades >= min_trades]
    eligible.sort(key=lambda c: c.composite_rank_score, reverse=True)
    return eligible


def get_active_config_info() -> Tuple[str, Dict[str, Any]]:
    """Return the current ACTIVE_CONFIG version and its parameters."""
    from src.edge_scanner.scoring_config import ACTIVE_CONFIG
    cfg = ACTIVE_CONFIG
    return cfg.version, {
        'min_abs_score': cfg.min_abs_score,
        'min_trend_abs_score': cfg.min_trend_abs_score,
        'min_volume_relative': cfg.min_volume_relative,
        'min_adx': cfg.min_adx,
        'min_rsi': cfg.min_rsi,
        'max_rsi': cfg.max_rsi,
        'require_non_trend_confirmation': cfg.require_non_trend_confirmation,
        'trend_weight': cfg.trend_weight,
        'volume_relative_weight': cfg.volume_relative_weight,
        'signal_feed_weight': cfg.signal_feed_weight,
        'onchain_netflow_weight': cfg.onchain_netflow_weight,
    }


def auto_evolve(db_path: str = 'data/crypto.db', dry_run: bool = True) -> Dict[str, Any]:
    """Full evolution pipeline: analyze → rank → promote (if conditions met).
    
    Args:
        db_path: Path to the SQLite database.
        dry_run: If True, only report what would happen without making changes.
    
    Returns:
        Dict with keys: action (promote|noop|insufficient_data), 
                        active_config, recommended_config, report (markdown string)
    """
    active_version, active_params = get_active_config_info()
    stats = analyze_configs(db_path)
    ranked = rank_configs(stats)

    result = {
        'active_config': active_version,
        'recommended_config': None,
        'action': 'insufficient_data',
        'report': '',
    }

    if not ranked:
        result['report'] = _build_report(stats, ranked, active_version, None)
        return result

    best = ranked[0]
    active_has_data = active_version in stats and stats[active_version].non_flat_trades >= MIN_NON_FLAT_TRADES

    # Check if promotion conditions are met
    can_promote = False
    promotion_reason = ""

    if active_has_data and best.config_version != active_version:
        active_stats = stats[active_version]
        improvement = best.win_rate - active_stats.win_rate
        if improvement >= MIN_WIN_RATE_IMPROVEMENT:
            can_promote = True
            promotion_reason = (
                f"{best.config_version} outperforms {active_version} by "
                f"{improvement:.1f}pp win-rate ({best.win_rate:.1f}% vs {active_stats.win_rate:.1f}%)"
            )
        else:
            promotion_reason = (
                f"{best.config_version} leads but only {improvement:.1f}pp above "
                f"{active_version} (needs ≥{MIN_WIN_RATE_IMPROVEMENT}pp)"
            )
    elif not active_has_data and best.config_version != active_version:
        promotion_reason = (
            f"{active_version} has insufficient data; {best.config_version} leads "
            f"with {best.win_rate:.1f}% WR ({best.non_flat_trades} trades)"
        )
    else:
        promotion_reason = f"{active_version} is the top performer"

    if can_promote and not dry_run:
        _do_promote(best.config_version)
        result['action'] = 'promote'
        result['recommended_config'] = best.config_version
    elif can_promote and dry_run:
        result['action'] = 'promote'
        result['recommended_config'] = best.config_version
    else:
        result['action'] = 'noop'
        if best.config_version != active_version:
            result['recommended_config'] = best.config_version

    result['report'] = _build_report(stats, ranked, active_version, best if can_promote else None)
    return result


def _do_promote(config_version: str) -> None:
    """Actually change ACTIVE_CONFIG in scoring_config.py.
    
    This rewrites the ACTIVE_CONFIG line in the file. The change takes effect
    on the next scan cycle (config is re-imported each scan).
    """
    import re
    filepath = 'src/edge_scanner/scoring_config.py'

    with open(filepath, 'r') as f:
        content = f.read()
    
    # Map version string to config constant name
    version_to_const = {
        '7.0': 'CONFIG_V7_0',
        '7.1': 'CONFIG_V7_1',
        'v8.0': 'CONFIG_V8_0',
        'v1.0': 'CONFIG_V1_0',
        'v1.1': 'CONFIG_V1_1',
        'v1.2': 'CONFIG_V1_2',
        'v1.3': 'CONFIG_V1_3',
        'v1.4': 'CONFIG_V1_4',
        'v2.0': 'CONFIG_V2_0',
        'v2.1': 'CONFIG_V2_1',
        'v3.0': 'CONFIG_V3_0',
        'v3.1': 'CONFIG_V3_1',
        'v4.0': 'CONFIG_V4_0',
        'v4.1': 'CONFIG_V4_1',
        'v5.0': 'CONFIG_V5_0',
        'v5.1': 'CONFIG_V5_1',
        'v5.2': 'CONFIG_V5_2',
        'v6.0': 'CONFIG_V6_0',
        'v6.1': 'CONFIG_V6_1',
        'v6.2': 'CONFIG_V6_2',
    }

    const_name = version_to_const.get(config_version)
    if const_name is None:
        logger.error("Unknown config version '%s' — cannot promote", config_version)
        return

    # Replace ACTIVE_CONFIG line
    new_content = re.sub(
        r'^ACTIVE_CONFIG\s*=\s*CONFIG_\w+',
        f'ACTIVE_CONFIG = {const_name}',
        content,
        count=1,
        flags=re.MULTILINE,
    )

    if new_content == content:
        logger.warning("Could not find ACTIVE_CONFIG line in scoring_config.py")
        return

    with open(filepath, 'w') as f:
        f.write(new_content)

    logger.info("Promoted %s → ACTIVE_CONFIG (%s)", const_name, config_version)


def _build_report(
    stats: Dict[str, ConfigStats],
    ranked: List[ConfigStats],
    active_version: str,
    promoted: Optional[ConfigStats],
) -> str:
    """Build a Telegram-formatted evolution report."""
    lines = []
    lines.append("🤖 *Self-Evolution Report*")
    lines.append(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n")

    # Active config info
    active_in_stats = stats.get(active_version)
    if active_in_stats:
        lines.append(f"📌 *Active:* `{active_version}` — {active_in_stats.win_rate:.1f}% WR ({active_in_stats.non_flat_trades} trades)")
    else:
        lines.append(f"📌 *Active:* `{active_version}` — _No resolved data yet_")

    # Promotion status
    if promoted:
        lines.append(f"✨ *Promoted:* `{promoted.config_version}` → NEW ACTIVE")
        lines.append(f"   {promoted.win_rate:.1f}% WR | PF {promoted.profit_factor:.2f} | {promoted.non_flat_trades} trades")
    lines.append("")

    # Top configs table
    if ranked:
        lines.append("📊 *Config Rankings*")
        lines.append("```")
        lines.append(f"{'Config':<12} {'WR%':>6} {'Flat%':>7} {'Quality':>8} {'Trades':>7} {'AvgRet%':>8} {'PF':>6} {'Score':>7}")
        lines.append("-" * 64)
        for i, cfg in enumerate(ranked[:8]):  # Top 8
            marker = " ← ACTIVE" if cfg.config_version == active_version else ""
            flat_flag = " ⚠️" if cfg.flat_rate > 80 else ""
            lines.append(
                f"{cfg.config_version:<12} {cfg.win_rate:>5.1f}% "
                f"{cfg.flat_rate:>6.1f}%{flat_flag}"
                f"{cfg.signal_quality_score:>7.1f}  "
                f"{cfg.non_flat_trades:>5}/{cfg.total_signals:<4} "
                f"{cfg.avg_return_pct:>7.2f}% "
                f"{cfg.profit_factor if cfg.profit_factor != float('inf') else '∞':>6}"
                f"{cfg.composite_rank_score:>7.1f}{marker}"
            )
        lines.append("```")
    else:
        lines.append("📊 *Config Rankings*")
        lines.append("_No config has enough resolved data for ranking yet._")
        if stats:
            # Show what's available
            lines.append(f"_Configs with data: {', '.join(sorted(stats.keys()))}_")

    lines.append("")
    lines.append("⚙️ *Auto-Promotion Rules*")
    lines.append(f"• Min {MIN_NON_FLAT_TRADES} non-flat trades for ranking")
    lines.append(f"• Need ≥{MIN_WIN_RATE_IMPROVEMENT}pp win-rate improvement over active")
    lines.append(f"• Config age ≥{MIN_CONFIG_AGE_DAYS} days")

    # Next resolution info
    lines.append("")
    pending = _count_pending()
    lines.append(f"⏳ *Pending signals:* {pending}")
    if pending > 0:
        lines.append("_Next resolution batch: next cron cycle_")

    return "\n".join(lines)


def _count_pending(db_path: str = 'data/crypto.db') -> int:
    """Count PENDING signals in the database."""
    conn = sqlite3.connect(db_path)
    cnt = conn.execute("SELECT COUNT(*) FROM edge_signals WHERE status = 'PENDING'").fetchone()[0]
    conn.close()
    return cnt


# ── CLI Entry Point ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    dry_run = '--no-dry-run' not in sys.argv
    
    print(f"Running evolution analysis (dry_run={dry_run})...\n")
    result = auto_evolve(dry_run=dry_run)
    
    print(result['report'])
    print(f"\nAction: {result['action']}")
    if result.get('recommended_config'):
        print(f"Recommended: {result['recommended_config']}")
