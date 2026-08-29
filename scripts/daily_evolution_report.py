#!/usr/bin/env python3
"""Daily evolution report generator for Telegram."""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env for Telegram bot token
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith('TELEGRAM_BOT_TOKEN='):
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")
            break

from dotenv import load_dotenv
load_dotenv()

from src.edge_scanner.evolution import auto_evolve, analyze_configs, rank_configs, MIN_NON_FLAT_TRADES, SIGNIFICANCE_LEVEL, MIN_CANDIDATE_RECENT_DAYS
from src.edge_scanner.scoring_config import ACTIVE_CONFIG, ALL_CONFIGS
from datetime import datetime, timezone
import sqlite3


def get_pending_and_resolved(db_path='data/crypto.db'):
    conn = sqlite3.connect(db_path)
    pending = conn.execute("SELECT COUNT(*) FROM edge_signals WHERE status = 'PENDING'").fetchone()[0]
    resolved = conn.execute("SELECT COUNT(*) FROM edge_signals WHERE status = 'RESOLVED'").fetchone()[0]
    conn.close()
    return pending, resolved


def _is_active(config_version: str) -> bool:
    """Check if a config version is not disabled."""
    try:
        cfg = ALL_CONFIGS.get(config_version)
        return cfg is not None and cfg.status != 'disabled'
    except Exception:
        return True  # If we can't check, assume active

def build_report() -> str:
    result = auto_evolve(dry_run=True)
    stats = analyze_configs()
    ranked = rank_configs(stats)
    pending, resolved = get_pending_and_resolved()

    active_ver = ACTIVE_CONFIG.version
    active_desc = ACTIVE_CONFIG.description
    active_s = stats.get(active_ver)
    active_wr = active_s.win_rate if active_s else 0.0
    active_trades = active_s.non_flat_trades if active_s else 0
    active_flat = active_s.flat_rate if active_s else 0.0

    # Top 3 eligible (active configs only)
    eligible = [c for c in stats.values() if c.non_flat_trades >= MIN_NON_FLAT_TRADES
                and _is_active(c.config_version)]
    eligible.sort(key=lambda c: c.composite_rank_score, reverse=True)
    top3 = eligible[:3] if eligible else []

    lines = []
    lines.append("\U0001f4ca *Daily Evolution Report*")
    lines.append(f"_{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
    lines.append("")

    # Section 1: Active config
    lines.append("\u2501\u2501\u2501\u2501\u2501 *ACTIVE CONFIG* \u2501\u2501\u2501\u2501\u2501")
    lines.append(f"\U0001f4cc *Config {active_ver} \u2014 {active_desc.split(':')[0]}*")
    lines.append(f"   WR: {active_wr:.1f}% | Trades: {active_trades} | Flat: {active_flat:.1f}%")
    if active_trades < MIN_NON_FLAT_TRADES:
        lines.append(f"   \u26a0\ufe0f  *Insufficient data* ({active_trades}/{MIN_NON_FLAT_TRADES} min)")
    lines.append(f"   _{active_desc}_")
    lines.append("")

    # Section 2: Top performers
    lines.append("\u2501\u2501\u2501\u2501\u2501 *TOP PERFORMERS* \u2501\u2501\u2501\u2501\u2501")
    if top3:
        for i, c in enumerate(top3, 1):
            medals = {1: "\U0001f947", 2: "\U0001f948", 3: "\U0001f949"}
            medal = medals.get(i, "")
            marker = " \u2190 *CURRENT*" if c.config_version == active_ver else ""
            pf_str = f"{c.profit_factor:.2f}" if c.profit_factor != float('inf') else "\u221e"
            lines.append(f"{medal} *{c.config_version}*{marker}")
            lines.append(f"   WR: {c.win_rate:.1f}% | Trades: {c.non_flat_trades} | PF: {pf_str}")
            lines.append(f"   Score: {c.composite_rank_score:.1f} | Flat: {c.flat_rate:.1f}% | AvgRet: {c.avg_return_pct:+.2f}%")
            lines.append(f"   Resolve: avg {c.avg_time_to_resolve_hours:.1f}h | Expectancy: {c.expectancy:+.2f}%")
            lines.append("")
    else:
        lines.append("_No config has enough data yet_")
        lines.append("")

    # Section 3: Promotion analysis
    lines.append("\u2501\u2501\u2501\u2501\u2501 *PROMOTION ANALYSIS* \u2501\u2501\u2501\u2501\u2501")
    if top3 and top3[0].config_version != active_ver:
        best = top3[0]

        lines.append(f"\U0001f50d *Recommended:* `{best.config_version}` \u2192 NEW ACTIVE")
        lines.append(f"   {best.win_rate:.1f}% WR vs {active_wr:.1f}% WR (current)")

        if active_trades >= MIN_NON_FLAT_TRADES:
            improvement = best.win_rate - active_wr
            lines.append(f"   Improvement: +{improvement:.1f}pp")
            lines.append(f"   \u2696\ufe0f  Must pass z-test (p < {SIGNIFICANCE_LEVEL}) vs active \u2014 dry run only")
        else:
            lines.append(f"   \u26a0\ufe0f  Active config ({active_ver}) has insufficient data for z-test")
            lines.append(f"   \u2705 *Config {best.config_version} has {best.non_flat_trades} trades* \u2014 sufficient")
            lines.append(f"   \U0001f4a1 *Strong candidate for promotion*")

        lines.append("")
        lines.append(f"   \U0001f4ca *Config {best.config_version} details:*")
        lines.append(f"   \u2022 Avg resolve: {best.avg_time_to_resolve_hours:.1f}h")
        lines.append(f"   \u2022 Expectancy per trade: {best.expectancy:+.2f}%")
    else:
        lines.append(f"\u2705 Current active `{active_ver}` is the top performer")
    lines.append("")

    # Section 4: Other notable
    notable = []
    if top3:
        notable = [c for c in stats.values()
                   if c.non_flat_trades >= MIN_NON_FLAT_TRADES
                   and c.win_rate > 55
                   and c.config_version not in [t.config_version for t in top3]
                   and _is_active(c.config_version)]
        notable.sort(key=lambda c: c.win_rate, reverse=True)
    if notable:
        lines.append("\u2501\u2501\u2501\u2501\u2501 *OTHER NOTABLE* \u2501\u2501\u2501\u2501\u2501")
        for c in notable[:3]:
            pf_str = f"{c.profit_factor:.2f}" if c.profit_factor != float('inf') else "\u221e"
            lines.append(f"\u2022 *{c.config_version}* \u2014 {c.win_rate:.1f}% WR ({c.non_flat_trades}t, PF {pf_str})")
        lines.append("")

    # Section 5: Summary
    lines.append("\u2501\u2501\u2501\u2501\u2501 *SUMMARY* \u2501\u2501\u2501\u2501\u2501")
    lines.append(f"\U0001f4c8 Total resolved signals: {resolved}")
    lines.append(f"\u23f3 Pending signals: {pending}")
    lines.append(f"📊 Active configs registered: {sum(1 for c in ALL_CONFIGS.values() if c.status != 'disabled')} ({len(ALL_CONFIGS)} total)")
    lines.append(f"\U0001f3c6 Configs with \u2265{MIN_NON_FLAT_TRADES}t data: {len(eligible)}")
    lines.append(f"\u2699\ufe0f  Min {MIN_NON_FLAT_TRADES} trades | z-test p<{SIGNIFICANCE_LEVEL} | signal within {MIN_CANDIDATE_RECENT_DAYS}d")
    lines.append("")
    lines.append("\U0001f916 *Edge Scanner \u2014 auto-generated*")

    return "\n".join(lines)


def send_telegram(message):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = -1001482338614

    if not bot_token:
        print("ERROR: TELEGRAM_BOT_TOKEN not found")
        return False

    import httpx
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = httpx.post(url, json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("ok"):
            print("Report sent successfully to Telegram")
            return True
        else:
            print(f"Telegram API error: {data}")
            return False
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")
        return False


if __name__ == '__main__':
    print("Generating daily evolution report...")
    report = build_report()
    print(report)
    print()
    print("Sending to Telegram...")
    ok = send_telegram(report)
    if ok:
        print("Done.")
    else:
        print("Failed to send report.")
        sys.exit(1)
