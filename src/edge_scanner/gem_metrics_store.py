"""Utility functions to persist gem social/developer metrics over time
so we can track progression (e.g. GitHub stars growth, commit activity,
community growth) across scan cycles.
"""
import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/crypto.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gem_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    coin_gecko_id TEXT,
    scanned_at TEXT NOT NULL,
    github_stars INTEGER,
    github_forks INTEGER,
    github_subscribers INTEGER,
    commit_count_4w INTEGER,
    code_additions_4w INTEGER,
    code_deletions_4w INTEGER,
    pull_requests_merged INTEGER,
    twitter_followers INTEGER,
    reddit_subscribers INTEGER,
    telegram_users INTEGER,
    has_burn_program INTEGER DEFAULT 0,
    score REAL DEFAULT 0,
    UNIQUE(symbol, scanned_at)
)
"""


def save_gem_metrics(gems):
    """Persist a snapshot of gem metrics to track progression over time."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    now = datetime.now(timezone.utc).isoformat()
    for g in gems:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO gem_metrics
                (symbol, coin_gecko_id, scanned_at, github_stars, github_forks,
                 github_subscribers, commit_count_4w, code_additions_4w,
                 code_deletions_4w, pull_requests_merged, twitter_followers,
                 reddit_subscribers, telegram_users, has_burn_program, score)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (g.symbol, g.coin_gecko_id, now, g.github_stars, g.github_forks,
                 g.github_subscribers, g.commit_count_4w, g.code_additions_4w,
                 g.code_deletions_4w, g.pull_requests_merged, g.twitter_followers,
                 g.reddit_subscribers, g.telegram_users, 1 if g.has_burn_program else 0,
                 g.score)
            )
        except Exception:
            continue
    conn.commit()
    conn.close()


def get_progression(symbol, lookback_scans=2):
    """Return previous snapshots for a gem to compute deltas."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    rows = conn.execute(
        """SELECT * FROM gem_metrics WHERE symbol=?
           ORDER BY scanned_at DESC LIMIT ?""",
        (symbol, lookback_scans)
    ).fetchall()
    conn.close()
    return rows


def format_progression_report(gems):
    """Format a progression report comparing current vs previous scan data."""
    current = {g.symbol: g for g in gems}
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    lines = ["📈 *Gem Progression (vs previous scan):*", "`" + "-" * 60 + "`"]
    for symbol, g in current.items():
        # Exclude the newest snapshot (current scan) — we want the PRIOR scan
        prev = conn.execute(
            """SELECT github_stars, commit_count_4w, github_forks, twitter_followers
               FROM gem_metrics WHERE symbol=?
               ORDER BY scanned_at DESC LIMIT 2 OFFSET 1""",
            (symbol,)
        ).fetchone()
        if not prev:
            continue
        stars_d = (g.github_stars or 0) - (prev[0] or 0)
        commits_d = (g.commit_count_4w or 0) - (prev[1] or 0)
        forks_d = (g.github_forks or 0) - (prev[2] or 0)
        tw_d = (g.twitter_followers or 0) - (prev[3] or 0)
        parts = []
        if stars_d:
            parts.append(f"⭐ {stars_d:+d}")
        if commits_d:
            parts.append(f"💻 {commits_d:+d} commits")
        if forks_d:
            parts.append(f"⑂ {forks_d:+d}")
        if tw_d:
            parts.append(f"🐦 {tw_d:+d}")
        if parts:
            lines.append(f"  {symbol:<10} {' '.join(parts)}")
    conn.close()
    return "\n".join(lines) if len(lines) > 1 else "No progression data yet (need 2+ scans)."