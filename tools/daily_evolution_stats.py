#!/usr/bin/env python3
"""Generate stats for daily evolution report."""
import sqlite3
from datetime import datetime, timezone, timedelta

conn = sqlite3.connect('/home/hermes/BacktestingMCP/data/crypto.db')
conn.row_factory = sqlite3.Row

# Active config in DB
active_db = conn.execute("SELECT * FROM scoring_configs WHERE is_active = 1").fetchone()
if active_db:
    print(f"ACTIVE_DB|{dict(active_db)}")
else:
    print("ACTIVE_DB|None")

# Config count
nc = conn.execute("SELECT COUNT(*) FROM scoring_configs").fetchone()[0]
print(f"CONFIG_COUNT|{nc}")

# Recent 7 days
week_ago = datetime.now(timezone.utc) - timedelta(days=7)
recent = conn.execute("""
    SELECT outcome, COUNT(*) as cnt
    FROM edge_signals
    WHERE entry_time >= ?
    GROUP BY outcome
""", (week_ago.isoformat(),)).fetchall()
outcomes = {}
for r in recent:
    outcomes[r['outcome'] or 'PENDING'] = r['cnt']
print(f"RECENT7|{outcomes}")

# Total, resolved, pending
tot = conn.execute("SELECT COUNT(*) FROM edge_signals").fetchone()[0]
res = conn.execute("SELECT COUNT(*) FROM edge_signals WHERE status='RESOLVED'").fetchone()[0]
pen = conn.execute("SELECT COUNT(*) FROM edge_signals WHERE status='PENDING'").fetchone()[0]
print(f"TOTALS|total={tot} resolved={res} pending={pen}")

# This week's resolutions
today = datetime.now(timezone.utc)
d7 = today - timedelta(days=7)
weekly = conn.execute("""
    SELECT outcome, COUNT(*) as cnt
    FROM edge_signals
    WHERE status='RESOLVED' AND resolved_at >= ? AND outcome IS NOT NULL
    GROUP BY outcome
""", (d7.isoformat(),)).fetchall()
print(f"WEEKLY_RESOLVED|{dict(weekly)}")

# All configs' win-rate data (non-flat)
rows = conn.execute("""
    SELECT config_version, outcome, COUNT(*) as cnt
    FROM edge_signals
    WHERE status='RESOLVED' AND outcome IS NOT NULL
    GROUP BY config_version, outcome
    ORDER BY config_version
""").fetchall()

config_data = {}
for r in rows:
    cv = r['config_version']
    if cv not in config_data:
        config_data[cv] = {'wins': 0, 'losses': 0, 'flats': 0, 'total': 0}
    config_data[cv][r['outcome'].lower()] = r['cnt']
    config_data[cv]['total'] += r['cnt']

print("CONFIG_PERF|")
for cv, d in sorted(config_data.items()):
    non_flat = d['wins'] + d['losses']
    wr = (d['wins'] / non_flat * 100) if non_flat > 0 else 0
    flat_pct = (d['flats'] / d['total'] * 100) if d['total'] > 0 else 0
    print(f"  {cv:>10} WR={wr:5.1f}% W={d['wins']:>4} L={d['losses']:>4} F={d['flats']:>4} T={d['total']:>5} FlatRate={flat_pct:5.1f}%")

# Pending that are due
now = datetime.now(timezone.utc)
due = 0
for sig in conn.execute("SELECT * FROM edge_signals WHERE status='PENDING'").fetchall():
    et = datetime.fromisoformat(sig['entry_time'])
    if et.tzinfo is None:
        et = et.replace(tzinfo=timezone.utc)
    if now >= et + timedelta(hours=sig['horizon_hours']):
        due += 1
print(f"PENDING_DUE|{due}")

# v1.0 composite score stats by outcome
score_data = conn.execute("""
    SELECT outcome,
           ROUND(MIN(composite_score), 2) as min_s,
           ROUND(AVG(composite_score), 2) as avg_s,
           ROUND(MAX(composite_score), 2) as max_s
    FROM edge_signals
    WHERE config_version='v1.0' AND status='RESOLVED' AND outcome IS NOT NULL
    GROUP BY outcome
""").fetchall()
print("V1_SCORES|")
for s in score_data:
    print(f"  {s['outcome']:>6}: min={s['min_s']:>6} avg={s['avg_s']:>6} max={s['max_s']:>6}")

# Last 10 resolved
last10 = conn.execute("""
    SELECT symbol, direction, outcome, forward_return_pct, config_version, resolved_at
    FROM edge_signals
    WHERE status='RESOLVED' AND outcome IS NOT NULL
    ORDER BY resolved_at DESC LIMIT 10
""").fetchall()
print("LAST10|")
for r in last10:
    print(f"  {r['symbol']:>10} dir={r['direction']:>5} outcome={r['outcome']:>6} ret={r['forward_return_pct']:>7.2f}% cfg={r['config_version']}")

conn.close()