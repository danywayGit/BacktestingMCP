#!/usr/bin/env python3
"""Crypto edge scanner health check - reports issues only."""
import sqlite3
from datetime import datetime, timezone

conn = sqlite3.connect('data/crypto.db')
now = datetime.now(timezone.utc)

issues = []

# 1. Funding history freshness
try:
    row = conn.execute('SELECT MAX(fetched_at) FROM funding_history').fetchone()
    if row and row[0]:
        ts = row[0]
        if isinstance(ts, str):
            from datetime import datetime as dt
            fetched = dt.fromisoformat(ts.replace('Z', '+00:00'))
        else:
            fetched = ts
        age_hours = (now - fetched).total_seconds() / 3600
        if age_hours > 1:
            issues.append(f"FUNDING_STALE: last funding poll was {age_hours:.1f}h ago ({ts})")
    else:
        issues.append("FUNDING_NO_DATA: funding_history table is empty")
except Exception as e:
    issues.append(f"FUNDING_ERROR: {e}")

# 2. Config flat rates > 90% with > 50 signals
try:
    rows = conn.execute("""
        SELECT config_version,
               COUNT(*) as total,
               SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
               SUM(CASE WHEN outcome = 'FLAT' THEN 1 ELSE 0 END) as flats
        FROM edge_signals
        WHERE status = 'RESOLVED'
        GROUP BY config_version
        ORDER BY total DESC
    """).fetchall()
    for r in rows:
        cv, total, wins, losses, flats = r
        if total > 50:
            flat_pct = round(flats / total * 100, 1)
            if flat_pct > 90:
                issues.append(f"HIGH_FLAT: {cv} has {flat_pct}% flat rate ({flats}/{total} signals)")
except Exception as e:
    issues.append(f"CONFIG_ERROR: {e}")

# 3. V8.0 signal accumulation
try:
    row = conn.execute("SELECT COUNT(*) FROM edge_signals WHERE config_version='8.0'").fetchone()
    v80_total = row[0] if row else 0
    if v80_total == 0:
        issues.append("V80_NO_SIGNALS: V8.0 has accumulated zero signals")
except Exception as e:
    issues.append(f"V80_ERROR: {e}")

# 4. V7.0 win rate
try:
    row = conn.execute("SELECT COUNT(*), SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END), SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) FROM edge_signals WHERE config_version='7.0' AND status='RESOLVED' AND outcome IS NOT NULL AND outcome != 'FLAT'").fetchone()
    total, wins, losses = row
    if (wins + losses) >= 10:
        wr = round(wins / (wins + losses) * 100, 1)
        if wr < 50:
            issues.append(f"V70_WR_BELOW_50: V7.0 win rate is {wr}% ({wins}/{wins+losses} non-flat trades)")
        else:
            print(f"V70_WR_OK: V7.0 win rate is {wr}% ({wins}/{wins+losses})")
    else:
        print(f"V70_INSUFFICIENT: V7.0 has only {wins+losses} non-flat trades (need 10)")
except Exception as e:
    issues.append(f"V70_ERROR: {e}")

conn.close()

if issues:
    print("---ISSUES---")
    for i in issues:
        print(i)
else:
    print("---ALL_HEALTHY---")