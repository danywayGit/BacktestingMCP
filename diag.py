#!/usr/bin/env python3
"""Check what outcome values V7.x signals actually have."""
import sqlite3

conn = sqlite3.connect('/home/hermes/BacktestingMCP/data/crypto.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Check V7.x outcomes
cur.execute("SELECT config_version, outcome, status, COUNT(*) as cnt FROM edge_signals WHERE config_version LIKE '%7.%' GROUP BY config_version, outcome, status ORDER BY config_version")
print("V7.x outcome/status breakdown:")
for r in cur.fetchall():
    print(f"  {r['config_version']}: outcome={repr(r['outcome'])}, status={r['status']}, count={r['cnt']}")

# Check all distinct outcomes
print()
cur.execute("SELECT DISTINCT outcome FROM edge_signals WHERE outcome IS NOT NULL")
print("All distinct non-NULL outcomes:")
for r in cur.fetchall():
    print(f"  outcome={repr(r['outcome'])}")

# Check a sample
print()
cur.execute("SELECT config_version, outcome, status, symbol FROM edge_signals WHERE config_version LIKE '%7.%' AND outcome IS NOT NULL LIMIT 5")
print("Sample V7.x with non-NULL outcome:")
for r in cur.fetchall():
    print(f"  {dict(r)}")
    
conn.close()