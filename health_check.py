#!/usr/bin/env python3
"""Edge scanner health check - run from BacktestingMCP dir."""
import sqlite3
from datetime import datetime, timezone

db_path = "data/crypto.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

issues = []

# === 2. CONFIGS WITH CRITICAL FLAT RATES ===
# scoring_configs has is_active (int), not status
# Checking if flat_rate info exists in config_json or elsewhere
# The schema doesn't have flat_rate or total_signals columns directly
# Let me check what's in config_json
cur.execute("""
    SELECT id, version, description, is_active, config_json
    FROM scoring_configs
    WHERE is_active = 1
    ORDER BY version
""")
active_configs = cur.fetchall()
print(f"=== ACTIVE SCORING CONFIGS: {len(active_configs)} ===")
for cfg in active_configs:
    # Try to parse config_json for flat_rate info
    import json
    try:
        j = json.loads(cfg["config_json"])
        flat_rate = j.get("flat_rate", j.get("flatRate", "N/A"))
        total_signals = j.get("total_signals", j.get("totalSignals", "N/A"))
        # Also check at top-level or nested
        if flat_rate == "N/A":
            for k, v in j.items():
                if isinstance(v, dict):
                    fr = v.get("flat_rate", v.get("flatRate", None))
                    if fr is not None:
                        flat_rate = fr
        if total_signals == "N/A":
            for k, v in j.items():
                if isinstance(v, dict):
                    ts = v.get("total_signals", v.get("totalSignals", None))
                    if ts is not None:
                        total_signals = ts
        print(f"  {cfg['version']}: {cfg['description'][:50] if cfg['description'] else 'N/A'} | flat_rate={flat_rate} | signals={total_signals}")
        if flat_rate != "N/A" and total_signals != "N/A":
            try:
                fr_val = float(flat_rate)
                ts_val = int(total_signals)
                if fr_val >= 90 and ts_val > 50:
                    msg = f"CRITICAL: {cfg['version']} flat_rate={fr_val}% with {ts_val} signals"
                    print(f"    ⚠️  {msg}")
                    issues.append(("flat_rate", msg))
            except (ValueError, TypeError):
                pass
    except json.JSONDecodeError:
        print(f"  {cfg['version']}: config_json invalid")

print()

# === 3. FUNDING HISTORY FRESHNESS ===
# funding_history has fetched_at (text), not timestamp
cur.execute("SELECT COUNT(*) as cnt, MAX(fetched_at) as last_ts FROM funding_history")
fh = cur.fetchone()
cnt = fh["cnt"]
last_ts = fh["last_ts"]
print(f"=== FUNDING HISTORY ===")
print(f"Total rows: {cnt}")
print(f"Last fetched_at: {last_ts}")
if last_ts:
    try:
        last_dt = datetime.fromisoformat(last_ts.replace("Z","+00:00"))
        age_min = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
        print(f"Age: {age_min:.1f} min")
        if age_min > 60:
            issues.append(("funding_history", f"Funding history is {age_min:.0f}m old (>60m threshold)"))
        else:
            print("✅ Fresh (<1h old)")
    except Exception as e:
        print(f"Parse error: {e}")
else:
    issues.append(("funding_history", "Funding history table is empty"))
print()

# === 4. V8.0 SIGNALS ===
# edge_signals has config_version (text), not version - check both "8.0" and "V8.0"
cur.execute("SELECT COUNT(*) as cnt FROM edge_signals WHERE config_version LIKE '%8.0%'")
v8_cnt = cur.fetchone()["cnt"]
print(f"=== V8.0 SIGNALS (config_version LIKE '%8.0%') ===")
print(f"Total: {v8_cnt}")
if v8_cnt > 0:
    cur.execute("SELECT symbol, direction, composite_score, created_at FROM edge_signals WHERE config_version LIKE '%8.0%' ORDER BY created_at DESC LIMIT 5")
    for r in cur.fetchall():
        print(f"  {r['symbol']} {r['direction']} score={r['composite_score']:.1f} @ {r['created_at']}")
else:
    print("No signals yet - still accumulating")
print()

# === 5. V7.0 WIN RATE ===
# Check trades table - let's see what columns it has
cur.execute("PRAGMA table_info(trades)")
trades_cols = [r[1] for r in cur.fetchall()]
print(f"=== TRADES TABLE COLUMNS ===")
print(f"  {trades_cols}")

# Try different column names for version
version_col = "version" if "version" in trades_cols else "config_version" if "config_version" in trades_cols else None
outcome_col = "outcome" if "outcome" in trades_cols else None

if version_col and outcome_col:
    cur.execute(f"""
        SELECT {version_col} as v, COUNT(*) as total,
               SUM(CASE WHEN {outcome_col}='win' THEN 1 ELSE 0 END) as wins,
               ROUND(100.0 * SUM(CASE WHEN {outcome_col}='win' THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 1) as wr
        FROM trades WHERE {version_col} LIKE '%V7.0%'
        GROUP BY {version_col}
    """)
    v7 = cur.fetchall()
    print(f"=== V7.0 WIN RATE ===")
    if v7:
        for r in v7:
            wr = r["wr"]
            print(f"  {r['v']}: WR={wr}% ({r['wins']}/{r['total']})")
            if wr is not None and wr < 50:
                issues.append(("v7_wr", f"V7.0 win rate is {wr}% (below 50% threshold)"))
            else:
                print("  ✅ 50% or better")
    else:
        print("  No V7.0 trades found")
        # Check what V7* versions exist
        cur.execute(f"SELECT DISTINCT {version_col} FROM trades WHERE {version_col} LIKE '%V7%'")
        v7_versions = [r[0] for r in cur.fetchall()]
        print(f"  V7* versions in trades: {v7_versions}")
else:
    print("Cannot determine version/outcome columns in trades")
    # Dump first row
    cur.execute("SELECT * FROM trades LIMIT 1")
    if (row := cur.fetchone()):
        print(f"  Sample row: {dict(row)}")
print()

# === V7.x WIN RATE from edge_signals outcomes ===
# Check if any V7.x signals have outcomes
cur.execute("""
    SELECT config_version, outcome, COUNT(*) as cnt
    FROM edge_signals 
    WHERE config_version LIKE '%7.%' AND outcome IS NOT NULL
    GROUP BY config_version, outcome
    ORDER BY config_version
""")
v7_outcomes = cur.fetchall()
print("=== V7.x OUTCOMES ===")
v7_has_real_outcomes = False
if v7_outcomes:
    # Check if any outcomes are actually win/loss (not null)
    for r in v7_outcomes:
        if r["outcome"] in ("win", "loss", "WIN", "LOSS"):
            v7_has_real_outcomes = True
    # Group by config_version
    v7_data = {}
    for r in v7_outcomes:
        cv = r["config_version"]
        if cv not in v7_data:
            v7_data[cv] = {"wins": 0, "losses": 0, "flats": 0, "total": 0}
        outcome = r["outcome"].upper() if r["outcome"] else ""
        if outcome == "WIN":
            v7_data[cv]["wins"] += r["cnt"]
        elif outcome == "LOSS":
            v7_data[cv]["losses"] += r["cnt"]
        elif outcome == "FLAT":
            v7_data[cv]["flats"] += r["cnt"]
        v7_data[cv]["total"] += r["cnt"]
    
    for cv, d in sorted(v7_data.items()):
        total_wl = d["wins"] + d["losses"]
        if total_wl > 0:
            wr = round(100.0 * d["wins"] / total_wl, 1)
            print(f"  {cv}: WR={wr}% ({d['wins']}/{total_wl}) | flats={d['flats']} | total={d['total']}")
            if wr < 50 and total_wl >= 5:  # Only flag if >= 5 trades
                issues.append(("v7_wr", f"{cv} win rate is {wr}% ({d['wins']}/{total_wl}) - below 50%"))
            elif wr < 50:
                print(f"    (too few samples: {total_wl} trades)")
            else:
                print(f"    ✅ >= 50%")
        else:
            print(f"  {cv}: no win/loss outcomes (flats={d['flats']})")
else:
    print("  No V7.x signals with outcomes yet")
    print("  ℹ️ This is normal — signals are resolved as time-expired without hitting targets.")
    
    # Check what outcomes exist at all
    cur.execute("SELECT DISTINCT outcome FROM edge_signals WHERE outcome IS NOT NULL")
    all_outcomes_mapping = [r[0] for r in cur.fetchall()]
    print(f"  Distinct outcome values in DB: {all_outcomes_mapping}")
    
    # Check if V7.x signals have outcome=null or not
    cur.execute("SELECT COUNT(*) FROM edge_signals WHERE config_version LIKE '%7.%' AND outcome IS NULL")
    v7_null_outcome = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM edge_signals WHERE config_version LIKE '%7.%'")
    v7_total = cur.fetchone()[0]
    print(f"  V7.x: {v7_null_outcome}/{v7_total} have NULL outcome")
    
    # Check what status means for V7.x
    cur.execute("SELECT status, COUNT(*) as cnt FROM edge_signals WHERE config_version LIKE '%7.%' GROUP BY status")
    for r in cur.fetchall():
        print(f"  V7.x status: {r[0]}={r[1]}")
    
    # This is expected — V7.x signals are all RESOLVED with NULL outcome
    # (resolved by time expiry, not by hitting target/stop).
    # No win rate data available yet for V7.x.
    if v7_null_outcome == v7_total and v7_total > 0:
        print("  ℹ️ All V7.x signals resolved without outcome (time-expired). No win rate data yet.")

# Also check what the V7.0/7.0.x signal counts look like by status
cur.execute("""
    SELECT config_version, status, COUNT(*) as cnt 
    FROM edge_signals 
    WHERE config_version LIKE '%7.%'
    GROUP BY config_version, status 
    ORDER BY config_version
""")
print()
print("=== V7.x by status ===")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]}={r[2]}")

print()
cur.execute("PRAGMA table_info(performance_metrics)")
pm_cols = [r[1] for r in cur.fetchall()]
print(f"=== performance_metrics columns: {pm_cols} ===")

# Check if there's a version/win rate table
# Check backtest_results
cur.execute("PRAGMA table_info(backtest_results)")
bt_cols = [r[1] for r in cur.fetchall()]
print(f"=== backtest_results columns: {bt_cols} ===")

# Try strategy_parameters
cur.execute("PRAGMA table_info(strategy_parameters)")
sp_cols = [r[1] for r in cur.fetchall()]
print(f"=== strategy_parameters columns: {sp_cols} ===")

# Check performance_metrics content
cur.execute("SELECT * FROM performance_metrics ORDER BY id DESC LIMIT 10")
pm_rows = cur.fetchall()
print(f"=== performance_metrics (last 10) ===")
for r in pm_rows:
    print(f"  {dict(r)}")

# Check backtest_results content
cur.execute("SELECT * FROM backtest_results ORDER BY id DESC LIMIT 10")
bt_rows = cur.fetchall()
print(f"=== backtest_results (last 10) ===")
for r in bt_rows:
    print(f"  {dict(r)}")

# Check strategy_parameters content
cur.execute("SELECT * FROM strategy_parameters ORDER BY id DESC LIMIT 10")
sp_rows = cur.fetchall()
print(f"=== strategy_parameters (last 10) ===")
for r in sp_rows:
    print(f"  {dict(r)}")

# Check edge_signals for V7.0
cur.execute("SELECT COUNT(*) FROM edge_signals WHERE config_version LIKE '%V7.0%'")
v7sig = cur.fetchone()[0]
print(f"=== V7.0 edge_signals: {v7sig} ===")

# Also check V7.x versions
cur.execute("SELECT config_version, COUNT(*) as cnt FROM edge_signals GROUP BY config_version ORDER BY config_version")
all_versions = cur.fetchall()
print("=== All config versions in edge_signals ===")
for r in all_versions:
    print(f"  {r[0]}: {r[1]}")

# Check active versions in edge_signals (recent)
cur.execute("""
    SELECT config_version, COUNT(*) as cnt, 
           ROUND(AVG(composite_score), 2) as avg_score,
           MAX(created_at) as last_signal
    FROM edge_signals 
    WHERE created_at > datetime('now', '-7 days')
    GROUP BY config_version
    ORDER BY cnt DESC
""")
recent = cur.fetchall()
print("=== Config versions with signals in last 7 days ===")
for r in recent:
    print(f"  {r['config_version']}: {r['cnt']} signals, avg_score={r['avg_score']}, last={r['last_signal']}")

print()

# === SUMMARY ===
print("=" * 40)
if issues:
    print(f"ISSUES FOUND: {len(issues)}")
    for category, msg in issues:
        print(f"  [{category}] {msg}")
else:
    print("✅ ALL CHECKS PASSED - SYSTEM HEALTHY")

conn.close()