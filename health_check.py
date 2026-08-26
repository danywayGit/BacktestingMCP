#!/usr/bin/env python3
"""Edge scanner health check — run from BacktestingMCP dir.

Checks (mirrors the system-health-check cron prompt):
  1. edge-scan cron freshness (last run output mtime)
  2. Config flat rates > 90% with > 50 resolved signals (from live outcomes)
  3. funding_history freshness (< 1h)
  4. V8.0 signal accumulation
  5. V7.x win rates (>= 50% with >= 10 non-flat trades)
  6. System memory usage (> 80%)
  7. Bot duplicate positions (DB open rows > unique symbols, via SSH)

Exit code 0 = healthy (or "---ALL_HEALTHY---"), 1 = issues found.
Run:  python3 health_check.py
"""
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

DB_PATH = "data/crypto.db"
CRON_OUT = Path.home() / ".hermes" / "cron" / "output" / "7fb16f3b1d9a"  # edge-scan
MEM_THRESHOLD_PCT = 80
FLAT_THRESHOLD_PCT = 90
FLAT_MIN_SIGNALS = 50
WR_THRESHOLD_PCT = 50
WR_MIN_NONFLAT = 10

issues = []
notes = []


def _fmt_ts(ts):
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
    return ts


def check_edge_scan_freshness():
    """1. edge-scan cron: any output in last 15 min = healthy."""
    try:
        if not CRON_OUT.exists():
            issues.append(f"EDGE_SCAN_NO_OUTPUT: {CRON_OUT} missing")
            return
        mds = sorted(CRON_OUT.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not mds:
            issues.append("EDGE_SCAN_NO_OUTPUT: no cron output files")
            return
        latest = mds[0]
        age_min = (datetime.now().timestamp() - latest.stat().st_mtime) / 60
        if age_min > 15:
            issues.append(f"EDGE_SCAN_STALE: last edge-scan output {age_min:.0f} min ago ({latest.name})")
        else:
            notes.append(f"edge-scan OK (last {age_min:.0f} min ago)")
        # Also flag a FAILED marker in the two most recent outputs
        for p in mds[:2]:
            if "FAILED" in p.read_text(errors="replace")[:2000]:
                issues.append(f"EDGE_SCAN_FAILED: {p.name} contains FAILED marker")
    except Exception as e:
        issues.append(f"EDGE_SCAN_ERROR: {e}")


def _enabled_config_versions():
    """Return the set of enabled config versions (real source of truth = code)."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.edge_scanner.scoring_config import ALL_CONFIGS
        return {v for v, c in ALL_CONFIGS.items() if c.status != "disabled"}
    except Exception:
        return None  # cannot determine → don't filter


def check_flat_rates(cur, enabled=None):
    """2. Configs with >90% FLAT and >50 resolved signals (enabled configs only)."""
    try:
        rows = cur.execute("""
            SELECT config_version,
                   COUNT(*) as total,
                   SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) as losses,
                   SUM(CASE WHEN outcome='FLAT' THEN 1 ELSE 0 END) as flats
            FROM edge_signals
            WHERE status='RESOLVED'
            GROUP BY config_version
            ORDER BY total DESC
        """).fetchall()
        flagged = 0
        for r in rows:
            cv, total, wins, losses, flats = r
            # Skip disabled/retired configs — their historical flats are noise.
            if enabled is not None and cv not in enabled:
                continue
            if total > FLAT_MIN_SIGNALS:
                flat_pct = round((flats or 0) / total * 100, 1)
                if flat_pct > FLAT_THRESHOLD_PCT:
                    issues.append(f"HIGH_FLAT: {cv} has {flat_pct}% flat rate ({flats}/{total} signals)")
                    flagged += 1
        if flagged == 0:
            notes.append("no enabled configs with critical flat rates")
    except Exception as e:
        issues.append(f"CONFIG_ERROR: {e}")


def check_funding_freshness(cur):
    """3. funding_history < 1h old."""
    try:
        row = cur.execute("SELECT MAX(fetched_at) FROM funding_history").fetchone()
        if not row or not row[0]:
            issues.append("FUNDING_NO_DATA: funding_history table is empty")
            return
        fetched = _fmt_ts(row[0])
        if fetched is None:
            issues.append(f"FUNDING_PARSE: cannot parse fetched_at {row[0]}")
            return
        age_min = (datetime.now(timezone.utc) - fetched).total_seconds() / 60
        if age_min > 60:
            issues.append(f"FUNDING_STALE: last funding poll was {age_min:.0f} min ago ({row[0]})")
        else:
            notes.append(f"funding fresh ({age_min:.0f} min)")
    except Exception as e:
        issues.append(f"FUNDING_ERROR: {e}")


def check_v8_accumulation(cur):
    """4. V8.0 has accumulated signals."""
    try:
        cnt = cur.execute("SELECT COUNT(*) FROM edge_signals WHERE config_version='8.0'").fetchone()[0]
        if cnt == 0:
            issues.append("V80_NO_SIGNALS: V8.0 has accumulated zero signals")
        else:
            notes.append(f"V8.0 has {cnt} signals")
    except Exception as e:
        issues.append(f"V80_ERROR: {e}")


def check_v7_win_rates(cur, enabled=None):
    """5. V7.x win rates >= 50% with >= 10 non-flat trades (enabled configs only)."""
    try:
        rows = cur.execute("""
            SELECT config_version,
                   COUNT(*) as total,
                   SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) as losses,
                   SUM(CASE WHEN outcome='FLAT' THEN 1 ELSE 0 END) as flats
            FROM edge_signals
            WHERE config_version LIKE '7.%' AND status='RESOLVED' AND outcome IS NOT NULL
            GROUP BY config_version
            ORDER BY config_version
        """).fetchall()
        checked = 0
        for r in rows:
            cv, total, wins, losses, flats = r
            if enabled is not None and cv not in enabled:
                continue
            nonflat = (wins or 0) + (losses or 0)
            if nonflat >= WR_MIN_NONFLAT:
                wr = round((wins or 0) / nonflat * 100, 1)
                checked += 1
                if wr < WR_THRESHOLD_PCT:
                    issues.append(f"V7_WR_LOW: {cv} win rate is {wr}% ({wins}/{nonflat} non-flat) — below 50%")
                else:
                    notes.append(f"{cv} WR {wr}% OK")
        if checked == 0:
            notes.append("no enabled V7.x config has >= 10 non-flat trades yet")
    except Exception as e:
        issues.append(f"V7_ERROR: {e}")


def check_memory():
    """6. System memory usage > 80%."""
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                k, v = line.split(":")
                mem[k] = int(v.strip().split()[0])  # kB
        total = mem["MemTotal"]
        available = mem["MemAvailable"]
        used_pct = round((total - available) / total * 100, 1)
        if used_pct > MEM_THRESHOLD_PCT:
            issues.append(f"MEM_HIGH: memory usage {used_pct}% (> {MEM_THRESHOLD_PCT}%)")
        else:
            notes.append(f"memory {used_pct}% used")
    except Exception as e:
        issues.append(f"MEM_ERROR: {e}")


def check_duplicate_positions():
    """7. Bot DB open rows vs unique symbols (duplicate-position detection).

    Checks the bot's trades.db on the server (via SSH): if the number of
    IsOpen=1 rows exceeds the number of DISTINCT symbols, the bot is stacking
    duplicate positions on the same symbol (wasted concurrent slots).
    See commit ceab5ca (Trading-WebHook-Bot).
    """
    try:
        import subprocess
        helper = os.path.expanduser("~/.hermes/scripts/ssh_sudo_run.py")
        if not os.path.exists(helper):
            notes.append("duplicate-check skipped (no ssh_sudo_run.py)")
            return
        # Runs a short python snippet on the server that prints "<rows> <unique>".
        remote = (
            "cd /opt/Trading-WebHook-Bot && /opt/Trading-WebHook-Bot/venv-bot/bin/python -c "
            "\"import sqlite3;c=sqlite3.connect('exchanges/trades.db');"
            "r=c.execute('SELECT COUNT(*),COUNT(DISTINCT Symbol) FROM Trades WHERE IsOpen=1').fetchone();"
            "print(r[0],r[1]);c.close()\""
        )
        cp = subprocess.run(
            ["python3", helper, remote], capture_output=True, text=True, timeout=60
        )
        # Extract "<rows> <unique>" from combined output.
        from re import search
        m = search(r"(\d+)\s+(\d+)", cp.stdout or "")
        if m:
            rows, unique = int(m.group(1)), int(m.group(2))
            if rows > unique:
                dupes = rows - unique
                issues.append(
                    f"DUP_POSITIONS: bot DB has {rows} open rows but only "
                    f"{unique} unique symbols ({dupes} duplicate positions "
                    f"stacking the same symbol)"
                )
            else:
                notes.append(f"bot positions: {unique} unique / {rows} DB rows")
        else:
            notes.append(f"duplicate-check: unexpected server output (rc={cp.returncode}, out={cp.stdout.strip()[:60]!r})")
    except Exception as e:
        notes.append(f"duplicate-check error: {e}")


def main():
    if not os.path.exists(DB_PATH):
        print("---ISSUES---")
        print(f"DB_MISSING: {DB_PATH} not found (run from BacktestingMCP dir)")
        return 1
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    enabled = _enabled_config_versions()
    if enabled is not None:
        notes.append(f"enabled configs: {len(enabled)}")

    check_edge_scan_freshness()
    check_flat_rates(cur, enabled)
    check_funding_freshness(cur)
    check_v8_accumulation(cur)
    check_v7_win_rates(cur, enabled)
    check_memory()
    check_duplicate_positions()

    conn.close()

    for n in notes:
        print(f"OK: {n}")
    if issues:
        print("---ISSUES---")
        for i in issues:
            print(f"ISSUE: {i}")
        return 1
    print("---ALL_HEALTHY---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
