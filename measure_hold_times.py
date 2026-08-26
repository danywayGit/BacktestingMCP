#!/usr/bin/env python3
"""Precise hold-time measurement: avg, median, percentiles per strategy/direction.

This is MEASURED data (not guesses). Pulls closed trades from the bot server
and resolved signals from the scanner DB, then reports hold-time / time-to-
resolve statistics precisely:
  - per strategy (EdgeScanner, ManualTrading, DCA_Spot)
  - per direction (LONG/SHORT)
  - win vs loss
  - percentiles (p10/p25/p50/p75/p90) so we see the real distribution,
    not just the mean (a few long holds distort the average)

Run: python3 measure_hold_times.py
"""
import base64
import json
import os
import statistics
import sqlite3
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

SSH_HELPER = os.path.expanduser("~/.hermes/scripts/ssh_sudo_run.py")
DB_PATH = "data/crypto.db"


def fetch_executed_raw():
    remote = (
        "/opt/Trading-WebHook-Bot/venv-bot/bin/python -c \""
        "import sqlite3,base64,json;"
        "c=sqlite3.connect('/opt/Trading-WebHook-Bot/exchanges/trades.db');"
        "rows=c.execute('SELECT Symbol,Side,Entry,Exit,StopLoss,TakeProfit,"
        "OpenDate,CloseDate,ProfitLoss,ProfitLossPercent,RiskRewardRatio,"
        "Leverage,StrategyName,AccountType FROM Trades WHERE IsOpen=0').fetchall();"
        "tr=[{'symbol':r[0],'side':r[1],'entry':r[2],'exit':r[3],'sl':r[4],"
        "'tp':r[5],'open':r[6],'close':r[7],'pnl':r[8],'pnl_pct':r[9],"
        "'rr':r[10],'lev':r[11],'strategy':r[12],'acct':r[13]} for r in rows];"
        "print(base64.b64encode(json.dumps(tr).encode()).decode())\""
    )
    cp = subprocess.run(["python3", SSH_HELPER, remote],
                        capture_output=True, text=True, timeout=90)
    for line in (cp.stdout or "").splitlines():
        line = line.strip()
        if len(line) > 50 and line[0] in "W0" and all(
                ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
                for ch in line):
            try:
                return json.loads(base64.b64decode(line).decode())
            except Exception:
                continue
    raise RuntimeError(f"fetch failed: {cp.stdout[-300:]} {cp.stderr[-200:]}")


def hold_hours(t):
    try:
        o = datetime.fromisoformat(t["open"].replace(" ", "T").replace("Z", "+00:00"))
        c = datetime.fromisoformat(t["close"].replace(" ", "T").replace("Z", "+00:00"))
        return (c - o).total_seconds() / 3600
    except Exception:
        return None


def fmt_dist(vals):
    """mean / median / p25-p75 with n."""
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return "no data"
    n = len(vals)
    def pct(p):
        return vals[min(n - 1, int(p / 100 * n))]
    return (f"n={n}  mean={statistics.mean(vals):.1f}h  med={statistics.median(vals):.1f}h  "
            f"p25={pct(25):.1f}  p75={pct(75):.1f}  p90={pct(90):.1f}")


def main():
    print("=" * 78)
    print("PRECISE HOLD-TIME MEASUREMENT — per strategy / direction / win-loss")
    print("=" * 78)
    print("\nNote: mean can be skewed by a few long holds. med + p25/p75/p90")
    print("show the REAL distribution. This is measured data, not guesses.\n")

    d = fetch_executed_raw()
    print(f"Closed trades fetched: {len(d)}\n")
    for t in d:
        t["hold_h"] = hold_hours(t)
        t["direction"] = "LONG" if t["side"] == "BUY" else "SHORT"
        t["result"] = "WIN" if t["pnl"] > 0 else ("LOSS" if t["pnl"] < 0 else "FLAT")

    # ── Executed side: per strategy × direction × result ──
    print("──────────────────────────────────────────────────────────────")
    print("EXECUTED (bot) — hold time in hours")
    print("──────────────────────────────────────────────────────────────")
    strat_dirs = defaultdict(list)
    for t in d:
        strat_dirs[(t["strategy"], t["direction"], t["result"])].append(t["hold_h"])

    for key in sorted(strat_dirs.keys()):
        strat, direction, result = key
        vals = strat_dirs[key]
        all_t = [t for t in d if t["strategy"] == strat and t["direction"] == direction]
        # only show groups with >= 2 trades
        if len(all_t) >= 2:
            print(f"{strat:<14} {direction:<5} {result:<4}: {fmt_dist(vals)}")

    # ── Scanner side: time_to_resolve per direction × outcome ──
    print()
    print("──────────────────────────────────────────────────────────────")
    print("BACKTEST (scanner) — time-to-resolve in hours (sent signals)")
    print("──────────────────────────────────────────────────────────────")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT direction, outcome, time_to_resolve_hours FROM edge_signals
        WHERE webhook_sent_at IS NOT NULL AND outcome IN ('WIN','LOSS')
    """).fetchall()
    conn.close()
    bt = defaultdict(list)
    for direction, outcome, ttr in rows:
        if ttr is not None:
            bt[(direction, outcome)].append(ttr)
    for key in sorted(bt.keys()):
        direction, outcome = key
        print(f"{'EdgeScanner':<14} {direction:<5} {outcome:<4}: {fmt_dist(bt[key])}")

    # ── Direct deviation: executed hold vs backtest ttr, matched direction ──
    print()
    print("──────────────────────────────────────────────────────────────")
    print("DEVIATION (executed hold vs backtest time-to-resolve)")
    print("──────────────────────────────────────────────────────────────")
    for direction in ["LONG", "SHORT"]:
        for outcome in ["WIN", "LOSS"]:
            ex = [t["hold_h"] for t in d
                  if t["direction"] == direction and t["result"] == outcome
                  and t["hold_h"] is not None and t["strategy"] == "EdgeScanner"]
            bts = bt.get((direction, outcome), [])
            if ex and bts:
                ex_m = statistics.mean(ex)
                bt_m = statistics.mean(bts)
                diff = ex_m - bt_m
                flag = ""
                if abs(diff) > 2:
                    flag = "  ⚠️ deviation"
                print(f"  EdgeScanner {direction} {outcome}: "
                      f"executed {ex_m:.1f}h vs backtest {bt_m:.1f}h = {diff:+.1f}h{flag}")

    return 0


if __name__ == "__main__":
    sys.exit(main())