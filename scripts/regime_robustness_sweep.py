#!/usr/bin/env python3
"""regime_robustness_sweep.py — rank edge-scanner configs by REGIME robustness.

The trusted way to judge a config before letting it trade real (esp. prop)
money. Aggregate win-rate is a trap: a config can "look fine on average" yet
bleed out any account in a bad week. This ranks configs by whether they STAY
positive across many rolling windows, and rejects them when the sample is too
thin to trust at all.

Input : BacktestingMCP/data/crypto.db -> edge_signals (realized forward returns
        per config_version for RESOLVED WIN/LOSS signals).
Output: prints + writes results/_regime_sweep.json:
        {config_version: {n, wr, wilson_l, avg_r, maxdd, win_windows, worst_win,
                          verdict, reasons[]}}
        Configs with n >= PASS_MIN_TRADES are eligible for a real verdict; the
        PASS bar is: wilson_l >= PASS_WILSON and win_windows >= PASS_WIN_WINDOWS.

Metrics (all computed per config, signals time-sorted):
  wr           aggregate win rate
  wilson_l     Wilson score 95% lower bound on win rate (penalizes small n)
  avg_r        mean realized forward-return % (signal predictive edge)
  maxdd        peak-to-trough drawdown of cumulative forward-return %
  win_windows  fraction of rolling 50-trade windows (slide 25) whose sum is > 0
  worst_win    worst single window total (%)

Extending the window/slide or the PASS thresholds changes the verdict, so they
are module constants you can tune and re-run — that's the point (iterative).
"""
import sqlite3, json, math, os
from collections import defaultdict

DB_PATH = "/home/hermes/BacktestingMCP/data/crypto.db"
OUT_PATH = "/home/hermes/BacktestingMCP/results/_regime_sweep.json"

# --- verdict bar -------------------------------------------------------------
MIN_TRADES      = 30    # below this: verdict = "insufficient", never judge
PASS_MIN_TRADES = 30    # eligible to PASS
PASS_WILSON     = 40.0  # Wilson lower bound on WR % (>= this to pass)
PASS_MIN_WR     = 45.0  # aggregate WR % floor
PASS_WIN_WINDOWS= 60.0  # fraction of positive rolling windows (>= this to pass)

# --- window config -----------------------------------------------------------
WIN   = 50    # trades per rolling window
SLIDE = 25    # slide between windows


def wilson_lower(w, n, z=1.96):
    if n == 0:
        return 0.0
    p = w / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / den
    return max(0.0, center - half)


def sweep(db_path=DB_PATH, win=WIN, slide=SLIDE):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """SELECT config_version, entry_time, forward_return_pct, outcome
           FROM edge_signals
           WHERE status='RESOLVED' AND outcome IN ('WIN','LOSS')
             AND forward_return_pct IS NOT NULL AND entry_time IS NOT NULL"""
    ).fetchall()
    conn.close()

    by_cfg = defaultdict(list)
    for cv, et, fr, _oc in rows:
        try:
            fr = float(fr)
        except (TypeError, ValueError):
            continue
        by_cfg[str(cv)].append((str(et), fr))

    results = {}
    for cv, items in by_cfg.items():
        items.sort(key=lambda x: x[0])
        frs = [x[1] for x in items]
        n = len(frs)
        w = sum(1 for x in frs if x > 0)
        wl = wilson_lower(w, n)
        avg_r = (sum(frs) / n) if n else 0.0

        # rolling windows
        windows = [sum(frs[i:i + win]) for i in range(0, n - win + 1, slide)]
        win_windows = (sum(1 for x in windows if x > 0) / len(windows)) if windows else 0.0
        worst_win = min(windows) if windows else 0.0

        # cumulative drawdown
        cum = 0.0; peak = 0.0; maxdd = 0.0
        for x in frs:
            cum += x; peak = max(peak, cum); maxdd = max(maxdd, peak - cum)

        reasons = []
        if n < MIN_TRADES:
            verdict = "insufficient"
            reasons.append(f"n={n} < {MIN_TRADES}")
        else:
            if wl * 100 < PASS_WILSON:
                reasons.append(f"wilsonLo={wl*100:.1f} < {PASS_WILSON}")
            if (w / n) * 100 < PASS_MIN_WR:
                reasons.append(f"wr={(w/n)*100:.1f} < {PASS_MIN_WR}")
            if win_windows * 100 < PASS_WIN_WINDOWS:
                reasons.append(f"winWin={win_windows*100:.0f} < {PASS_WIN_WINDOWS}")
            verdict = "PASS" if not reasons else "FAIL"

        results[cv] = {
            "n": n, "wr": round(w / n, 4) if n else 0.0,
            "wilson_l": round(wl, 4), "avg_r": round(avg_r, 3),
            "maxdd": round(maxdd, 1), "win_windows": round(win_windows, 3),
            "worst_win": round(worst_win, 2), "verdict": verdict, "reasons": reasons,
        }

    return results


def render(results):
    """Rank printed output: PASS first (by wilson_l x win_windows), then FAIL, then insufficient."""
    def key(kv):
        m = kv[1]
        prio = {"PASS": 0, "FAIL": 1, "insufficient": 2}[m["verdict"]]
        return (prio, -(m["wilson_l"] * m["win_windows"]))

    rank = sorted(results.items(), key=key)
    lines = [f"{'cfg':>7} {'n':>5} {'WR%':>6} {'wilLo':>6} {'avgR%':>7} {'maxDD%':>8} {'winWin%':>7} {'verdict':<9}"]
    for cv, m in rank:
        lines.append(f"{cv:>7} {m['n']:>5} {m['wr']*100:>6.1f} {m['wilson_l']*100:>6.1f} "
                     f"{m['avg_r']:>7.2f} {m['maxdd']:>8.1f} {m['win_windows']*100:>7.0f} {m['verdict']:<9} {' | '.join(m['reasons'])}")
    return "\n".join(lines)


def main():
    results = sweep()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    passed = [cv for cv, m in results.items() if m["verdict"] == "PASS"]
    print(f"wrote {OUT_PATH} — {len(results)} configs, {len(passed)} PASS")
    print()
    print(render(results))


if __name__ == "__main__":
    main()