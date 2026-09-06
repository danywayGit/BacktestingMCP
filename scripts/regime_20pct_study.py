#!/usr/bin/env python3
"""
Historical regime study: after a fast ~20%+ up-move following a downtrend on
BTC/ETH, what happens 5/10/15/20 days later? Distinguish 'continue up' vs
'correction', and measure how much (median + percentiles).

Definition per episode:
  - Pre-condition (long downtrend): trailing 45-day return < -8%  OR a flat/
    weak baseline (|45d return| small) -- we report both variants below.
  - Rally: close rises >= +18% from a local low to a peak (the 'entry').
    Rapid rally window duration <= 21 days.
  - Forward returns measured from the rally peak at +5/+10/+15/+20 days.
"""
import sys, os, sqlite3, datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DB = "data/crypto.db"
RALLY_PCT = 0.18      # +18%+ move
MAX_RALLY_DAYS = 21   # rally completes within 3 weeks
PRE_DOWN_PCT = -0.08  # trailing 45d baseline < -8% (downtrend precondition)
HORIZONS = [5, 10, 15, 20]

def daily_closes(sym):
    c = sqlite3.connect(DB)
    rows = c.execute("SELECT timestamp, close FROM market_data "
                     "WHERE symbol=? AND timeframe='1h' AND timestamp>? "
                     "ORDER BY timestamp", (sym, datetime.datetime(2019,1,1).timestamp())).fetchall()
    c.close()
    daily = {}
    for ts, cl in rows:
        d = datetime.datetime.utcfromtimestamp(ts).date()
        daily[d] = cl  # last close of the day wins
    return [(d, v) for d, v in sorted(daily.items())]

def analyze(sym, pre_down_pct):
    daily = daily_closes(sym)
    dates = [d for d, _ in daily]
    closes = [v for _, v in daily]
    idx = {d: i for i, d in enumerate(dates)}
    n = len(closes)
    episodes = []
    i = 0
    while i < n - 20:
        # find a local low
        low = closes[i]
        # rally: peak within MAX_RALLY_DAYS reaching >= +RALLY_PCT from low
        best_j = None
        for j in range(i, min(n, i + MAX_RALLY_DAYS)):
            if (closes[j] - low) / low >= RALLY_PCT:
                best_j = j
                break
        if best_j is None:
            i += 1
            continue
        # preclude: trailing 45d return before the low must be < pre_down_pct
        if i >= 45:
            base = closes[i-45]
            pre_ret = (low - base) / base
        else:
            pre_ret = None  # insufficient history
        if pre_ret is not None and pre_ret < pre_down_pct:
            peak = closes[best_j]
            peak_date = dates[best_j]
            fw = {}
            for h in HORIZONS:
                k = best_j + h
                if k < n:
                    fw[h] = (closes[k] - peak) / peak
                else:
                    fw[h] = None
            episodes.append(dict(
                low_date=str(dates[i]), low=round(low,0),
                peak_date=str(peak_date), peak=round(peak,0),
                rally_pct=round((peak-low)/low*100,1),
                pre45_pct=round((pre_ret)*100,1) if pre_ret is not None else None,
                **{f"fw{h}": (round(v*100,1) if v is not None else None) for h,v in fw.items()}
            ))
            i = best_j + 5  # skip past the peak to avoid overlapping episodes
        else:
            i += 1
    # Stats per horizon
    print(f"\n{'='*70}\n{sym}  — episodes after +{RALLY_PCT*100:.0f}% rally, "
          f"pre-condition 45d < {pre_down_pct*100:.0f}% (downtrend)\n{'='*70}")
    print(f"Found {len(episodes)} episodes\n")
    if not episodes:
        return
    # print episodes
    hdr = f"{'peak_date':11}{'rally%':>7}{'pre45%':>8}" + "".join(f"{h:>6}d" for h in HORIZONS)
    print(hdr)
    for e in episodes:
        row = f"{e['peak_date']:11}{e['rally_pct']:>7}{e['pre45_pct'] if e['pre45_pct'] is not None else '':>8}"
        for h in HORIZONS:
            v = e.get(f"fw{h}")
            row += f"{('' if v is None else v):>7}"
        print(row)
    print()
    import statistics as st
    for h in HORIZONS:
        vals = [e[f"fw{h}"] for e in episodes if e[f"fw{h}"] is not None]
        if not vals:
            continue
        # classification: price above peak at horizon = continued up; below = below peak
        up = [v for v in vals if v >= 0]
        down = [v for v in vals if v < 0]
        nup, ndn = len(up), len(down)
        print(f"+{h:>2}d: n={len(vals):>2}  ↑continue {nup:>2} ({nup/len(vals)*100:.0f}%)  "
              f"↓below-peak {ndn:>2} ({ndn/len(vals)*100:.0f}%)   "
              f"median {st.median(vals):>6.1f}% | p25 {sorted(vals)[len(vals)//4]:>6.1f} p75 {sorted(vals)[3*len(vals)//4]:>6.1f}  "
              f"mean {st.mean(vals):>6.1f}%")
        # downside tail: how deep are the corrections
        dlows = sorted(down)
        if dlows:
            print(f"        correction depth: median loss {st.median(dlows):.1f}%, worst {dlows[0]:.1f}%")

for sym in ["BTCUSDT", "ETHUSDT"]:
    analyze(sym, PRE_DOWN_PCT)

print("\nNote: 'continue' meaning price >= peak (still green from peak); "
      "'below-peak' = price under the rally peak (a correction/retrace).")