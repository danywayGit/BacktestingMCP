#!/usr/bin/env python3
"""
Behavior-matched regime study (NOT price-matched).

Criteria for an 'episode':
  1. LONG DOWNTREND: trailing ~270d (9 months) baseline, net change <= -15%
     (a real, sustained downtrend).
  2. SHARP RALLY: from a local LOW, price rises >= +20% within <= 3 calendar
     days (the 'spike').
  3. Measure what happens AFTER the spike peak:
       - CONSOLIDATION: days price stays within +/-8% of the peak before a
         decisive move.
       - CORRECTION depth: max pullback FROM PEAK within 60 days, expressed as
         % of the spike gain retraced AND at Fibonacci levels.
       - '50% given back' flag: does the retrace reach 50% of the spike? 61.8%?
       - OR 'minimal pullback' path: does price keep rising with only a 5-10%
         give-back (spike fully held)?
  Classification at 60d: continue-up (close > peak) vs corrected (close < peak)
  and the max drawdown from peak in the window.

We report: P(correction beyond 50% retrace), P(>61.8% retrace = full give-back),
median time-to-trough, median max-retrace-% of spike, and P(5-10% only).
"""
import sys, os, sqlite3, datetime, statistics as st
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DB="data/crypto.db"

DOWN_270D = -0.15     # 9-month trailing net change <= -15% (downtrend)
RALLY_PCT = 0.20      # >= +20%
MAX_SPIKE_DAYS = 3    # moves <= 3 calendar days
WINDOW = 60           # observe 60 days after spike peak
FIB_RETRACE = [0.382, 0.50, 0.618]

def daily_series(sym):
    c=sqlite3.connect(DB)
    rows=c.execute("SELECT timestamp,close FROM market_data WHERE symbol=? AND timeframe='1h' AND timestamp>? ORDER BY timestamp",
                   (sym, datetime.datetime(2015,1,1).timestamp())).fetchall(); c.close()
    days={}
    for ts,cl in rows:
        d=datetime.datetime.utcfromtimestamp(ts).date(); days[d]=cl
    dates=sorted(days)
    return dates, [days[d] for d in dates]

def analyze(sym):
    dates, closes = daily_series(sym)
    n=len(closes)
    episodes=[]
    i=0
    while i < n:
        # local low candidate
        low_i = i
        low = closes[low_i]
        # find spike peak within MAX_SPIKE_DAYS
        peak_j=None
        for j in range(low_i+1, min(n, low_i+MAX_SPIKE_DAYS+1)):
            if (closes[j]-low)/low >= RALLY_PCT:
                peak_j=j; break
        if peak_j is None:
            i+=1; continue
        # 9-month downtrend precondition: closes[low_i] low relative to 270d prior
        if low_i >= 270:
            base_270 = closes[low_i-270]
            net = (low - base_270)/base_270
        else:
            # use earliest available as baseline
            base_270 = closes[0]; net = (low - base_270)/base_270
        if net <= DOWN_270D:
            peak=closes[peak_j]; spike=peak-low
            # forward window
            endk=min(n-1, peak_j+WINDOW)
            seg=[closes[k] for k in range(peak_j, endk+1)]
            # consolidation: find first k where |close-peak|/peak > 5% -> define 'break'
            break_day=None
            for k in range(1, len(seg)):
                if abs(seg[k]-peak)/peak > 0.06:
                    break_day=k; break
            # max pullback from peak within window
            trough=min(seg); trough_k=seg.index(trough)+peak_j
            max_retrace=(peak-trough)/spike   # fraction of spike retraced
            # min pullback = how far it held (max high beyond peak)
            max_high=max(seg)
            max_gain=(max_high-peak)/peak
            end_close=closes[endk]
            continue_up = end_close>=peak
            # classify retrace to fib
            fib_hit={f: (max_retrace>f) for f in FIB_RETRACE}
            minimal = max_retrace<=0.10  # held spike, gave back <=10% of spike... but use <=8%? use 0.10
            episodes.append(dict(
                date=str(dates[peak_j]), peak=round(peak,0),
                spike_pct=round(spike/low*100,1), net270=round(net*100,1),
                consol_days=break_day, trough_day=trough_k-peak_j,
                max_retrace=round(max_retrace,2), max_gain=round(max_gain*100,1),
                continue_up=continue_up, end_close=round(end_close,0),
                f38=fib_hit[0.382], f50=fib_hit[0.50], f618=fib_hit[0.618],
                minimal=minimal,
            ))
            i = peak_j + 1  # advance past spike
        else:
            i += 1
    # Aggregate
    print(f"\n{'='*72}\n{sym}: {len(episodes)} episodes def'd by long downtrend(9mo<-15%) + sharp {RALLY_PCT*100:.0f}% rally in {MAX_SPIKE_DAYS}d\n{'='*72}")
    if not episodes:
        print("  none"); return
    hdr=f"{'peak_date':11}{'spike%':>7}{'9mo%':>6}{'consol':>7}{'trough':>7}{'retrace':>9}{'maxgain':>8}{'endDir':>7}"
    print(hdr)
    for e in episodes:
        print(f"{e['date']:11}{e['spike_pct']:>7}{e['net270']:>6}{e['consol_days'] if e['consol_days'] is not None else '-':>7}"
              f"{e['trough_day']:>7}{e['max_retrace']:>9.1f}{str(e['max_gain'])+'%':>8}{'UP' if e['continue_up'] else 'DOWN':>7}")
    N=len(episodes)
    print()
    print("Aggregates:")
    print(f"  P(continue up, close>=peak @60d): {100*sum(e['continue_up'] for e in episodes)/N:.0f}%")
    for f in FIB_RETRACE:
        print(f"  P(retrace beyond {f*100:.1f}% of spike): {100*sum(1 for e in episodes if e['max_retrace']>f)/N:.0f}%")
    print(f"  P(minimal pullback, gave back <=10% of spike): {100*sum(e['minimal'] for e in episodes)/N:.0f}%")
    print(f"  Median max retrace (% of spike given back): {st.median(e['max_retrace'] for e in episodes)*100:.0f}%")
    print(f"  Median trough arrival (days after peak): {st.median(e['trough_day'] for e in episodes):.0f}d")
    con=[e['consol_days'] for e in episodes if e['consol_days'] is not None]
    if con: print(f"  Median consolidation (days before >6% break): {st.median(con):.0f}d")

for sym in ["BTCUSDT","ETHUSDT"]:
    analyze(sym)