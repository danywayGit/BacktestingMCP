#!/usr/bin/env python3
"""
DCA-focused study: after a long downtrend (9mo) + sharp rally, if you waited to
buy instead of buying at the spike peak, did the wait pay?  And does the price
come BACK to a '65k-like' entry level (i.e. deep pullback) or keep running?

Relaxed criteria for more samples:
  - 9-month downtrend: trailing 270d net <= -12%
  - sharp rally: >= +15% within <= 7 days (not the ultra-strict 20%/3d)
We then ask, from the spike peak:
  - Best entry in next 90d: lowest price hit, and how long to wait
  - vs 'buy now': price 90d later
  - P(a lower entry than peak within 90d)  -> waiting got a better fill
  - P(price 90d later >= peak)              -> buying at peak was fine
  - median gain if you bought at peak and held 90d
  - and crucially for DCA at 65k: P(retrace deep enough to re-approach 65k-ish,
    i.e. retrace >=15% the rally, >=20%, >=25% from peak)
"""
import sys, os, sqlite3, datetime, statistics as st
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DB="data/crypto.db"

DOWN_270D=-0.12; RALLY_PCT=0.15; MAX_SPIKE_DAYS=7; WINDOW=90

def series(sym):
    c=sqlite3.connect(DB)
    rows=c.execute("SELECT timestamp,close FROM market_data WHERE symbol=? AND timeframe='1h' AND timestamp>? ORDER BY timestamp",
                   (sym, datetime.datetime(2015,1,1).timestamp())).fetchall(); c.close()
    days={datetime.datetime.utcfromtimestamp(ts).date():cl for ts,cl in rows}
    dates=sorted(days); return dates,[days[d] for d in dates]

def analyze(sym):
    dates,closes=series(sym); n=len(closes)
    eps=[]; i=0
    while i<n:
        low_i=i; low=closes[low_i]; pk=None
        for j in range(low_i+1, min(n,low_i+MAX_SPIKE_DAYS+1)):
            if (closes[j]-low)/low >= RALLY_PCT: pk=j; break
        if pk is None: i+=1; continue
        base=closes[low_i-270] if low_i>=270 else closes[0]
        net=(low-base)/base
        if net<=DOWN_270D:
            peak=closes[pk]; spike=peak-low
            endk=min(n-1,pk+WINDOW)
            seg=closes[pk:endk+1]
            trough=min(seg); tk=seg.index(trough)
            trough_retrace=(peak-trough)/spike
            end_close=closes[endk]
            # lower entry than peak found?
            lower_than_peak = trough < peak  # always (trough<=peak) but meaningful for deep
            retrace_15 = trough_retrace>=0.15
            retrace_20 = trough_retrace>=0.20
            retrace_25 = trough_retrace>=0.25
            eps.append(dict(date=str(dates[pk]), peak=peak, spike=spike/low*100,
                            net=net*100, trough_retrace=trough_retrace, trough_day=tk,
                            end90_close=end_close,
                            buy_hold90=(end_close-peak)/peak*100,  # if bought at peak, held 90d
                            wait_best=retrace_15 and trough_retrace,  # marker
                            r15=retrace_15, r20=retrace_20, r25=retrace_25))
            i=pk+1
        else: i+=1
    N=len(eps)
    print(f"\n{'='*74}\n{sym}: {N} episodes  [9mo<{DOWN_270D*100:.0f}% + sharp>{RALLY_PCT*100:.0f}% in {MAX_SPIKE_DAYS}d]\n{'='*74}")
    if not eps: print("none"); return
    hdr=f"{'peak':10}{'spike%':>7}{'9mo%':>6}{'retra':>6}{'trDay':>6}{'60dhld':>8}"
    print(hdr)
    for e in sorted(eps,key=lambda x:x['date']):
        print(f"{e['date']:10}{e['spike']:>7.1f}{e['net']:>6.1f}{e['trough_retrace']:>6.2f}{e['trough_day']:>6}{e['buy_hold90']:>8.1f}")
    print("\nDCA-relevant aggregates (from spike peak):")
    # Waiting to buy -> better entry?
    print(f"  P(min pullback hit, i.e. retrace>=15% of spike): {100*sum(e['r15'] for e in eps)/N:.0f}%")
    print(f"  P(retrace>=20% of spike): {100*sum(e['r20'] for e in eps)/N:.0f}%")
    print(f"  P(retrace>=25% of spike): {100*sum(e['r25'] for e in eps)/N:.0f}%")
    # Doesn't matter: for DCA the useful number is: after spike peak, how much
    # was given back on a MEDIAN basis, and how often did waiting help.
    outp=[e['buy_hold90'] for e in eps]
    med_retr=st.median([e['trough_retrace'] for e in eps])
    up90=sum(1 for e in eps if e['end90_close']>=e['peak'])
    print(f"  Median max retrace (% of spike given back): {med_retr*100:.0f}%")
    print(f"  Median 90d return if bought at peak (buy&hold): {st.median(outp):+.1f}%")
    print(f"  P(price >= peak at 90d): {100*up90/N:.0f}%")
    # timing
    print(f"  Median days to trough (when waiting for pullback): {int(st.median(e['trough_day'] for e in eps))}d")

for sym in ["BTCUSDT","ETHUSDT"]:
    analyze(sym)