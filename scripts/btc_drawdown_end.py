#!/usr/bin/env python3
"""Clarify: does BTC END below 69k, and what's the median max drawdown in 60d?
Distinguishes 'touched at some point' from 'closes November there'."""
import sys, os, sqlite3, datetime, statistics as st
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DB="data/crypto.db"; CUR=78800.0; H=60
def daily(sym):
    c=sqlite3.connect(DB)
    rows=c.execute("SELECT timestamp,close FROM market_data WHERE symbol=? AND timeframe='1h' AND timestamp>? ORDER BY timestamp",
                   (sym, datetime.datetime(2015,1,1).timestamp())).fetchall(); c.close()
    d={datetime.datetime.utcfromtimestamp(ts).date():cl for ts,cl in rows}
    dates=sorted(d); return dates,[d[x] for x in dates]
def lows(sym):
    c=sqlite3.connect(DB)
    rows=c.execute("SELECT timestamp,low FROM market_data WHERE symbol=? AND timeframe='1h' AND timestamp>? ORDER BY timestamp",
                   (sym, datetime.datetime(2015,1,1).timestamp())).fetchall(); c.close()
    out={}
    for ts,lo in rows:
        day=datetime.datetime.utcfromtimestamp(ts).date(); out[day]=min(out.get(day,lo),lo)
    return out
dates,closes=daily("BTCUSDT"); lows_d=lows("BTCUSDT"); n=len(dates)
tch={}; end={}; dd=[]
for S in range(n-H):
    seg_low=[lows_d.get(dates[k],closes[k]) for k in range(S+1,S+H+1)]
    seg_close=[closes[k] for k in range(S+1,S+H+1)]
    maxdd=-((min(seg_low)-closes[S])/closes[S])*100
    dd.append(maxdd)
    for lbl,price in [("69k",69000),("65k",65000)]:
        if min(seg_low)<=price: tch[lbl]=tch.get(lbl,0)+1
        if seg_close[-1]<=price: end[lbl]=end.get(lbl,0)+1
tot=len(dd)
print(f"Across all {tot} historical 60-day windows (BTC, 2017-2026):")
print(f"  P(touches <=69k at some point): {100*tch['69k']/tot:.1f}%")
print(f"  P(ENDS the 60-day window <=69k): {100*end['69k']/tot:.1f}%")
print(f"  P(touches <=65k at some point): {100*tch['65k']/tot:.1f}%")
print(f"  P(ENDS <=65k): {100*end['65k']/tot:.1f}%")
print(f"  Median max drawdown from start within 60d: {st.median(dd):.1f}%")
print(f"  P(any drawdown >12% in 60d): {100*sum(1 for d_ in dd if d_>12)/tot:.1f}%")
print(f"  P(any drawdown >20% in 60d): {100*sum(1 for d_ in dd if d_>20)/tot:.1f}%")