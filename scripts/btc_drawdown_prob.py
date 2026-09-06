#!/usr/bin/env python3
"""
Probability that BTC drops to 65-69k from ~78.8k within 60 days (now -> Nov).

We measure, over all historical rolling 60-day windows: P(min low in next 60d
<= target). Targets are the drawdowns implied from current ~78.8k:
  78.8k * (1 - d) = target.
We also break out two conditioning states:
  - ALL states (unconditional)
  - 'after rally' state: starting point is within ~20 days of a >=12% up-move
    (closest analogue to right now).
"""
import sys, os, sqlite3, datetime, statistics as st
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DB="data/crypto.db"

CUR = 78800.0          # BTC ~ as of now (Sep 1, 2026)
HORIZON = 60           # days
TARGETS = {  # name -> price
    "69k": 69000,
    "67k": 67000,
    "65k": 65000,
}

def daily(sym):
    c=sqlite3.connect(DB)
    rows=c.execute("SELECT timestamp,close FROM market_data WHERE symbol=? AND timeframe='1h' AND timestamp>? ORDER BY timestamp",
                   (sym, datetime.datetime(2015,1,1).timestamp())).fetchall()
    c.close()
    d={}
    for ts,cl in rows:
        day=datetime.datetime.utcfromtimestamp(ts).date()
        d[day]=cl
    dates=sorted(d)
    # also build daily LOW for drawdown-from-peak measurement? For "reaches 65k", use close only
    # is fine (intraday lows are lower but we only have 1h; use 1h lows instead for accuracy)
    return dates, [d[x] for x in dates]

def hourly_lows(sym):
    c=sqlite3.connect(DB)
    rows=c.execute("SELECT timestamp,low FROM market_data WHERE symbol=? AND timeframe='1h' AND timestamp>? ORDER BY timestamp",
                   (sym, datetime.datetime(2015,1,1).timestamp())).fetchall()
    c.close()
    out={}
    for ts,lo in rows:
        day=datetime.datetime.utcfromtimestamp(ts).date()
        out[day]=min(out.get(day, lo), lo)
    return out

dates, closes = daily("BTCUSDT")
lows = hourly_lows("BTCUSDT")
n=len(dates)

# Identify 'after rally' days: starting day S where there is a ~>=12% up move in
# the 5-21 days BEFORE S (i.e. we just saw a fast rally -> entering from a peak).
def is_after_rally(S):
    for k in range(5, min(S,22)):
        # price rose >=12% between (S-k) and S
        if (closes[S]-closes[S-k])/closes[S-k] >= 0.12:
            return True
    return False

results={}
for label,h in [('ALL', HORIZON), ('AFTER_RALLY', HORIZON)]:
    hits={t:0 for t in TARGETS}
    cnts={t:0 for t in TARGETS}
    total_ok=0
    for S in range(n-HORIZON):
        if label=='AFTER_RALLY' and not is_after_rally(S):
            continue
        total_ok+=1
        for t,price in TARGETS.items():
            # does min low over S+1..S+horizon reach price?
            seg=[lows.get(dates[k], closes[k]) for k in range(S+1, S+h+1)]
            if min(seg) <= price:
                hits[t]+=1
    results[label]=dict(hits=hits,total=total_ok)

print(f"BTC now ~${CUR:,.0f}. Probability of reaching target within {HORIZON} days (historical, ~2017-2026):")
print(f"{'condition':12} {'target':>7} {'n':>4} {'P(hit)':>8}")
for label,r in results.items():
    for t,price in TARGETS.items():
        n=r['total']
        pct=100*r['hits'][t]/max(1,r['total'])
        print(f"{label:12} {t:>7} {n:>4} {pct:>7.1f}%")
    print()

# Also direct: from a starting price within +/-10% of CUR, how often does it hit 65-69k in 60d?
print("Conditional on START price within +/-12% of current (~69k-88k):")
for t,price in TARGETS.items():
    hn=ht=0
    for S in range(n-HORIZON):
        start=closes[S]
        if abs(start-CUR)/CUR <= 0.12:
            hn+=1
            seg=[lows.get(dates[k], closes[k]) for k in range(S+1,S+HORIZON+1)]
            if min(seg)<=price:
                ht+=1
    print(f"  P(hit {t}) = {100*ht/max(1,hn):.1f}%  (n={hn})")