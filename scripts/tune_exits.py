#!/usr/bin/env python3
"""tune_exits.py — sweep EXIT TYPES (atr_rr vs ema_atr vs pct_rr) over a chosen
strategy's signal, train-60/OOS-40 direction split, cost-adjusted.

Tests whether a richer exit geometry (e.g. an EMA-anchored stop) rescues a
signal whose fixed atr_rr exit was flat/marginal. Uses each exit type's NEW
store semantics (ema_atr anchors the stop under EMA exit_ema). Nothing written
to configs — reports the best (exit × params × sl/rr) combo per strategy/dir.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy_search import load_full, resample, COST, SYMBOLS
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# local ATR (Wilder-ish, matches strategy_search)
def _atr(h,l,c,n=14):
    h=np.asarray(h,float);l=np.asarray(l,float);c=np.asarray(c,float)
    pc=np.roll(c,1);pc[0]=c[0]
    tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    out=np.empty(len(tr));s=tr[0];out[0]=s;a=2.0/(n+1)
    for i in range(1,len(tr)):
        s=a*tr[i]+(1-a)*s;out[i]=s
    return np.concatenate([[tr[0]],out[1:]])

def _ema(x,n):
    x=np.asarray(x,float);out=np.empty_like(x);a=2.0/(n+1);s=x[0];out[0]=s
    for i in range(1,len(x)):
        s=a*x[i]+(1-a)*s;out[i]=s
    return out

from tune_reuse import TYPES as REUSE_TYPES   # the 9 ported signal fns

# Map target -> signal fn: reuse types from tune_reuse, OHLC-core types from ohlc_strategies.
import os as _os
if BASE not in sys.path:
    sys.path.insert(0, BASE)
from src.edge_scanner import ohlc_strategies as OS

TARGET_FNS = dict(REUSE_TYPES)
TARGET_FNS.update(OS.TYPES)  # ohlc_strategies.TYPES covers trend/ma_cross/meanrev/etc.

# strategy configs to re-tune: label -> (tf, signal_fn, param-grid)
# pick the MARGINAL / FLAT ones Didier wants rescued, + a couple strong re-check.
TARGETS = {
    # meanrev is in ohlc_strategies, not tune_reuse; import it
    "25.2_meanrev_LONG": ("meanrev", 1, {"rsi":14,"oversold":30,"overbought":70}, 1),  # 25.2 flat +0.01%
    "25.3_meanrev_SHORT": ("meanrev", 1, {"rsi":14,"oversold":30,"overbought":70}, -1),  # recheck steady
    "ema_rejection_LONG": ("ema_rejection", 4, {"ema200":100,"reject_look":10,"rsi":14,"rsi_ema":9,"rsi_confirm":3,"htf":9}, 1),
    "vwap_meanrev_LONG": ("vwap_meanrev", 1, {"band":50,"mult":2.0,"vol_exh":0.4,"vol_avg":20,"rsi":14,"rsi_floor":35,"rsi_ceil":65}, 1),
    "supertrend_LONG": ("supertrend_adx", 4, {"st_period":7,"st_factor":3.0,"adx":20}, 1),
    "aggr_pullback_SHORT": ("aggr_pullback", 1, {"ema":20,"ema200":200,"pullback":1,"swing":7,"massive":2.0,"atr":14}, -1),
}

# exit types to sweep: label -> compute(stop_dist_factory, ema_series, i, atr_i, entry, dir, mult, rr)
EXITS = {
    "atr_rr": {"ema_anchor": False},
    "ema_atr": {"ema_anchor": True},
    "pct_rr": {"pct": True},
}

def sim_exit(sig, o,h,l,c, a, atr_series, entry_idx, want, sl_mult, rr, ema_f=None, exit_type="atr_rr", pct=0.0):
    """Simulate one trade from entry_idx with the given exit type. Returns (R, next_idx)."""
    d = sig[entry_idx]; n=len(c); i=entry_idx
    entry=c[i]; sd = a[i]*sl_mult if atr_series[i]>0 else entry*0.02*sl_mult
    if exit_type=="ema_atr" and ema_f is not None and np.isfinite(ema_f[i]):
        ema_ref=ema_f[i]
        if d>0:  # LONG stop under EMA
            stop_price=ema_ref - atr_series[i]*sl_mult; sd=max(entry-stop_price, entry*0.0005)
        else:
            stop_price=ema_ref + atr_series[i]*sl_mult; sd=max(stop_price-entry, entry*0.0005)
    elif exit_type=="pct_rr" and pct>0:
        sd=entry*pct/100.0
    tpd=sd*rr
    slp=entry-d*sd; tpp=entry+d*tpd; res=None; j=i+1
    while j<n:
        if d>0:
            if l[j]<=slp: res="L"; break
            if h[j]>=tpp: res="W"; break
        else:
            if h[j]>=slp: res="L"; break
            if l[j]<=tpp: res="W"; break
        j+=1
    if res=="W": r_val=rr-COST/sd
    elif res=="L": r_val=-1.0-COST/sd
    else: r_val=-COST/sd
    return r_val, j

def ev(cube, symbols, tf, sigfn, params, want, sl, rr, exit_type, ema_p, pct):
    tr=oo=0.0; tr_n=oo_n=0
    for s in symbols:
        rows=cube[s]; rb=resample(rows,tf)
        if len(rb)<220: continue
        O=np.asarray([x[1] for x in rb],float);H=np.asarray([x[2] for x in rb],float)
        L=np.asarray([x[3] for x in rb],float);C=np.asarray([x[4] for x in rb],float)
        V=np.asarray([x[5] for x in rb],float)
        sig=sigfn(O,H,L,C,V,params); a=_atr(H,L,C,14)
        ema_f = _ema(C,ema_p) if (ema_p and exit_type=="ema_atr") else None
        res=[]; i=1
        while i<len(C):
            if sig[i]==0 or not np.isfinite(a[i]) or a[i]<=0: i+=1; continue
            d=sig[i]
            if d!=want: i+=1; continue
            r_val,j=sim_exit(sig,O,H,L,C,a,a,i,want,sl,rr,ema_f,exit_type,pct)
            res.append(r_val); i=max(j,i+1)
        if len(res)<20: continue
        n=len(res); k=int(n*0.60)
        tr+=sum(res[:k]);oo+=sum(res[k:]);tr_n+=len(res[:k]);oo_n+=len(res[k:])
    if tr_n<15 or oo_n<15: return None
    return (round(tr/tr_n,4), round(oo/oo_n,4), int(tr_n), int(oo_n))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    ap.add_argument("--out", default="results/_exit_tune.json")
    ap.add_argument("--target", default=None)
    args=ap.parse_args()
    syms=[s.strip().upper() for s in args.symbols.split(",")]
    cube={s:load_full(s) for s in syms}
    OUT=os.path.join(BASE,args.out)
    print(f"symbols={syms} cost={COST*10000:.0f}bp | exit-sweep train60/OOS40\n")

    allrows=[]
    for label,(sfname,tf,params,want) in TARGETS.items():
        if args.target and args.target not in label: continue
        # resolve signal fn
        fn = TARGET_FNS.get(sfname)
        dname="LONG" if want==1 else "SHORT"
        for exit_type in ["atr_rr","ema_atr","pct_rr"]:
            ema_p = 50 if exit_type=="ema_atr" else 0
            pct = 2.0 if exit_type=="pct_rr" else 0.0
            # grid: sl/rr for atr & ema; pct uses fixed 2% + rr grid
            grid = [(1.0,1.5),(1.5,1.5),(1.5,2.0),(2.0,2.0),(2.0,2.5),(3.0,2.0)] if exit_type!="pct_rr" else [(0,1.5),(0,2.0)]
            best=None
            for sl,rr in grid:
                r=ev(cube,syms,tf,fn,params,want,sl,rr,exit_type,ema_p,pct)
                if r is None: continue
                tr_ev,oo_ev,tr_n,oo_n=r
                if tr_ev>0.05 and tr_n>=30:
                    if best is None or oo_ev>best["oos_ev"]:
                        best={"target":label,"dir":dname,"tf":tf,"exit":exit_type,"ema":ema_p,
                              "sl":sl,"rr":rr,"tr_ev":tr_ev,"oos_ev":oo_ev,"tr_n":tr_n,"oo_n":oo_n}
            if best is None:
                print(f"  {label:<26} {dname:<5} {exit_type:<9}: no stable")
                continue
            allrows.append(best)
            tag="STABLE" if best["oos_ev"]>0 else "OVERFIT"
            print(f"  {label:<26} {dname:<5} {exit_type:<9}: sl={best['sl']} rr={best['rr']} "
                  f"train={best['tr_ev']:+.3f} OOS={best['oos_ev']:+.3f} (n={best['tr_n']}/{best['oo_n']}) {tag}")
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    with open(OUT,"w") as f: json.dump(allrows,f,indent=2)
    print(f"\nwrote {OUT}")

if __name__=="__main__":
    main()