#!/usr/bin/env python3
"""
7.x robust validation — OHLC signal GENERATOR (vectorized).

Generates LONG/SHORT entries from 1h OHLC using each 7.x config's
OHLC-derivable filters (ADX, RSI band, ATR%, volume-relative) + a momentum
trend proxy, then sweeps exit geometry (atr_stop_mult, rr_ratio, max_hold).

Indicators computed ONCE per symbol with numpy; exits evaluated for all
combos per signal by walking forward bars (vectorized bar advance per signal).
Answer: does the recommended 4.0/2.5 exit survive 6yr regimes + deep high-caps?

Usage:
  python backtest_7x_generator.py [cfg] [SYM1,SYM2] [start_YYYY-MM-DD] [end]
"""
import sqlite3, sys, json
import numpy as np
from itertools import product

DB = '/home/hermes/BacktestingMCP/data/crypto.db'
CONFIGS = ['7.2', '7.3', '7.4', '7.5', '7.6', '7.8']
SWEEP = {
    'atr_stop_mult': [1.5, 2.0, 2.5, 3.0, 4.0],
    'rr_ratio':      [1.0, 1.2, 1.5, 2.0, 2.5],
    'max_hold_hours': [0, 24, 48],
}

class OHLC:
    def __init__(self, ts, o, h, l, c, v):
        self.ts=np.array(ts); self.o=np.array(o); self.h=np.array(h)
        self.l=np.array(l); self.c=np.array(c); self.v=np.array(v,dtype=float)
        self.n=len(ts)

def load_ohlc(conn, symkey):
    rows = conn.execute(
        "SELECT timestamp,open,high,low,close,volume FROM market_data "
        "WHERE timeframe='1h' AND symbol=? ORDER BY timestamp", (symkey,)).fetchall()
    if not rows: return None
    return OHLC(*([r[i] for r in rows] for i in range(6)))

def compute_indicators(ch, period=14):
    """Vectorized RSI, ADX, ATR arrays (numpy) over full series."""
    c=ch.c; h=ch.h; l=ch.l
    n=c.size
    # rolling mean helper
    def roll_mean(x, win):
        out=np.full(n,np.nan)
        if n<win: return out
        csum=np.zeros(n+1); csum[1:]=np.cumsum(x)
        out[win-1:]= (csum[win:]-csum[:-win])/win
        return out
    # ATR
    prev_c=np.roll(c,1); prev_c[0]=c[0]
    tr=np.maximum(h-l, np.maximum(np.abs(h-prev_c), np.abs(l-prev_c)))
    atr_arr=roll_mean(tr, period)
    # RSI (Wilder-lite: SMA of gains/losses)
    rsi_arr=np.full(n,np.nan)
    if n>period:
        diff=np.diff(c)
        gain=np.maximum(diff,0).astype(float); loss=np.maximum(-diff,0).astype(float)
        avg_g=roll_mean(np.concatenate([[0.0],gain]), period)
        avg_l=roll_mean(np.concatenate([[0.0],loss]), period)
        with np.errstate(divide='ignore',invalid='ignore'):
            rs=np.where(avg_l==0, np.inf, avg_g/avg_l)
            rsi_arr=np.where(avg_l==0, 100.0, 100-100/(1+rs))
    # ADX
    adx_arr=np.full(n,np.nan)
    if n>period*2:
        up=h-np.roll(h,1); dn=np.roll(l,1)-l
        up[0]=0.0; dn[0]=0.0
        plus=np.where((up>dn)&(up>0),up,0.0)
        minus=np.where((dn>up)&(dn>0),dn,0.0)
        ps=roll_mean(plus,period); ms=roll_mean(minus,period); ts=roll_mean(tr,period)
        with np.errstate(divide='ignore',invalid='ignore'):
            pdi=np.where(ts==0,0.0,100*ps/ts)
            mdi=np.where(ts==0,0.0,100*ms/ts)
            s=pdi+mdi
            adx_arr=np.where(s==0,0.0,100*np.abs(pdi-mdi)/np.where(s==0,1,s))
    return rsi_arr, adx_arr, atr_arr

def gen_signals(ch, cfg, start_ts, end_ts, rsi_arr, adx_arr, atr_arr, warmup=100, emit_every=1):
    min_adx=cfg.get('min_adx',0); min_rsi=cfg.get('min_rsi',0); max_rsi=cfg.get('max_rsi',0)
    min_atr_pct=cfg.get('min_atr_pct',0); min_volrel=cfg.get('min_volume_relative',0)
    c=ch.c; n=ch.n; ts=ch.ts
    sigs=[]; last_emit=-emit_every
    # EMA50 proxy
    ema=np.full(n,np.nan)
    if n>50:
        ema[49:]=np.convolve(c,np.ones(50)/50,mode='valid')
    for i in range(warmup, n):
        if ts[i]<start_ts or ts[i]>end_ts: continue
        if min_adx>0 and (np.isnan(adx_arr[i]) or adx_arr[i]<min_adx): continue
        if (min_rsi>0 or max_rsi>0):
            r=rsi_arr[i]
            if np.isnan(r): continue
            if min_rsi>0 and r<min_rsi: continue
            if max_rsi>0 and r>max_rsi: continue
        if min_atr_pct>0:
            av=atr_arr[i]
            if np.isnan(av) or c[i]<=0 or av/c[i]*100<min_atr_pct: continue
        if min_volrel>0 and i>=20:
            avg=ch.v[i-20:i].mean()
            if avg>0 and ch.v[i]/avg<min_volrel: continue
        if i>=50 and not np.isnan(ema[i]):
            direction='LONG' if c[i]>ema[i] else 'SHORT'
            if i-last_emit<emit_every: continue
            last_emit=i
            sigs.append((i,direction))
    return sigs

def sim_exit_sequential(ch, entry_idxs, directions, atr_arr, mult, rr, max_hold):
    """Sequential per-symbol simulation: walk the timeline, open a position at
    each qualifying entry bar ONLY when flat, resolve to SL/TP/max-hold, then
    advance past the exit (no overlapping positions). Returns (W,L,F).

    This yields realistic, independent trade counts — unlike firing a signal
    on every bar, which produces ~250k overlapping and meaningless samples.
    """
    w=0; l=0; f=0
    mhb=int(max_hold) if max_hold and max_hold>0 else None
    n=ch.n; c_arr=ch.c; o_arr=ch.o; h_arr=ch.h; l_arr=ch.l
    # pointer into entry list
    eptr=0
    bar=0
    ne=len(entry_idxs)
    while bar < n and eptr < ne:
        idx=entry_idxs[eptr]
        if idx < bar:
            eptr+=1; continue
        if idx > bar:
            bar=idx
        av=atr_arr[idx]
        if np.isnan(av) or av<=0:
            eptr+=1; continue
        d=directions[eptr]
        entry=c_arr[idx]
        if d=='LONG':
            sl=entry-av*mult; tp=entry+av*mult*rr
        else:
            sl=entry+av*mult; tp=entry-av*mult*rr
        exit_bar=None; kind=None
        k=idx+1
        while k<n:
            sl_hit=(l_arr[k]<=sl) if d=='LONG' else (h_arr[k]>=sl)
            tp_hit=(h_arr[k]>=tp) if d=='LONG' else (l_arr[k]<=tp)
            if sl_hit and tp_hit:
                if abs(o_arr[k]-sl)<=abs(o_arr[k]-tp): kind='LOSS'
                else: kind='WIN'
                break
            if tp_hit: kind='WIN'; break
            if sl_hit: kind='LOSS'; break
            if mhb and (k-idx)>=mhb: kind='FLAT'; break
            k+=1
        if kind is None:
            kind='FLAT'; exit_bar=n-1
        else:
            exit_bar=k
        if kind=='WIN': w+=1
        elif kind=='LOSS': l+=1
        else: f+=1
        # advance past exit and this entry
        bar=exit_bar+1
        # skip all entries before bar
        while eptr<ne and entry_idxs[eptr]<bar:
            eptr+=1
    return w,l,f

def main():
    conn=sqlite3.connect(DB)
    args=sys.argv[1:]
    sel_cfg=args[0] if args and args[0] in CONFIGS else None
    cfg_list=[sel_cfg] if sel_cfg else CONFIGS
    syms=None
    for a in args:
        if ',' in a: syms=[s for s in a.split(',') if s.strip()]; break
    if syms is None:
        _nk=[a for a in args if a not in cfg_list and not(len(a)==10 and a[0]=='2') and a.isupper()]
        if _nk: syms=[_nk[0]]

    from datetime import datetime as _dt, timezone as _tz
    dates=[a for a in args if len(a)==10 and a[0]=='2']
    start_ts=int(_dt.strptime(dates[0],'%Y-%m-%d').replace(tzinfo=_tz.utc).timestamp()) if dates else int(_dt(2024,1,1,tzinfo=_tz.utc).timestamp())
    end_ts=int(_dt.strptime(dates[1],'%Y-%m-%d').replace(tzinfo=_tz.utc).timestamp()) if len(dates)>1 else int(_dt.now(_tz.utc).timestamp())

    from src.edge_scanner.scoring_config import ALL_CONFIGS
    cfg_params={}
    for v in cfg_list:
        c=ALL_CONFIGS.get(v)
        if c:
            cfg_params[v]={'rr_ratio':c.rr_ratio,'atr_stop_mult':c.atr_stop_mult,'min_adx':c.min_adx,
                           'min_rsi':c.min_rsi,'max_rsi':c.max_rsi,'min_atr_pct':c.min_atr_pct,
                           'min_volume_relative':c.min_volume_relative}

    # resolve symbols
    ohlc_cache={}
    def get_ohlc(sym):
        if sym not in ohlc_cache:
            for k in (sym+'USDT', sym+'USDTUSDT'):
                ch=load_ohlc(conn,k)
                if ch: ohlc_cache[sym]=(k,ch); break
            else: ohlc_cache[sym]=(None,None)
        return ohlc_cache[sym]

    # Precompute indicators per (symbol,). Reuse across configs.
    ind_cache={}
    def get_ind(sym):
        if sym not in ind_cache:
            _,ch=get_ohlc(sym)
            if ch: ind_cache[sym]=compute_indicators(ch)
            else: ind_cache[sym]=None
        return ind_cache[sym]

    out={}
    for cfg in cfg_list:
        print(f"\n{'='*70}\n Config V{cfg}\n{'='*70}")
        agg={}
        total=0
        syms_used=[]
        for sym in (syms or []):
            k,ch=get_ohlc(sym)
            if not ch: continue
            ind=get_ind(sym)
            if ind is None: continue
            rsi_arr,adx_arr,atr_arr=ind
            sigs=gen_signals(ch,cfg_params[cfg],start_ts,end_ts,rsi_arr,adx_arr,atr_arr)
            total+=len(sigs); syms_used.append(sym)
            entry_idx=[s[0] for s in sigs]; dirs=[s[1] for s in sigs]
            for mult,rr,mh in product(*SWEEP.values()):
                key=(mult,rr,mh)
                if key not in agg: agg[key]=[0,0,0]
                w,l,f=sim_exit_sequential(ch,entry_idx,dirs,atr_arr,mult,rr,mh)
                agg[key][0]+=w; agg[key][1]+=l; agg[key][2]+=f
        combos=[]
        for (mult,rr,mh),(w,l,f) in agg.items():
            n=w+l+f
            if n<40: continue
            wr=w/(w+l)*100 if (w+l)>0 else 0
            ev=(w*rr-l)/n
            combos.append({'mult':mult,'rr':rr,'hold':mh,'n':n,'w':w,'l':l,'f':f,'wr':round(wr,1),'ev':round(ev,3)})
        combos.sort(key=lambda x:x['ev'],reverse=True)
        print(f"  entries: {total}  symbols: {syms_used}")
        print(f"  {'mult':<5}{'rr':<5}{'hold':<6}{'n':<7}{'W/L/F':<12}{'WR%':<8}{'EV_R':<7}")
        for c in combos[:12]:
            w_l_f=f"{c['w']}/{c['l']}/{c['f']}"
            print(f"  {c['mult']:<5}{c['rr']:<5}{c['hold']:<6}{c['n']:<7}{w_l_f:<12}{c['wr']:<8}{c['ev']:<7}")
        if combos:
            out[cfg]={'best':combos[0],'entries':total,'top5':combos[:5],'symbols':syms_used,
                      'window':[str(_dt.fromtimestamp(start_ts,_tz.utc))[:10],str(_dt.fromtimestamp(end_ts,_tz.utc))[:10]]}
    conn.close()
    with open('/home/hermes/BacktestingMCP/results/7x_generator_validate.json','w') as f:
        json.dump(out,f,indent=2,default=str)
    print("\nSaved results/7x_generator_validate.json")

if __name__=='__main__':
    main()