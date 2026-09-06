#!/usr/bin/env python3
"""
7.x config parameter sweep via OHLC signal-replay.

For each REAL logged signal of an enabled 7.x config, we load forward OHLC
bars for its symbol, compute ATR(14) at entry, then for each parameter combo
(atr_stop_mult, rr_ratio, max_hold_hours) simulate which exit fires first:
  SL  = entry ∓ ATR*mult      (long: below, short: above)
  TP  = entry ∓ ATR*mult*rr   (long: above, short: below)
  max_hold = force exit at close after N hours if neither hit.

This re-labels WIN/LOSS/FLAT under each parameter set — a faithful re-test of
exit geometry, using the actual signals the configs produced.

Usage: python backtest_7x_sweep.py [config_list] [symbols]
"""
import sqlite3, datetime, sys, json, collections, math
from itertools import product

DB = '/home/hermes/BacktestingMCP/data/crypto.db'
ATR_PERIOD = 14
WARMUP = 60  # bars of history before entry for ATR

# Enabled 7.x configs (per Didier: all 7.x except disabled 7.0, 7.7)
CONFIGS = ['7.2', '7.3', '7.4', '7.5', '7.6', '7.8']

SWEEP = {
    'atr_stop_mult': [1.5, 2.0, 2.5, 3.0, 4.0],
    'rr_ratio':      [1.0, 1.2, 1.5, 2.0, 2.5],
    'max_hold_hours': [0, 24, 48],  # 0 = no max hold (wait indefinitely)
}

def load_ohlc(conn, symbol_key):
    """Return dict ts->(open,high,low,close) + sorted ts list for a symbol."""
    rows = conn.execute(
        "SELECT timestamp,open,high,low,close FROM market_data "
        "WHERE timeframe='1h' AND symbol=? ORDER BY timestamp",
        (symbol_key,)).fetchall()
    if not rows:
        return None
    ts = [r[0] for r in rows]
    o = [r[1] for r in rows]; h=[r[2] for r in rows]
    l = [r[3] for r in rows]; c=[r[4] for r in rows]
    return {'ts': ts, 'open':o, 'high':h, 'low':l, 'close':c}

def compute_atr(ch, idx, period=ATR_PERIOD):
    """ATR(14) at bar idx (uses bars [idx-period, idx])."""
    if idx < period:
        return None
    trs = []
    for i in range(idx-period, idx):
        h,l,c = ch['high'][i], ch['low'][i], ch['close'][i-1] if i>0 else ch['close'][i]
        tr = max(h-l, abs(h-c), abs(l-c))
        trs.append(tr)
    return sum(trs)/period

def sim_signal(ch, entry_idx, direction, entry_price, atr_mult, rr, max_hold):
    """Simulate one signal. Returns ('WIN'|'LOSS'|'FLAT', exit_price, exit_idx_off, reason)."""
    if entry_idx is None or entry_idx < 0:
        return ('SKIP', None, 0, 'no_entry_bar')
    atr = compute_atr(ch, entry_idx)
    if atr is None or atr <= 0:
        return ('SKIP', None, 0, 'no_atr')
    # price at entry = close of entry bar (or next bar open). Use close of entry bar.
    entry_actual = ch['close'][entry_idx]
    # SL/TP from stored signal entry price geometry: use entry_price as given
    if direction.upper() == 'LONG':
        sl = entry_price - atr*atr_mult
        tp = entry_price + atr*atr_mult*rr
    else:
        sl = entry_price + atr*atr_mult
        tp = entry_price - atr*atr_mult*rr
    max_hold_bars = int(max_hold) if max_hold and max_hold>0 else None

    start = entry_idx + 1
    for k in range(start, len(ch['ts'])):
        o,h,l,c = ch['open'][k],ch['high'][k],ch['low'][k],ch['close'][k]
        # Check SL then TP using within-bar intrabar logic (conservative:
        # if both extremes touched, assume SL hit first for longs? We resolve
        # by which limit is touched — use oracle: whichever is closer to open.)
        if direction.upper() == 'LONG':
            sl_hit = l <= sl
            tp_hit = h >= tp
        else:
            sl_hit = h >= sl
            tp_hit = l <= tp
        # Both touched same bar: decide by relative distance from open
        if sl_hit and tp_hit:
            dist_sl = abs(o - sl)
            dist_tp = abs(o - tp)
            if dist_sl <= dist_tp:
                return ('LOSS', sl, k-start, 'sl')
            else:
                return ('WIN', tp, k-start, 'tp')
        if tp_hit:
            return ('WIN', tp, k-start, 'tp')
        if sl_hit:
            return ('LOSS', sl, k-start, 'sl')
        # max hold
        if max_hold_bars and (k - start) >= max_hold_bars:
            return ('FLAT', c, k-start, 'maxhold')
    # exhausted data
    return ('FLAT', ch['close'][-1], len(ch['ts'])-1-start, 'enddata')

def run_config(conn, cfg, symbol_whitelist=None):
    """Load signals for a config, map to OHLC, run all sweep combos."""
    q = f"""
        SELECT symbol, direction, entry_price, created_at, outcome, composite_score
        FROM edge_signals
        WHERE config_version=? AND entry_price>0 AND created_at IS NOT NULL
    """
    params=[cfg]
    if symbol_whitelist:
        q += f" AND symbol IN ({','.join('?'*len(symbol_whitelist))})"
        params += symbol_whitelist
    sigs = conn.execute(q, params).fetchall()

    # symbol -> ohlc cache
    ohlc_cache = {}
    def get_ohlc(sym):
        if sym not in ohlc_cache:
            for cand in (sym+'USDT', sym+'USDTUSDT'):
                ch = load_ohlc(conn, cand)
                if ch: ohlc_cache[sym] = (cand, ch); break
            else:
                ohlc_cache[sym] = (None, None)
        return ohlc_cache[sym]

    # Pre-index: for each signal find entry bar (created_at rounded to hour)
    sig_meta = []
    for sym, direction, entry_price, created_at, outcome, score in sigs:
        _, ch = get_ohlc(sym)
        if not ch:
            continue
        try:
            t = datetime.datetime.fromisoformat(str(created_at).replace(' ','T'))
            ets = int(t.timestamp())
        except Exception:
            continue
        # find bar index with ts == ets (or nearest before)
        idx = -1
        # binary search
        import bisect
        i = bisect.bisect_left(ch['ts'], ets)
        if i < len(ch['ts']) and abs(ch['ts'][i]-ets) <= 3600:
            idx = i
        elif i>0:
            idx = i-1
        if idx is None or idx < WARMUP:
            continue
        sig_meta.append({'sym':sym,'dir':direction,'entry':entry_price,
                         'idx':idx,'outcome_signal':outcome,'score':score,'created':created_at})

    # apply sweep
    combos = list(product(SWEEP['atr_stop_mult'], SWEEP['rr_ratio'], SWEEP['max_hold_hours']))
    results = {}
    for mult, rr, mh in combos:
        w=l=f=0; rets=[]; samples=collections.Counter()
        for s in sig_meta:
            _, ch = get_ohlc(s['sym'])
            res, exit_px, bars, reason = sim_signal(
                ch, s['idx'], s['dir'], s['entry'], mult, rr, mh)
            if res=='WIN': w+=1
            elif res=='LOSS': l+=1
            elif res=='FLAT': f+=1
            else: continue
        n = w+l+f
        if n==0:
            continue
        wr = w/(w+l)*100 if (w+l)>0 else 0
        # EV in R-multiples per trade (flats dilute EV): wins give +RR, losses −1
        ev_r = (w*rr - l*1.0)/n if n>0 else 0
        results[(mult,rr,mh)] = dict(total=n, wins=w, losses=l, flats=f,
                                     wr=wr, ev_r=round(ev_r,3))
    return sig_meta, results

def main():
    import logging
    conn = sqlite3.connect(DB)
    symbols = sys.argv[2].split(',') if len(sys.argv)>2 and sys.argv[2].strip() else None
    selected = [c for c in CONFIGS if not sys.argv[1:] or c == sys.argv[1]] or CONFIGS

    out = {}
    for cfg in selected:
        print(f"\n{'='*70}\n  Config V{cfg}\n{'='*70}", flush=True)
        sig_meta, results = run_config(conn, cfg, symbols)
        print(f"  Signals replayable: {len(sig_meta)}")
        # rank by EV_R over combos with >=30 resolved (reliable sample)
        ranked = sorted(
            [(k,v) for k,v in results.items() if v['total']>=30],
            key=lambda kv: kv[1]['ev_r'], reverse=True)
        if not ranked:
            ranked = sorted(results.items(), key=lambda kv: kv[1]['ev_r'], reverse=True)
        print(f"  {'mult':<5}{'rr':<5}{'hold':<6}{'n':<6}{'W/L/F':<12}{'WR%':<8}{'EV_R':<8}")
        for (mult,rr,mh),v in ranked[:12]:
            w_l_f = f"{v['wins']}/{v['losses']}/{v['flats']}"
            print(f"  {mult:<5}{rr:<5}{mh:<6}{v['total']:<6}{w_l_f:<12}{v['wr']:<8.1f}{v['ev_r']:<8.2f}")
        if len(ranked)>12:
            print("  ... (worst 3)")
            for (mult,rr,mh),v in ranked[-3:]:
                w_l_f = f"{v['wins']}/{v['losses']}/{v['flats']}"
                print(f"  {mult:<5}{rr:<5}{mh:<6}{v['total']:<6}{w_l_f:<12}{v['wr']:<8.1f}{v['ev_r']:<8.2f}")
        if ranked:
            out[cfg] = {'best': {'mult':ranked[0][0][0],'rr':ranked[0][0][1],'hold':ranked[0][0][2],**ranked[0][1]},
                        'n_signals': len(sig_meta),
                        'top5': [{'mult':k[0],'rr':k[1],'hold':k[2],**v} for k,v in ranked[:5]],
                        'worst': {'mult':ranked[-1][0][0],'rr':ranked[-1][0][1],'hold':ranked[-1][0][2],**ranked[-1][1]} if ranked else None}
    conn.close()
    import json as _j
    with open('/home/hermes/BacktestingMCP/results/7x_sweep_results.json','w') as f:
        _j.dump(out, f, indent=2, default=str)
    print("\n\nSaved results/7x_sweep_results.json")

if __name__ == '__main__':
    main()