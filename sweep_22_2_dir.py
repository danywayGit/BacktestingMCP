#!/usr/bin/env python3
"""Direction-split geometry sweep for V22.2 — is LONG saving the both-direction config?"""
import sys, sqlite3, json
sys.path.insert(0, '/home/hermes/BacktestingMCP')
import backtest_7x_sweep as M

majors = json.load(open('/home/hermes/BacktestingMCP/data/major_symbols.json'))['symbols_bare']
ORIG = M.SWEEP
M.SWEEP = {
    'atr_stop_mult': [1.5, 2.0, 2.5, 3.0],
    'rr_ratio':      [1.2, 1.5, 2.0, 2.5, 3.0],
    'max_hold_hours': [24, 48, 72],
}
conn = sqlite3.connect('/home/hermes/BacktestingMCP/data/crypto.db')
for direction in ('LONG', 'SHORT'):
    # filter signals by direction, then run the sweep per direction
    q = f"""SELECT symbol, direction, entry_price, created_at, outcome, composite_score
            FROM edge_signals WHERE config_version='22.2' AND direction='{direction}'
            AND entry_price>0 AND created_at IS NOT NULL"""
    # monkeypatch run_config query by wrapping query result filtering:
    # simplest: copy run_config logic but add direction to WHERE
    sig_meta, results = M.run_config(conn, '22.2', symbol_whitelist=majors)
    # re-filter by direction from signal meta
    sig_meta_d = [s for s in sig_meta if s['dir']==direction]
    print(f"\n===== 22.2 {direction}  replayable={len(sig_meta_d)} =====")
    # rerun combos only for this direction's subset
    import collections
    from itertools import product
    combo_res = {}
    for mult, rr, mh in product(M.SWEEP['atr_stop_mult'], M.SWEEP['rr_ratio'], M.SWEEP['max_hold_hours']):
        w=l=f=0
        for s in sig_meta_d:
            ch = None
            for cand in (s['sym']+'USDT', s['sym']+'USDTUSDT'):
                c = M.load_ohlc(conn, cand)
                if c: ch=c; break
            if not ch: continue
            res,_,_,_ = M.sim_signal(ch, s['idx'], s['dir'], s['entry'], mult, rr, mh)
            if res=='WIN': w+=1
            elif res=='LOSS': l+=1
            elif res=='FLAT': f+=1
        n=w+l+f
        if n==0: continue
        wr = w/(w+l)*100 if (w+l)>0 else 0
        ev = (w*rr - l)/n
        combo_res[(mult,rr,mh)]=dict(total=n,wins=w,losses=l,flats=f,wr=wr,ev_r=round(ev,3))
    ranked=sorted(combo_res.items(), key=lambda kv: kv[1]['ev_r'], reverse=True)
    rel=[kv for kv in ranked if kv[1]['total']>=20]
    pick=rel if rel else ranked
    print('mult  rr   hold  n     W/L/F       WR%    EV_R')
    for (mult,rr,mh),v in pick[:10]:
        print(f"{mult:<5} {rr:<4} {mh:<5} {v['total']:<6} {v['wins']}/{v['losses']}/{v['flats']:<3} {v['wr']:5.1f}  {v['ev_r']:+.3f}")
M.SWEEP = ORIG
conn.close()