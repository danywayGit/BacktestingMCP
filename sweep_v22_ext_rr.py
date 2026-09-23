#!/usr/bin/env python3
"""Extended-RR majors-only sweep for V22 configs."""
import sys, sqlite3, json
sys.path.insert(0, '/home/hermes/BacktestingMCP')
import backtest_7x_sweep as M

majors = json.load(open('/home/hermes/BacktestingMCP/data/major_symbols.json'))['symbols_bare']

ORIG = M.SWEEP
M.SWEEP = {
    'atr_stop_mult': [3.0, 4.0, 5.0, 6.0],
    'rr_ratio':      [3.0, 4.0, 5.0, 6.0, 8.0],
    'max_hold_hours': [48, 72, 96, 144, 192],
}
conn = sqlite3.connect('/home/hermes/BacktestingMCP/data/crypto.db')
for cfg in ('22.0', '22.1'):
    sig_meta, results = M.run_config(conn, cfg, symbol_whitelist=majors)
    print('=' * 66)
    print(f'Config {cfg}  replayable={len(sig_meta)}  (MAJORS-ONLY)')
    print('mult  rr   hold  n     W/L/F       WR%    EV_R')
    ranked = sorted(results.items(), key=lambda kv: kv[1]['ev_r'], reverse=True)
    for (mult, rr, mh), res in ranked[:16]:
        lr = res['losses']
        wr = f"{res['wr']:5.1f}" if res['total'] else '  n/a'
        print(f"{mult:<5} {rr:<4} {mh:<5} {res['total']:<6} {res['wins']}/{lr}/{res['flats']:<3} {wr}  {res['ev_r']:+.3f}", flush=True)
    print('   ...worst:')
    for (mult, rr, mh), res in ranked[-3:]:
        print(f"{mult:<5} {rr:<4} {mh:<5} {res['total']:<6} {res['wins']}/{res['losses']}/{res['flats']:<3} {res['wr']:5.1f}  {res['ev_r']:+.3f}")
M.SWEEP = ORIG
conn.close()