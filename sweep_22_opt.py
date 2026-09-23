#!/usr/bin/env python3
"""Full re-optimization sweep for 22.0 (LONG) and 22.1 (SHORT) on real majors signals."""
import sys, sqlite3, json
sys.path.insert(0, '/home/hermes/BacktestingMCP')
import backtest_7x_sweep as M

majors = json.load(open('/home/hermes/BacktestingMCP/data/major_symbols.json'))['symbols_bare']

ORIG = M.SWEEP
M.SWEEP = {
    'atr_stop_mult': [1.5, 2.0, 2.5, 3.0, 4.0],
    'rr_ratio':      [1.2, 1.5, 2.0, 2.5, 3.0, 4.0],
    'max_hold_hours': [24, 48, 72, 144],
}
conn = sqlite3.connect('/home/hermes/BacktestingMCP/data/crypto.db')
for cfg in ('22.0', '22.1'):
    sig_meta, results = M.run_config(conn, cfg, symbol_whitelist=majors)
    print('=' * 70)
    print(f'Config {cfg}  replayable={len(sig_meta)}  (MAJORS-ONLY)')
    ranked = sorted(results.items(), key=lambda kv: kv[1]['ev_r'], reverse=True)
    reliable = [kv for kv in ranked if kv[1]['total'] >= 40]
    pick = reliable if reliable else ranked
    print('mult  rr   hold  n     W/L/F       WR%    EV_R')
    shown = 0
    for (mult, rr, mh), res in pick:
        lr = res['losses']
        wr = f"{res['wr']:5.1f}" if res['total'] else '  n/a'
        flag = ' <--' if shown == 0 else ''
        print(f"{mult:<5} {rr:<4} {mh:<5} {res['total']:<6} {res['wins']}/{lr}/{res['flats']:<3} {wr}  {res['ev_r']:+.3f}{flag}")
        shown += 1
        if shown >= 18:
            break
    print('   ...worst (lowest EV):')
    for (mult, rr, mh), res in ranked[-3:]:
        print(f"{mult:<5} {rr:<4} {mh:<5} {res['total']:<6} {res['wins']}/{res['losses']}/{res['flats']:<3} {res['wr']:5.1f}  {res['ev_r']:+.3f}")
M.SWEEP = ORIG
conn.close()