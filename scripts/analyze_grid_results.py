#!/usr/bin/env python3
"""Analyze grid search: rank configs at bottoms vs full period."""
import sys, os
sys.path.insert(0, '/home/hermes/BacktestingMCP')
os.chdir('/home/hermes/BacktestingMCP')
import pandas as pd

df = pd.read_csv("research/bottom_strategy_grid_results.csv")
df = df[df['trades'] > 5].copy()

STRAT_PARAMS = {
    "unusual_volume_breakout": ["volume_lookback","volume_multiplier","breakout_lookback"],
    "resistance_breakout": ["resistance_lookback","level_tolerance","min_touches"],
    "new_local_high_breakout": ["local_high_lookback","min_relative_volume"],
    "bollinger_bands": ["bb_period","bb_std"],
    "macd": ["macd_fast","macd_slow","macd_signal"],
    "swing5_keltner_breakout": ["kc_length","kc_mult"],
    "vp1_volume_profile_breakout": ["profile_lookback"],
    "swing2_bb_squeeze": ["bb_length","bb_mult"],
}

df['period_type'] = df['period'].apply(lambda p: 'FULL' if p=='FULL' else 'BOTTOM')
print(f"Valid rows: {len(df)} | strategies: {df['strategy'].nunique()} | bottom windows: {df.loc[df.period_type=='BOTTOM','period'].nunique()}")

# ---- 1. Strategy level: FULL vs BOTTOM (median return & PF) ----
print("\n" + "="*88)
print("STRATEGY FIT: median return & PF — FULL (2020-26) vs BOTTOM windows (BTC/ETH/SOL/BNB)")
print("="*88)
strategy_summary = []
for strat in df['strategy'].unique():
    fc = df[(df.strategy==strat)&(df.period_type=='FULL')]
    bc = df[(df.strategy==strat)&(df.period_type=='BOTTOM')]
    strategy_summary.append({
        "strategy": strat,
        "full_ret": fc['return_pct'].median(), "full_pf": fc['pf'].median(),
        "btm_ret": bc['return_pct'].median(), "btm_pf": bc['pf'].median(),
        "btm_wr": bc['winrate'].median(), "btm_sharpe": bc['sharpe'].median(),
    })
ss = pd.DataFrame(strategy_summary).sort_values('btm_ret', ascending=False)
for _,r in ss.iterrows():
    print(f"{r['strategy']:30s} FULL ret={r['full_ret']:>7.1f}% pf={r['full_pf']:>4.2f}  |  BOTTOM ret={r['btm_ret']:>7.1f}% pf={r['btm_pf']:>4.2f} wr={r['btm_wr']:>5.1f}% shr={r['btm_sharpe']:>5.2f}")

# ---- 2. Best configs at bottoms (must appear in >=6/10 windows) ----
print("\n" + "="*88)
print("TOP 15 STRATEGY+PARAMS AT BOTTOMS (median, must appear in >=6 of 10 windows)")
print("="*88)
bottom = df[df.period_type=='BOTTOM'].copy()
cfg_rows = []
for strat, params_cols in STRAT_PARAMS.items():
    sub = bottom[bottom.strategy==strat]
    if sub.empty: continue
    g = sub.groupby(params_cols).agg(
        mean_ret=('return_pct','mean'), med_ret=('return_pct','median'),
        mean_pf=('pf','mean'), mean_wr=('winrate','mean'),
        n=('trades','count'), n_sym=('symbol','nunique'), n_periods=('period','nunique')
    ).reset_index()
    g = g[g.n_periods>=6]
    for _,row in g.iterrows():
        p = {k:row[k] for k in params_cols if pd.notna(row[k])}
        cfg_rows.append({**p, "strategy":strat, "mean_ret":row['mean_ret'], "med_ret":row['med_ret'],
                         "mean_pf":row['mean_pf'], "mean_wr":row['mean_wr'], "n_periods":row['n_periods'],
                         "n":row['n']})
cfg = pd.DataFrame(cfg_rows).sort_values('mean_ret', ascending=False)
for i,(_,row) in enumerate(cfg.head(15).iterrows()):
    p = {k:v for k,v in row.items() if k in
         ['volume_lookback','volume_multiplier','breakout_lookback','resistance_lookback','level_tolerance',
          'min_touches','local_high_lookback','min_relative_volume','bb_period','bb_std','macd_fast','macd_slow',
          'macd_signal','kc_length','kc_mult','profile_lookback','bb_length','bb_mult'] and pd.notna(v)}
    print(f"{'%2d'%(i+1)}. {row['strategy']:30s} {p}  mean_ret={float(row['mean_ret']):>7.1f}% pf={float(row['mean_pf']):>4.2f} wr={float(row['mean_wr']):>5.1f}% [{row['n_periods']} bottoms]")

# ---- 3. Best configs on FULL period ----
print("\n" + "="*88)
print("TOP 10 STRATEGY+PARAMS ON FULL PERIOD 2020-2026 (median)")
print("="*88)
full = df[df.period_type=='FULL'].copy()
cfg_rows_full = []
for strat, params_cols in STRAT_PARAMS.items():
    sub = full[full.strategy==strat]
    if sub.empty: continue
    g = sub.groupby(params_cols).agg(
        mean_ret=('return_pct','mean'), med_ret=('return_pct','median'),
        mean_pf=('pf','mean'), mean_wr=('winrate','mean'),
        n=('trades','count'), n_sym=('symbol','nunique')
    ).reset_index()
    for _,row in g.iterrows():
        p = {k:row[k] for k in params_cols if pd.notna(row[k])}
        cfg_rows_full.append({**p, "strategy":strat, "mean_ret":row['mean_ret'], "med_ret":row['med_ret'],
                              "mean_pf":row['mean_pf'], "mean_wr":row['mean_wr'], "n_sym":row['n_sym']})
cfgf = pd.DataFrame(cfg_rows_full).sort_values('mean_ret', ascending=False)
for i,(_,row) in enumerate(cfgf.head(10).iterrows()):
    p = {k:v for k,v in row.items() if k in
         ['volume_lookback','volume_multiplier','breakout_lookback','resistance_lookback','level_tolerance',
          'min_touches','local_high_lookback','min_relative_volume','bb_period','bb_std','macd_fast','macd_slow',
          'macd_signal','kc_length','kc_mult','profile_lookback','bb_length','bb_mult'] and pd.notna(v)}
    print(f"{'%2d'%(i+1)}. {row['strategy']:30s} {p}  mean_ret={float(row['mean_ret']):>7.1f}% pf={float(row['mean_pf']):>4.2f} wr={float(row['mean_wr']):>5.1f}%")

# Save
ss.to_csv('/tmp/strategy_summary.csv', index=False)
cfg.to_csv('/tmp/bottom_cfgs.csv', index=False)
cfgf.to_csv('/tmp/full_cfgs.csv', index=False)
print("\nSaved summaries to /tmp/csv files")