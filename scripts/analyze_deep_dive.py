#!/usr/bin/env python3
"""Deeper dive: BTC-only bottom configs, WR-optimized, and per-window consistency."""
import sys, os
sys.path.insert(0, '/home/hermes/BacktestingMCP')
os.chdir('/home/hermes/BacktestingMCP')
import pandas as pd

df = pd.read_csv("research/bottom_strategy_grid_results.csv")
df = df[df['trades'] > 5].copy()
STRAT_PARAMS = {
    "unusual_volume_breakout": ["volume_lookback","volume_multiplier","breakout_lookback"],
    "swing5_keltner_breakout": ["kc_length","kc_mult"],
    "swing2_bb_squeeze": ["bb_length","bb_mult"],
    "vp1_volume_profile_breakout": ["profile_lookback"],
}
df['period_type'] = df['period'].apply(lambda p: 'FULL' if p=='FULL' else 'BOTTOM')
bottom = df[df.period_type=='BOTTOM'].copy()

# BTC-only, top configs at bottoms
print("=== BTC-USDT ONLY: best configs at bottoms (mean across 10 windows) ===")
btc = bottom[bottom.symbol=='BTCUSDT'].copy()
btc_agg = []
for strat, pc in STRAT_PARAMS.items():
    for _,g in btc[btc.strategy==strat].groupby(pc):
        if g['period'].nunique() >= 8:
            btc_agg.append({**{k:g[k].iloc[0] for k in pc}, "strategy":strat,
                "mean_ret":g['return_pct'].mean(), "med":g['return_pct'].median(),
                "pf":g['pf'].mean(), "wr":g['winrate'].mean(), "wins":(g['return_pct']>0).sum(), "n":len(g)})
ba = pd.DataFrame(btc_agg).sort_values('mean_ret', ascending=False)
for _,r in ba.head(10).iterrows():
    print(f"  {r['strategy']:28s} { {k:r[k] for k in ['kc_length','kc_mult','bb_length','bb_mult','profile_lookback','volume_lookback','volume_multiplier'] if k in r and pd.notna(r[k])} }  ret={r['mean_ret']:>6.1f}% pf={r['pf']:>4.2f} wr={r['wr']:>5.1f}% won_in={r['wins']}/10")

# Per-window consistency of the best config (swing5 kc=20 mult=2.5)
print("\n=== swing5_keltner kc=20,mult=2.5 BTC — per bottom window ===")
best = btc[(btc.strategy=='swing5_keltner_breakout')&(btc.kc_length==20)&(btc.kc_mult==2.5)]
for _,r in best.sort_values('period').iterrows():
    print(f"  {r['period']}: ret={r['return_pct']:>7.1f}%  pf={r['pf']:>5.2f}  trades={r['trades']:.0f}")

# WR-optimized configs at bottoms (all symbols)
print("\n=== HIGHEST WIN-RATE configs at bottoms (mean WR, >=8 windows) ===")
btm_agg_wr = []
for strat, pc in STRAT_PARAMS.items():
    sub = bottom[bottom.strategy==strat]
    for _,g in sub.groupby(pc):
        if g['period'].nunique() >= 8:
            btm_agg_wr.append({**{k:g[k].iloc[0] for k in pc}, "strategy":strat,
                "wr":g['winrate'].mean(), "ret":g['return_pct'].mean(), "pf":g['pf'].mean(), "n":len(g)})
bw = pd.DataFrame(btm_agg_wr).sort_values('wr', ascending=False)
for _,r in bw.head(10).iterrows():
    print(f"  {r['strategy']:28s} wr={r['wr']:>5.1f}%  ret={r['ret']:>6.1f}%  pf={r['pf']:>4.2f}")

# Compare: is the bottom-window edge stable vs random? Show buy&hold base
print("\n=== Buy&Hold baseline at bottoms (avg across windows/symbols) ===")
print(f"  BOTTOM buy&hold mean: {bottom['buy_hold'].mean():.1f}%  |  FULL buy&hold mean: {df[df.period_type=='FULL']['buy_hold'].mean():.1f}%")