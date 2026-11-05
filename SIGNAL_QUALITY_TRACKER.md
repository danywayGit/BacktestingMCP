# SIGNAL-QUALITY RESEARCH — LIVING TRACKER

> **Purpose:** One file tracking the prop-firm signal-quality research so it
> survives context resets, new sessions, or new Telegram topics. Every entry is
> result-first with the full reproducible command. If a session loses context,
> read this file — it is the source of truth for where we are and what's next.

**Project goal:** find configs + parameters that trade real prop money with
better winrate and less drawdown. Prop routing is STOPPED until a config survives
the full validation stack below.

---

## ⚠️ STATUS (Nov 2026)
**Prop routing:** STOPPED (3 accounts lost: HyroTrader, Velotrade, Bitfunded).
No config changes to live routing until the below candidate is approved by Didier.

**FINAL VALIDATED CANDIDATES (survive every layer: own-signal OOS + dual-regime + symbol breadth):**
- **`6.0`** — PASS_FINAL. UP WR 53% (n445), DOWN WR 60% (n405), both WilsonLo>0,
  109/173 symbols positive. Robust in both directions (long 59% / short 49%).
- **`12.0`** — PASS_FINAL. UP WR 64% (n92), DOWN WR 57% (n90), 50/80 syms positive.
  Long-driven.

**Rejected (do NOT route):** 1.0, 6.2, 16.0 (single-regime / sub-60% symbol breadth
in final gate). Also rejected earlier & confirmed: 1.5, 6.0-under-MTF-gate-view,
14.0, 8.0, and the entire 22.x live family (22.0/22.1/22.2), 1.x–4.x bulk.

---

## THE VALIDATION STACK (run in order; a config must pass all to be bankable)
1. **Regime robustness** (`scripts/regime_robustness_sweep.py`) — survives positive
   across rolling 50-trade windows, not just aggregate WR.
2. **Out-of-sample** (`scripts/oos_signal_sweep.py`) — config's OWN signals: tune
   score-threshold on train 60%, validate WR on unseen 40%, cost-adjusted breakeven
   52% WR + Wilson lower + direction split.
3. **Regime-stratified** (`scripts/oos_regime_stratified.py`) — must win in BOTH
   up and down market windows + >=60% symbols positive. (The less-drawdown proof.)

Insight: a config that looks good in hindsight but fails unseen data / a single
regime is exactly what lost the 3 accounts. The stack catches it before money.

---

## FULL RUN HISTORY (result-first)

### 2026-11-04 (night)
- **Regime sweep, n>=100 + wilsonLo>=40 + WR>=45% + >=60% windows:** 13 PASS
  (1.0,1.1,1.2,1.3,1.4,1.5,1.6,2.0,3.0,3.1,4.0,6.0,14.0). 22.x family ALL FAIL.
- **Cross-symbol exit grid** (`overnight_exit_grid.py`, 12,000 sims, 120 syms):
  NO exit combo is profitable cross-symbol (best +0.0 EV, 47.7% of syms). Exit
  tuning cannot manufacture edge — edge must come from the entry.
- **MTF confirmation test** (`mtf_confirmation_test.py`): as a universal filter,
  barely helps (WR 48.6→49.1%, trades less). As a per-config gate it rescues
  specific configs (8.0→WR68%, 12.0, 14.0) — but see the caveat below.

### 2026-11-05 (day)
- **MTF-gated grid** (`mtf_gated_exit_grid.py`): 4h+1h-confirmed entries made all
  5 target configs positive cross-symbol (8.0 +0.75, 6.0 +0.37...). ⚠️ BUT this
  used a SHARED LONG gate across configs → did NOT rank configs (artifact).
- **Bias diagnostic** (`oos_bias_diagnostic.py`): exposed that the 73/81 "PASS*"
  was a shared-gate artifact + LONG-bias (LONG median EV +0.152 vs SHORT −0.049).
  Not bull-luck (OOS window median drift −36.7%) but direction-lopsided.
- **Config-signal-driven OOS** (`oos_signal_sweep.py`): final sharp ranking by each
  config's OWN signals. Only 6.0 (WR 57%, 200t, two-sided) robustly clears +
  12.0/1.0/6.2/16.0 pass long-biased. Others OVERFIT or below breakeven.
- **Final regime gate** (`oos_regime_stratified.py`): **only 6.0 and 12.0 survive.**
  1.0, 6.2, 16.0 REJECTED.

---

## KEY LESSONS (why we trust the stack, not the headlines)
- Shared synthetic entry gates measure one transform N times — never use to rank
  configs. Always use each config's OWN realized signals (`edge_signals`).
- Direction honesty: report LONG-only, SHORT-only, blended. A long-only pass may
  be regime luck.
- `edge_signals` carries config, symbol, direction, composite_score, forward_return,
  outcome — the source of truth for config-driven validation.
- Stored raw 4h OHLC is inconsistent with 1h — resample 1h→4h ourselves.
- GPU (4090) MCP: CPU fallback (`exit_geometry_sweep.py`) is byte-identical, so a
  sleeping GPU never blocks the pipeline.

---

## COMMANDS (reproduce each result)
```bash
cd /home/hermes/BacktestingMCP
venv/bin/python scripts/regime_robustness_sweep.py   # -> results/_regime_sweep.json
venv/bin/python scripts/overnight_exit_grid.py       # -> results/_overnight_exit_grid.json
venv/bin/python scripts/mtf_confirmation_test.py     # -> results/_mtf_comparison.json
venv/bin/python scripts/mtf_gated_exit_grid.py       # -> results/_mtf_gated_grid.json
venv/bin/python scripts/oos_signal_sweep.py          # -> results/_oos_signal_sweep.json
venv/bin/python scripts/oos_regime_stratified.py     # -> results/_oos_regime_stratified.json
venv/bin/python scripts/oos_sweep_all.py             # (shared-gate, known artifact — diagnostic only)
venv/bin/python scripts/oos_bias_diagnostic.py       # -> results/_oos_bias_diagnostic.json
```

## NEXT STEPS (pending — ask Didier)
- [ ] Approve re-enabling `6.0` (and optionally `12.0`) for prop routing.
- [ ] Wire MTF 4h+1h confirmation as an entry gate on the approved config.
- [ ] Decide exit geometry for the approved config (from the validated grid).