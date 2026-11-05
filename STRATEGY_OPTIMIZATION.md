# Strategy Optimization — Process, Inventory, Filter Design (living doc)

> **Purpose:** One file for the "find better strategies / optimize existing ones"
> workflow. Read this before any config-tuning session: it holds the process,
> a field-by-field inventory of the live configs (excluding disabled), the
> Step-2 filter design (market-cap tier + volatility band), and the todo.
>
> **Cross-references:** `SIGNAL_QUALITY_TRACKER.md` (the OOS validation results —
> this doc's "gate"), `edge-scanner-config-backtesting` skill (methodology),
> `GPU_MCP_START.md` (how to start the GPU server).

---

## 1. The process (edge comes from the ENTRY)

**Non-negotiable:** exit tuning (stop/RR/hold) shapes an edge, never creates
one. Two distinct jobs:

- **Optimize an existing config** = tune parameters on its OWN real
  `edge_signals`. Loop: inventory → pick ONE knob → sweep exits → 3-layer gate.
- **Find a NEW strategy** = find a new *entry trigger* (new data source /
  signal logic). Any candidate enters the same gate before it can route.

Every change goes through the **3-layer validation gate** (see
SIGNAL_QUALITY_TRACKER): regime robustness → own-signal OOS (60/40) →
regime-stratified (win up AND down + ≥60% symbols). Only a survivor is bankable.
A scope/param change **invalidates all prior realized stats** — judge post-change
signals only.

**Rule:** sweep / validate / get Didier's approval BEFORE editing
`scoring_config.py`. Disabled configs are never re-enabled on a headline; they
pass the gate first.

---

## 2. Inventory — live (enabled) configs (Nov 2026)

Source truth: `src/edge_scanner/scoring_config.py` (working tree, NOT the docs
table / git HEAD). 67 configs total: **39 enabled, 28 disabled**. Active config
that generates alerts: **1.4**.

### What's actually set across the 39 enabled configs

**Direction** — only 4 are single-direction; the other 35 run BOTH:
- LONG-only: 14.0, 14.0.0.1 (BTC), 14.0.1.1 (ETH)
- SHORT-only: 14.1
- everything else `allowed_directions=['LONG','SHORT']`

**Symbol scope** — almost nothing filters by symbol:
- BTC/ETH whitelist: 14.0, 14.1, 14.0.0.1 (BTC only), 14.0.1.1 (ETH only)
- **All other 35 configs: scope = ANY** (empty whitelist, no dyn-majors). The
  `use_dynamic_majors=True` configs (22.x) are all **disabled**.

**Market cap** — the field exists everywhere (`min/max_market_cap_usd`) but is
**0.0 on every config** → NO config filters by market cap today.

**Volatility** — only a *floor* (`min_atr_pct`): ranges 0.15–0.5 across families.
There is NO volatility ceiling / regime band anywhere.

**Volume** — `min_volume_relative` set on ~14 configs (0.3–1.5). Rest unfiltered.

**ADX / RSI** — set mostly on the LLM-evolved families (1.6+, 3.3+): adx 23–25,
RSI bands. Not universal.

**Exit geometry** — ALL ATR-based SL (`atr_stop_mult`), TP = RR ratio:
- 3.0-stop / rr1.0–1.5 (1.4, 1.5, 12.0, 9.0, 18.0, 6.2, 14.x): the validated core
- 3.0-stop / rr2.0–2.5 (1.2, 1.3, 2.x, 3.x, 5.x, part of 1.x): older / unvalidated
- 1.5–2.0-stop / rr2.0–2.5 (1.6–1.16, 3.5): LLM-evolved, tight stop high RR

**Multi-timeframe:** 2.1, 2.2 use ST/MT soft alignment; 15.0 uses binary
`require_multi_timeframe_alignment`. MTF 4h+1h gate (per-config, from research)
is NOT wired as a config field yet — it's an entry-trigger upgrade candidate.

### Inventory gaps (what Didier asked that the pipeline can't do today)
1. **Market-cap tier filter** — field exists but unused; no named tier presets.
2. **Volatility band / regime** (max ATR, not just a floor) — doesn't exist.
3. **Fixed % SL/TP** — only ATR-SL + RR-TP exist. (Note: ATR is the robust
   default per backtests; fixed % is *worse*, but it's a valid option to add.)
4. **Volatility as a symbol-selection dim** — ATR% exists per config but there's
   no universe-level vol sort (e.g. "only symbols with ATR% in band X").

---

## 3. Step 2 — New first-class filter design (pending approval)

Goal: make **market-cap tier** and **volatility band** real, first-class knobs
Mirroring the existing `min_market_cap_usd` / `min_atr_pct` pattern so they drop
into `passes_filters()` and are sweapable.

### 3a. Market-cap tier (`mcap_tier`)
A named preset that maps to min/max USD, so configs say `mcap_tier="mid"`
instead of raw numbers. Proposed presets (matching the existing docstring bands):

| tier | min (USD) | max | intent |
|------|-----------|-----|--------|
| micro | 50M | — | micro-cap gems |
| small | 200M | — | small-cap, still liquid |
| mid | 500M | — | mid-cap balance |
| large | 1B | ∞ | majors |
| mega | 10B | ∞ | blue chips |

`None` = off. Implemented as a helper that derives `min/max_market_cap_usd`;
keeps raw USD fields authoritative. Data already available (`MARKET_CAP` from
altFINS screener row).

### 3b. Volatility band (`min_atr_pct` AND new `max_atr_pct`)
Add `max_atr_pct` so a config can target a volatility **band**, not just a floor:
- low-vol regime only (`min=0, max=0.2`) — reversion/time-of-day styles
- mid-vol band (`min=0.2, max=0.5`) — trend core
- high-vol only (`min=0.5`) — squeeze/breakout styles

Same ATR% (14p ATR/close) source as `min_atr_pct`. 0.0 = no ceiling. This is the
"volatility (filtered by...)" knob Didier asked for.

### 3c. Volatility as a symbol-universe dimension (optional, later)
A config-level universe spec like `universe=[{min_atr_pct:0.3}]` — filters the
*symbol pool* by vol regime before scoring. Kept as a future item; the per-config
`min/max_atr_pct` band covers most of the intent with far less plumbing.

### 3d. Fixed % SL/TP (optional, later)
A `fixed_sl_pct`/`fixed_tp_pct` pair that, when set, overrides ATR-SL/RR-TP for
that config. Documented caveat: backtests across every family show ATR-based
stops dominate fixed % (fixed % doesn't adapt to vol regime); only add if a
specific strategy genuinely wants a static stop.

---

## 4. GPU MCP — how to use it (from this box)

Full start/verify step-by-step is in `GPU_MCP_START.md`. Short form:

1. **Start on the Windows 4090 box** (from home dir, NOT repo, clear PYTHONPATH):
   ```bat
   cd C:\Users\danyw
   set PYTHONPATH=
   python C:\Users\danyw\Documents\Git\DanywayGit\BacktestingMCP\src\mcp\gpu_server.py
   ```
2. **This box is already registered** (`hermes mcp list` → gpu4090 ✓ enabled,
   url `http://192.168.1.7:8765/mcp`).
3. `hermes mcp test gpu4090` → expect ✓ Connected + 5 tools.
4. Tools: `gpu_health`, `run_exit_geometry_sweep`, `run_sweep_async`,
   `sweep_status`, `sweep_result`.
5. **GPU asleep = pipeline never blocks:** CPU fallback
   (`scripts/exit_geometry_sweep.py`) is byte-identical to the GPU math. A sweep
   driver should try-GPU-else-CPU. The GPU box sleeping/off is the ONE hardware
   dependency — analysis jobs run on this box (CPU) regardless.

---

## 5. Todo

Action items tracked here; check off as done. Each completed item gets a dated
log line (see Section 6).

- [ ] **Approved by Didier:** mcap-tier + volatility-band filter design (§3a/3b)
- [x] **DONE (2026-11-05):** `mcap_tier` preset → maps to min/max market cap in `passes_filters()`
- [x] **DONE (2026-11-05):** `max_atr_pct` ceiling → volatility BAND; ATR added to fetched display_types when min/max_atr_pct set
- [x] **DONE (2026-11-05):** new `scripts/oos_filter_sweep.py` — OOS-tests mcap-tier + ATR-band overlays on each config's own signals (see §7)
- [ ] Decide which configs get which filter (data in §7) — pending Didier
- [ ] (later) fixed % SL/TP pair
- [ ] (later) volatility-based universe spec
- [ ] Revalidate the 39 enabled configs through the 3-layer gate; candidates only via approval
- [x] **DONE (2026-11-05):** chart-pattern hypothesis TESTED & REJECTED out-of-sample (see §8) — 10.1 not enabled
- [x] **DONE (2026-11-05):** exit-geometry rescue hypothesis TESTED & REJECTED — apparent "rescue" was a stop-hunt artifact (see §9); exits don't create edge

---

## 7. Step-c result: OOS filter overlay sweep (2026-11-05)

Tool: `scripts/oos_filter_sweep.py` (new). Overlays the two new filters on each
config's OWN `edge_signals`, re-runs the train/OOS gate. Data honesty: ATR%
reconstructed **exactly** from 1h OHLC at each entry bar (only XMR lacks OHLC of
191 tested); market cap is a **current coingecko snapshot proxy** per symbol
(not historical — treat mcap-tier reads as indicative).

Command: `venv/bin/python scripts/oos_filter_sweep.py [--atr-band lo,hi] [--mcap-tier X]`
→ `results/_oos_filter_sweep.json` (has overlay + baseline side-by-side).

### Headline findings
1. **`mcap_tier=large` under-samples the alt-heavy configs.** 1.1-1.6, 5.x, 6.0,
   22.x all drop to `no-data`/`no-oos` (large-cap trims n below the 40-trade
   gate). Retrofitting a mcap tier onto an existing high-volume config **shreds
   its sample** — mcap tier is a universe-shaping knob best set at config design,
   not a cheap retrofit.
2. **ATR band does NOT rescue losers.** Configs that are `OVERFIT`/`BELOW-BRKEVEN`
   on baseline stay that way under high-vol (ATR≥0.3). Volatility banding does not
   manufacture an edge that isn't in the entry.
3. **The same winners stay winning:** 1.0, 6.0, 12.0, 16.0 hold PASS* under both
   overlays and baseline (6.0: 56.5-57% OOS; 12.0: 53.7-56.5%; 1.0: ~54-55%).
   These are the configs to keep; the filters don't change that.

### Takeaways for step b (which configs to apply filters to)
- **Do NOT bolt mcap-tier/ATR-band onto the 35 ANY-scope both-direction configs** —
  the data says it won't rescue them and mcap-tier kills their sample.
- The new knobs are for **building filtered variants/descendants of the proven
  winners** (6.0 / 12.0 / 1.0 / 16.0) — e.g. a `12.1 = 12.0 + mcap_tier=mid +
  atr band` as a NEW candidate, validated before enabling. That respects both
  the "no overwrite, only new versions" rule and the OOS-first gate.
- Proposed next action (pending Didier): create ONE scoped candidate from a
  proven winner (not mutate live), run it through the full 3-layer stack, and
  only enable on PASS.

---

## 8. Chart-pattern hypothesis — TESTED & REJECTED OOS (2026-11-05)

**Source:** altFINS "26 AI-detected patterns" doc (success rates 84% Inverse H&S down
to 56% Pennant; Breakout>Emerging; 1D>1h reliability). Didier re-surfaced it.

**What we found / did:** current code (composite.py:654) used only outlook+stage
(Breakout×2 / Emerging×1) — pattern TYPE was ignored. Built success-weighted
per-cell multipliers (Breakout·Bull 2x, Emerging·Bull 0.5x, Emerging·Bear 0.2x,
Breakout·Bear 1x), created config **10.1** (status=disabled) + wired into
composite.py. Defaults reproduce old flat behavior (verified) so live configs
unchanged.

**Replay test (our OWN edge_signals, n=335 chart-pattern rows):**
- In-sample looked great: Breakout·Bull 64.6% (n181), Falling-Wedge·Breakout 87%.
- **OUT-OF-SAMPLE (later 40% by time): 10.0 flat OOS 47.0%, 10.1 weighted OOS
  48.0%.** The overweighted Breakout·Bull cell OOS = **48.0%** (n102) — not the
  64.6% in-sample. Both BELOW the 52% cost-adjusted breakeven.

**Verdict: REJECTED out-of-sample.** The pattern "edge" was in-sample luck / the
altFINS backtest does not transfer to our realized signals. 10.1 stays DISABLED.
Textbook case of the OOS gate catching a hindsight-winner before money.

**Lesson logged:** never trust a vendor/backtest success-rate table as our edge;
always replay against OUR `edge_signals` through a train/OOS split before buying a
pattern weighting. In-sample cell WR is a hypothesis, not evidence.

---

## 9. Exit-geometry rescue hypothesis — TESTED & REJECTED (2026-11-05)

**Question (Didier):** would a different SL/TP — %, ATR-multiplier, indicator, or
trailing — rescue some of the OOS-failing configs before we disable them?
Requested all 3 SL modes × 3 TP modes from day one, indicator-exits second pass.

**Tool built:** `scripts/oos_exit_geometry_sweep.py` — replay each config's OWN
`edge_signals` against forward 1h OHLC, re-simulate exits under a 180-combo grid
(pct/atr/trail SL × pct/rr/atr TP), train/OOS split, pick best-train by EV, judge
OOS EV>0 → RESCUED or no-rescue.

**Two artifacts found & fixed (v1 → v2):**
1. Stored `edge_price` is a scan-time live price, hours off the OHLC bar at the
   same timestamp; measuring SL/TP against it made every config "rescue" at the
   same combo. **Fixed** by re-anchoring entry = close of the bar at entry_time.
2. 76% ground-truth vs the bot's own stops is a DEFINITION gap, not a bug: the
   bot labels WIN/LOSS by net-move-at-horizon (`forward_return` sign), not by
   first-touch of a real SL/TP order. Resolver models first-touch = realistic.

**The decisive artifact (why the result stays REJECTED even after v2 fixes):**
the winning combos were all `pct=0.5` SL. Audit proved it's a **stop-hunt
double-count**, not edge:
- Median next-bar high-low range = 1.90% ≈ 4× the 0.5% stop → the stop is a
  wicked wick-catcher, not protection.
- The 0.5% stop "touches" on the next bar 53% of the time; of those, 76% later
  reached the +5% target.
- The sim counts the touch as -1R AND then (because it moves on) still lets the
  survivors capture the big +6-8R winners. A real exchange stop would fill you
  OUT at the wick — you can't be both stopped out and still holding. Fantasy EV.

**Verdict: REJECTED.** Real stop distances the bot uses, and honest ATR (1-5x) /
% (1-5%) stops, give `no-rescue` OOS on 12.0/1.6/1.7/7.8 etc. **Exit geometry does
not manufacture edge not in the entry.** Entries are the problem, not exits. This
re-confirms the Oct 2026 cross-symbol exit grid (no profitable combo).

**Lesson:** any exit sweep that "rescues" a majority of configs at one ultra-tight
stop is measuring the stop-hunt artifact — require the winning stop to be wider
than typical bar noise, or a flat `n` of winners composed of genuinely-held runs,
before believing it. In-sample EV is hypothesis, not evidence.

---

## 10. Exit-geometry rescue — REOPENED, VALIDATED & COST-SURVIVING (2026-11-05)

Didier challenged the §9 rejection: "exit MUST matter — a 17x TP is harder to hit
than a 1.6x TP, so WR must move." He was RIGHT and §9 over-claimed (it only judged
the single train-best combo per config, not all combos on OOS).

**Fix:** evaluate EVERY combo's OOS EV directly (not one train-pick), with realistic
stops (>=1% or >=1.0 ATR, excluding the tight-stop artifact). Result — most configs
have a STABLE positive-EV exit (positive on BOTH train and OOS):
- The WR-vs-TP mechanism is real and monotonic: on 6.2, TP 1.5->2->3 ATR moves WR
  41.5%->36.3%->27.6% and EV 0.29->0.40->0.44R. Farther TP = lower WR, per Didier.
- Winning family: 1% stop + far 3xATR target (tight stop, ride the runner), and
  wide 5xATR stop + 3R for several configs.

**Cost-adjusted validation (the honest final gate):** subtracted round-trip taker fee
+ slippage (10-20bps). Every stable exit SURVIVES even at 20bps (worst chain: 7.8
still +8bp; 5.2 +16bp; 1.4 the ACTIVE +29bp; 8 configs >+50bp). Gross edges are
thick (30-280bp), so fees don't erase them.

**Verdict:** the earlier "disable the 39" conclusion is SUPERSEDED (see §9). The
losing configs are MIS-GEARED, not dead — they carry real edge under a different
SL/TP. Next: apply re-gears, re-run the OOS WR-gate at the re-geared exits, and only
disable the true no-survivor. Logged as hypothesis TESTED, VALIDATED (cost-surviving).

**Lesson:** a WR-gate verdict is only valid at the config's CURRENT exit; an exit
sweep that reports only the single train-best combo will mislabel stable exits as
no-rescue. Always sweep the full grid on both train AND OOS, pick stable (both-sided
positive), and subtract cost before declaring a config dead or alive.

**Cost-surviving re-gear table (stable exits, net @20bps):**
| cfg | stable exit | net @20bp | n |
|---|---|---|---|
| 12.0 | 5xATR stop / 3R | +261bp | 72 |
| 1.0 | 5xATR stop / 1.5R | +184bp | 826 |
| 13.0 | 5xATR stop / 3R | +153bp | 103 |
| 10.0 | 5xATR stop / 1.2R | +145bp | 98 |
| 4.1 | 5xATR stop / 3R | +107bp | 387 |
| 6.0 | 4xATR stop / 1.5R | +68bp | 342 |
| 3.5 | 5xATR stop / 1.5R | +56bp | 72 |
| 1.6 | 1% stop / 3xATR | +75bp | 53 |
| 1.7 | 1% stop / 3xATR | +75bp | 40 |
| 3.6 | 1% stop / 3xATR | +48bp | 36 |
| 1.4 (ACTIVE) | 1% stop / 3xATR | +29bp | 1641 |
| 6.2 | 1% stop / 3xATR | +23bp | 394 |
| 2.1 | 1% stop / 3xATR | +25bp | 721 |
| 5.0 | 1% stop / 3R | +23bp | 146 |
| 5.2 | 1% stop / 3R | +17bp | 170 |
| 7.8 | 1% stop / 1.5xATR | +8bp | 139 |

---

## 12. Strategy-type search — FIRST PASS (fixed params, cost-adjusted OOS EV)

Tool: `scripts/strategy_search.py`. Fixed parameters (no tuning — honest "which TYPE
wins" test), 6 TFs × 6 types × 6 exits on BTC/ETH/BNB, 513 candidates, EV = net R
per trade on OOS (later 40%) minus 15bp cost. Full dump `results/_strategy_search.json`.

**ROBUST, in Didier's band (WR 40-60%, rr 1.0-1.9, n>=100 OOS):**
| sym | tf | type | sl | rr | OOS EV | WR | n |
|---|---|---|---|---|---|---|---|
| ETH | 4h | momentum | 3×ATR | 1.5 | +0.335R | 53.0% | 100 |
| BTC | 12h | trend | 1.5×ATR | 1.5 | +0.233R | 49.3% | 144 |
| BNB | 2h | breakout | 3×ATR | 1.5 | +0.218R | 48.5% | 202 |
| BTC | 4h | momentum | 3×ATR | 1.5 | +0.214R | 48.6% | 105 |
| BNB | 2h | momentum | 3×ATR | 1.5 | +0.190R | 47.4% | 192 |
| BNB | 4h | momentum | 3×ATR | 1.5 | +0.185R | 47.0% | 100 |
| BNB | 4h | trend | 3×ATR | 1.0 | +0.173R | 58.3% | 156 |
| ETH | 6h | breakout | 3×ATR | 1.0 | +0.168R | 58.4% | 101 |
| BNB | 4h | rangeratio | 1.5×ATR | 1.5 | +0.144R | 45.8% | 212 |

**Findings:** momentum/breakout/trend on 2h-4h-6h win consistently (EV +0.13 to +0.34R),
thick sample, cost-surviving, cluster across symbols (real edge, not one fluke).
185/513 in band with n>=100. Long-term, concentrated profile Didier wants (5-15 t/wk).

**Next:** shortlist -> TestNet bridge forward-validation (step 5). ONLY forward trades
close the loop. Fixed-param first pass done; second pass = parameter tuning of the
shortlist types once a type+TF shows forward edge.

---

## 13. Strategy-type search — PASS 2 TUNED (params tuned on train, validated OOS)

Tool: `scripts/strategy_search.py --tune`. Parameter-tuned each type on the
AGGREGATED train (60%, all symbols) then validated on the unseen OOS (40%), minus
15bp cost. Output `results/_strategy_tuned_search.json`.

The value of pass 2: it separated real edge from curve-fit. High train-EV configs
that collapsed OOS were overfit (trend ema200 +0.56->-0.14, momentum +0.52->+0.02,
rangeratio +0.60->+0.13). The stable ones held OOS at or above train.

**Stable OOS-validated winners (n>=100 OOS):**
| tf | type | params | sl/rr | train EV | OOS EV | n |
|---|---|---|---|---|---|---|
| 1h | meanrev | rsi14 30/70 | 1.5x/1.5 | +0.21 | +0.24 | 111 |
| 12h | ma_cross | fast10/slow50 | 1.5x/1.5 | +0.19 | +0.24 | 107 |
| 1h | momentum | roc5 thr1.5 vol1.2 | 5x/2.0 | +0.20 | +0.23 | 205 |
| 6h | ma_cross | fast20/slow100 | 1.5x/1.0 | +0.16 | +0.11 | 129 |
| 1h | ma_cross | fast10/slow100 | 3x/1.5 | +0.09 | +0.09 | 533 |
| 1h | breakout | chan30 | 5x/2.0 | +0.17 | +0.08 | 255 |
| 4h | ma_cross | fast20/slow100 | 1.5x/1.5 | +0.17 | +0.05 | 202 |
| 2h | ma_cross | fast10/slow50 | 3x/1.0 | +0.08 | +0.03 | 415 |
| 1h | trend | ema200 | 5x/2.0 | +0.14 | +0.03 | 181 |

Strongest stable family: MA-cross / mean-reversion / momentum on 1h-12h, WR
~50-60% @ rr 1.0-1.5, concentrated (5-15 trades/wk). Top pick: 1h meanrev and 12h
ma_cross both +0.24R OOS. NOTE: pass-2 discovered 1h meanrev & 12h ma_cross that
pass-1 never surfaced (their fixed-param defaults were bad) — tuning was essential.

Next: forward-validate top candidates on the testnet bridge (Didier step 5).

---

## 14. V23/V24 OHLC-native family — REGISTERED (2026-11-05)

Added 7 configs to `scoring_config.py` (backup `scoring_config.py.bak-20261105-v23`),
new `burgertype="ohlc"` signal source on `ScoringConfig` (with `ohlc_strategy`,
`ohlc_tf_hours`, `ohlc_params`). Enabled: **V23.0 (12h MA-cross)**, **V23.1 (1h
mean-reversion)** — Didier-approved forward-test. Disabled candidates: V23.2, V24.0,
V24.1, V24.2, V24.3. Total configs 75, enabled 41.

**Enriched OOS data (BTC+ETH+BNB aggregated, after 15bp cost):**
| ver | name | tf | n | WR | avg win | avg loss | win/loss | exp/tx % | exp/tx R |
|---|---|---|---|---|---|---|---|---|---|
| V23.0 | ma_cross | 12h | 107 | 49.5% | +6.19% | -4.48% | 1.38x | **+0.66%** | +0.18R |
| V23.1 | meanrev | 1h | 111 | 49.5% | +1.88% | -1.38% | 1.36x | **+0.08%** | +0.17R |
| V23.2 | ma_cross | 6h | 129 | 55.8% | +2.90% | -3.21% | 0.90x | +0.05% | +0.06R |
| V24.0 | momentum | 1h | 205 | 40.5% | +10.50% | -7.08% | 1.48x | -0.12% | +0.00R |
| V24.1 | momentum | 2h | 90 | 38.9% | +16.94% | -9.54% | 1.78x | +0.61% | +0.08R |
| V24.2 | breakout | 1h | 248 | 33.9% | +11.23% | -6.03% | 1.86x | **-0.34%** | -0.03R |
| V24.3 | rangeratio | 1h | 145 | 31.0% | +13.54% | -6.83% | 1.98x | **-0.66%** | -0.08R |

**Verdict:** V23.0 (+0.66%/tx, 1.38x win/loss) and V23.1 (+0.08%/tx, 1.36x) are the
positive, cost-surviving, stable picks — correct choices to forward-test. V24.2 and
V24.3 are NEGATIVE after cost (high win-size can't beat low hit-rate + friction) —
flag for re-validation (n or params) before any enable. V24.0/24.1 marginally
positive-to-flat.

**Caveat:** these configs are REGISTERED and now WIRED — the OHLC-native signal
ENGINE (`src/edge_scanner/ohlc_strategies.py`) is built and integrated into
`run_parallel_scan` (multi_version_scan._score_one_config bypasses composite for
`burgertype=ohlc`), the composite path guards OHLC configs out (composite.py
`_is_ohlc_config`), store.py computes SL/TP on the config's OWN timeframe, and
the bridge `CONFIG_PRIORITY` carries 23.0/23.1 (floor 8.0) so they forward to
the TestNet bot. Verified end-to-end: engine resamples trusted 1h→config TF,
fires only on a FRESH signal transition (ma_cross/trend persist otherwise),
and a fired score-8.0 OHLC signal is SELECTED by the bridge (dry-run test, LONG
BTC accepted with eff_RR 1.50). Deliberate design: a 12h strategy should fire
~2-6×/week, not every 15-min scan — the engine logs a signal only when the
latest closed bar records a genuine new cross/touch. Forward-validate V23.0/23.1
on TestNet now; the disabled candidates (23.2, 24.x) will be enabled only after
their siblings prove out live.

---

## 6. Changelog

**2026-11-05** — Exit-geometry §9 verdict REVERSED after Didier's correct challenge
+ full-grid OOS + cost adjustment. Most failing configs have a STABLE cost-surviving
exit (see §10). "Disable the 39" superseded — re-gear first, disable only true
no-survivors.

**2026-11-05** — Exit-geometry rescue hypothesis tested & rejected. Built
`scripts/oos_exit_geometry_sweep.py` (180-combo grid, pct/atr/trail SL × pct/rr/atr
TP, train/OOS by EV). Fixed two artifacts (stale-entry anchoring → re-anchor to
OHLC bar; ground-truth definition gap vs net-move labels). Decisive: the only
"rescue" was pct=0.5 stop — audit proved stop-hunt double-count (0.5% stop < 1.9%
median bar noise, touched 53% next-bar, 76% recovered to target). Honest ATR/%/
stops give no-rescue OOS. Exits don't create edge. See §9.

**2026-11-05** — Chart-pattern hypothesis tested & rejected. Per-cell success
multipliers added to ScoringConfig (defaults = old flat behavior, verified),
config 10.1 (status=disabled) created, wired into composite.py. OOS replay:
10.1 = 48.0% vs 10.0 = 47.0%, both <52% breakeven; the overweighted cell also
48% OOS (vs 64.6% in-sample). Not enabled. See §8.

**2026-11-05** — Step c: new `scripts/oos_filter_sweep.py` drives the config-signal
OOS gate under `mcap_tier` + ATR-band overlays per config. ATR% reconstructed
exactly from 1h OHLC (VerOK on 182/191 of 12.0's signals; only XMR lacks OHLC);
mcap = current-coingecko proxy. Ran `--mcap-tier large` and `--atr-band 0.3,`
across 81 configs → findings in §7. Result: filters don't rescue losers and
mcap-tier under-samples alt configs; winners (1.0/6.0/12.0/16.0) unchanged.

**2026-11-05** — Implementation of the two Step-2 filters in
`src/edge_scanner/scoring_config.py` (working tree; backup
`scoring_config.py.bak-20261105`). Compile OK, 64 lines changed.
- Added `mcap_tier` field (micro/small/mid/large/mega) + module `MCAP_TIERS` map
  + `resolve_mcap_bounds()`; raw USD bounds win over a tier. `passes_filters()`
  now resolves bounds from either/both.
- Added `max_atr_pct` field → volatility band (min AND max). ATR% computed as
  ATR/price from screener data (top-level `lastPrice` or additionalData PRICE).
- `get_required_display_types()` now requests `TR_VS_ATR`+`ATR` when a config sets
  `min_atr_pct` or `max_atr_pct` — otherwise altFINS wouldn't return ATR and the
  band would silently no-op.
- Verified: tier resolution, in-band pass, ceiling/floor rejection, ATR present
  in display_types only when the band is set, baseline configs unchanged,
  `ALL_CONFIGS` (67) / enabled (39) intact, `from_dict` roundtrip of new fields.
  NOTE: these knobs are await-Didier on which configs to actually apply them to —
  implementation is wired but no live config sets `mcap_tier`/`max_atr_pct` yet.

**2026-11-05** — Created this doc. Ran full field-by-field inventory of
`scoring_config.py` (67 configs: 39 enabled / 28 disabled). Drafted §3 filter
design (mcap-tier + volatility band) for approval. Suggested initial sweep:
screen the 35 "ANY-scope, both-direction" enabled configs — most have never been
through the OOS gate and are likely noise.