# Liquidation Strategies (V22.x) — Reference

> Hub doc for the liquidation-driven strategies. Mirrors what posts to the
> Telegram "Liquidation (V22)" topic (Edge Scanner group, thread 8043).
> **Last reviewed:** 2026-09-19

## What V22 is

Three direction-specialized configs in `src/edge_scanner/scoring_config.py` that
trade **squeeze cascades** off liquidation imbalances. Liquidation imbalance is the
STAR signal (`liquidation_weight=6.0`); volume confirms (`volume_relative_weight=2.0`).
All other drivers (trend, BB, ATR-expansion, precursors, regime) are **stripped** —
this is the SIMPLE redesign (2026-09-19, per Didier: liquidation + direction + SL/TP/RR
+ volume only).

| Config | Direction | Physics | Params |
|--------|-----------|---------|--------|
| **22.0** Liquidation Squeeze LONG | LONG only | SHORT liqs dominate → short-squeeze → price UP | 4×ATR stop / rr 6.0 |
| **22.1** Liquidation Squeeze SHORT | SHORT only | LONG liqs dominate → long-squeeze → price DOWN | 4×ATR stop / rr 6.0 |
| **22.2** Liquidation Squeeze (agg) | BOTH | any squeeze direction | 6×ATR stop / rr 8.0 (A/B variant) |

All three: `liquidation_weight=6.0`, `volume_relative_weight=2.0`, everything else 0,
`min_precursors=0`, `min_abs_score=5.0`, `use_dynamic_majors=True`, `market_regime_filter=OFF`.

### Trigger (event_watcher.py) — only REAL cascades
`LIQ_VOL_THRESHOLD = $1,000,000` (was $250k) liquidation $ in a 5-min window **AND**
`LIQ_MIN_IMBALANCE = 0.30` (≥30% one-sided) → only fires on genuine squeezes, not
routine liquidation noise. Imbalance sign routes direction: short-heavy (imb>0) →
22.0 LONG; long-heavy (imb<0) → 22.1 SHORT; both → 22.2.

## Pipeline

```
binance_liq_ws.py ──(real-time !forceOrder@arr + REST)──> data/liquidation_snapshot.json
        │
        └─ event_watcher.py ──(liq $ ≥$1M in 5min AND |imbalance|≥30%)──> targeted scan
                │   short-heavy (imb>0) → 22.0 LONG      score_symbol(H1, 30 bars)
                │   long-heavy  (imb<0) → 22.1 SHORT
                │   both                                → 22.2 (agg)
                └─ resolve_due_signals() ──> webhook_bridge ──> bot execution
```

- **Daemons** (keep-alive crons every 5 min): `liq_ws_daemon.sh` (WS feed),
  `event_watcher_ensure.sh` (event scanner). Both no-op if already running.
- **Debounce:** at most one targeted run per symbol per 60 s.
- **Config status:** all three `enabled` (scan + DB log + send).
- **Trigger bar** (2026-09-19): was $250k volume; now requires **≥$1M volume AND
  30%+ one-sided imbalance** → only genuine squeeze cascades (the 7–30% moves).
  This is the fix for "doesn't catch big moves": routine liquidation noise no
  longer fires; real cascades do, and the wide stop lets them ride.

## Monitor cron (this topic)

`v22-liq-monitor` (job `2f03f7df3f75`) — every **15 min**, hidden job ID subject to change.
Posts only when *something changed*: feed problem, fresh liquidation spike, or newly
resolved V22 signal. **Silent** otherwise (watchdog pattern — the scheduler sends
nothing on empty stdout).

What it reports:
1. **Feed health** — WS daemon / event watcher up, snapshot freshness (<360 s).
   🚨 **ALERT** on stale/missing liquidation data (a hard fail-closed for V22 — no
   data ⇒ cannot score squeezes).
2. **Liquidation activity** — top symbols by $ liq volume in the last 5 min.
3. **Realized performance** — sent→resolved W/L, WR, avg ±, EV%/trade for 22.0/22.1.
4. **Recent resolved signals.**

Health restart commands:
```bash
bash ~/.hermes/scripts/liq_ws_daemon.sh        # restart WS feed
bash ~/.hermes/scripts/event_watcher_ensure.sh # restart event watcher
```

## Liquidation event history archive (2026-09-19)

`binance_liq_ws.py` now appends every WS `!forceOrder` event to
`data/liquidation_history/YYYY-MM-DD.jsonl` (JSONL, deduped) — **forward
only**. Purpose: the new V22 trigger (volume ≥$1M + ≥30% imbalance) needs
historical event volumes+sides to backtest, but the rolling snapshot only kept
~1h. Once weeks of archive accumulate, we can replay the trigger against real
cascade events.

**Backfill status (2026-09-19): NO free public historical liquidation-event
feed exists on Binance.** `fapi/v1/forceOrders` requires an API key and returns
only *your own* force orders; `allForceOrders` is not public. The public
`/futures/data` history (taker buy/sell ratio, global/top long-short, open
interest) retains only ~30 days and is ratio/OI **proxy** imbalance, not the
event-level data the trigger consumes. So the exact-data path is the forward
archive; ratio history is only a coarse proxy.

## Realized performance (sent→resolved) — 2026-09-19

| Config | Sent | W/L | WR | avg+ | avg− | EV/trade | Verdict |
|--------|------|-----|----|------|------|----------|---------|
| 22.0 LONG | 53 | 27/14 | 66% | +3.56% | −2.83% | **+1.38%** | ✅ Positive — keep |
| 22.1 SHORT | 22 | 6/9 | 40% | +6.00% | −3.78% | **+0.13%** | ⚠️ Marginal — rework |

- EV is not Wilson-CI significant yet (n 22.0 = 41 W/L, 22.1 = 15 W/L; trust after ~30).
- **22.0 was already dropped from the Bybit/HyroTrader route** (realized loser on that
  acct), still live on Binance testnet route.
- 22.1 SHORT has a wide avg-loss (−3.78%) vs 22.0 (−2.83%) — the short squeeze
  signals are more violent/reversion-prone.

## ⚠️ Known issue: NOT majors-only in practice → FIXED (2026-09-19)

Configs were described *"BTC/ETH only"* / *"designed for major crypto symbols"* but
`symbol_whitelist=[]` + `coin_type_filter=[ANY]` fired across ~150 symbols incl.
long-tail alts. **Fixed:** both V22 configs now set `use_dynamic_majors=True`,
restricting the universe to the **dynamic weekly majors list**:

- `scripts/update_major_symbols.py` → `data/major_symbols.json` = top-40 USDT-M perp
  symbols by 24h quote volume (≥$20M), filtered to **known crypto** via
  `get_coin_type != OTHER` (drops tokenized stocks XAU/XAG/SOXL/MSTR/SNDK/CL/G +
  spam 龙虾/PUMP) + `is_stablecoin_or_stock` (stables/wrapped/stocks).
- Weekly cron `update-majors-weekly` (job `c0f017380e40`, Mon 05:00 UTC, deliver=local)
  re-fetches the list. `ScoringConfig._dynamic_majors()` caches it 5 min, so the
  5-min scan and the event watcher pick up the refresh within a cycle.
- **Daemons were restarted** (2026-09-19) so the event-triggered V22 path runs the
  new gating code. `binance_liq_ws` + `event_watcher` verified running + fresh snapshot.

Current majors list (top 40): BTC, ETH, SOL, ZEC, XRP, NEAR, UNI, HYPE, DOGE, BNB,
ARB, SUI, ENA, WLD, ADA, LINK, STRK, BCH, TAO, AAVE, ONDO, XMR, FIL, APT, LTC, DASH,
INJ, XLM, DOT, OP, SYN, ETHFI, FET, TIA, HBAR, PENDLE, TRX, AERO, CRV, JUP.

## Backtest & rework state

- Config `v22_liq_monitor.py` (`~/.hermes/scripts/`) is self-contained; rerun to
  refresh stats: `python3 ~/.hermes/scripts/v22_liq_monitor.py`.
- Backtest methodology for the 7.x family (`edge-scanner-config-backtesting` skill):
  signal-replay or OHLC-generator via `backtest_7x_sweep.py` / `backtest_7x_generator.py`,
  always sequential (one position at a time) + vectorized indicators + ≥6yr validation.
  V22 has real realized signals for a signal-replay sweep once a decision is made on
  stop/TP geometry.

### Exit-geometry sweep on real V22 signals (backtest_7x_sweep.py, 2026-09-19)

Replays every real logged signal against forward OHLC and re-simulates exit under
each (atr_stop_mult × rr_ratio × max_hold). EV_R = R-multiples/trade.

**Original full-universe (mostly alts) replay** suggested tight stop + rr 1.5–2.0
on 607 (22.0) / 249 (22.1) signals. **But after V22 became majors-only, the
majors-only replay is the valid one:**

**V22.0 (LONG) majors-only — 201 signals:**
| mult | rr | hold | n | W/L/F | EV_R |
|---|---|---|---|---|---|
| 1.5 | 1.5 | 24 | 201 | 69/86/46 | **+0.087** |
| 1.5 | 1.5 | 72 | 201 | 79/104/18 | +0.072 |
| 1.5 | 2.0 | 24 | 201 | 45/98/58 | −0.040 |
| 1.5 | 3.0 | 24 | 201 | 25/107/69 | −0.159 |

**V22.1 (SHORT) majors-only — 69 signals:**
| mult | rr | hold | n | W/L/F | EV_R |
|---|---|---|---|---|---|
| 1.5 | 2.0 | 48+ | 69 | 28/41/0 | **+0.217** |
| 1.5 | 1.5 | any | 69 | 32/37/0 | +0.159 |
| 1.5 | 2.0 | 24 | 69 | 27/41/1 | +0.188 |

### CRITICAL finding: RR alone can't capture the big squeezes
Didier's target moves (BTC 75→82k, ETH 2400→2600 = **6–10% in a day**, up to
**20–30% over 2–3 days** for BTC/ETH/BNB) require **RR ~6–10 at a ~0.9–1% stop**.
But the majors-only replay shows extreme RR is **strongly negative EV** (rr 4 →
−0.6, rr 6–8 → −0.8, ~0 wins): the tight 1.5×ATR stop (~0.9% on BTC/ETH/BNB)
gets **stopped out by normal whipsaw before** the far target is reached. Extending
the hold to 72–144h adds **zero** new wins (FLATs just convert to LOSS). The big
moves are **tail events** a tight-ATR stop model structurally cannot hold through.

**Conclusion:** lifting RR without widening the stop does not capture 30% moves —
it converts winners to stop-outs. The data-backed ceiling is rr 1.5–2.0.

### Final applied params (live, 2026-09-19)
| Config | stop_mult | rr | dynamic majors | status |
|---|---|---|---|---|
| **22.0** LONG | 1.5 (was 3.0) | **1.5** (was 1.2) | ✅ | enabled |
| **22.1** SHORT | 1.5 (was 3.5) | **2.0** (was 1.2) | ✅ | **enabled — NOT disabled** |

- 22.0: rr 1.5 (its majors-only best). 22.1: rr 2.0 (beats 1.5, EV +0.22).
- Both daemons restarted with new params (liq_ws, event_watcher).
- Backup: `scoring_config.py.bak-20260919`.
- ⚠️ Signal-replay ranks RELATIVE quality reliably; absolute WR won't match live.
- If Didier still wants the 20–30% tail-ride, the lever is a **wider stop** (e.g.
  3–4×ATR) to survive whipsaw to the far target — a separate sweep, not just RR.
  Monitor realized WR via `v22_liq_monitor.py` over the next 30+ trades and adjust
  only with data.