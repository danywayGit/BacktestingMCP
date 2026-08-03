# Edge Scanner Strategy Evolution Log

> **Purpose:** Document every strategy change, why it was made, and what happened.
> **Goal:** Prevent repeating mistakes, track what works, enable systematic improvement.
> **Last updated:** 2026-08-03

---

## 2026-08-03: arXiv Research Findings

### Key Papers

| Paper | Finding | Application |
|-------|---------|-------------|
| **2506.11921** (Jun 2025) | Dynamic Grid Trading (DGT) beats static grid + buy-hold on BTC/ETH | Adapt grid levels to market conditions |
| **1411.5062** (Leung & Li) | **Higher stop-loss → lower optimal take-profit** (mathematically proven) | **Confirms our R:R 1.0 strategy** |
| **2407.11786** (Hafid et al.) | EMA + MACD as XGBoost features for BTC prediction | ML signal generation |
| **2208.07168** (Jevtic et al.) | LSTM/RF/SVR vs conventional; performance varies by volatility regime | Market regime awareness needed |
| **1806.06632** (Burnie) | Crypto correlation networks — related coins move together | Add correlation filter to avoid stacking correlated positions |

### Critical Insight from 1411.5062
> "A higher stop-loss level always implies a lower optimal take-profit level."

This **mathematically validates** our V12.0 design: we widened the stop (atr_stop_mult 1.5→3.0) AND lowered the target (rr_ratio 2.0→1.0). These two changes are complementary, not contradictory.

---

## 2026-08-03: Deep Analysis & Major Fixes

### Context
Bot lost 8.8% ($5000→$4560) in 2 weeks. All 5 most recent trades were losses.

### Analysis Findings

| Finding | Evidence | Action Taken |
|---------|----------|--------------|
| **Stops too tight** | ETH stop 0.7% (price moves 1% in minutes) | `atr_stop_mult` 1.5→3.0 (ALL configs) |
| **12+ scores lose** | 12+ scores: 22% WR vs 10-12: 59.4% WR | Score cap at 11.0 in bridge |
| **5 symbols 0% WR** | BTW, EUL, EIGEN, MORPHO, DGB never won | Symbol exclusion list |
| **V1.0 actual R:R 1.5** | Designed R:R=2.0 degraded by MARKET fills | Raised bridge threshold to 10.0 |
| **50% flats** | 102/203 signals flat, 29h avg capital lock | V6.4 Flat Killer + V11.0 |

### New Configs Created

| Config | Purpose | Key Parameters |
|--------|---------|----------------|
| V6.4 | Flat Killer | atr_stop_mult=3.0, rr_ratio=1.2, min_adx=20 |
| V10.0 | Chart Pattern Hunter | chart_pattern_weight=5.0, Breakout=2x score |
| V11.0 | Optimized Pro | min_abs_score=7.0, pattern+divergence confirmation |

### Bridge Priority (Aug 3)
1. V11.0 (7.0) — Optimized Pro
2. V5.1 (7.5) — AI-focused
3. V10.0 (7.5) — Chart Patterns
4. V6.2 (7.5) — Pullback
5. V6.3 (7.5) — Pullback v2
6. V6.4 (7.5) — Flat Killer
7. V3.1 (7.5) — ADX Trend
8. V4.1 (7.5) — Breakout
9. V9.0 (7.0) — Vol Imbalance
10. V1.5 (7.5) — Conservative R:R
11. V1.0 (10.0) — Only elite signals

---

## 2026-08-01: V10.0 Chart Pattern Strategy

### What
Created V10.0 that scrapes altFINS technical-analysis page for live chart patterns.

### Parameters
- chart_pattern_weight=5.0
- Breakout stage = 2x score, Emerging = 1x
- Bullish = +score, Bearish = -score

### Why
The altFINS website shows 25+ patterns per day (Bullish Flag, Falling Wedge, etc.) that weren't being used in scoring.

### Lessons
- Vaadin web apps are hard to scrape but Playwright works
- 25 patterns visible per day; Breakout stage is strongest
- Daily cron at 06:00 UTC refreshes pattern cache

---

## 2026-07-31: Resolution Time Fix

### Issue
All signals showed 24-26h resolution time. Bug in `_find_first_hit_hours`:
- `isinstance(idx, datetime)` failed with pandas Timestamps
- Fallback stored `(resolved_at - entry_time)` always showing ~24h

### Fix
- `.to_pydatetime()` for pandas Timestamps
- Offset-naive vs offset-aware datetime comparison
- Cleared stale `.pyc` cache

### Result
- WIN avg: 26h → 15.3h
- LOSS avg: 26h → 9.3h
- Re-resolved 321+ signals

### Lesson
**Stale `.pyc` cache is a recurring issue.** After code changes, always force recompile: `py_compile.compile()` or delete `__pycache__/`.

---

## 2026-07-30: V1.5 Conservative R:R

### What
Created V1.5 with rr_ratio=1.2 (was 2.0) and target distance 2.4×ATR.

### Why
Analysis showed 178/186 signals sent were V1.0 (rr_ratio=2.0) with targets too far. Only 1 V1.5 signal sent.

### Parameters
- rr_ratio=1.2
- min_abs_score=4.0
- min_volume_relative=0.5

### Lesson
Lower R:R = higher TP hit rate = better for real execution. But too few signals were sent to prove it.

---

## 2026-07-29: MARKET Orders for EdgeScanner

### What
Changed order type from LIMIT to MARKET for EdgeScanner signals.

### Why
LIMIT orders were not filling, causing missed trades. MARKET orders fill instantly.

### Parameters
- `use_market_order=(strategy_name == 'EdgeScanner')`
- ManualTrading/DCA keep LIMIT orders

### Trade-off
- MARKET: 0.04% fee, instant fill
- LIMIT: 0.02% fee, may not fill

### Lesson
MARKET orders are better for systematic signals where speed matters more than 0.02% fee.

---

## 2026-07-28: PnL Fix

### Issue
Dashboard showed $0 PnL for all positions. `get_live_price` was using admin user's API keys instead of target user's.

### Fix
`get_live_price` now accepts `user_id` parameter and fetches using target user's API keys.

---

## 2026-07-27: Stop Loss Placement Fix

### Issue
Stop losses were not being placed on the exchange. The bot opened positions without SL protection.

### Fix
Emergency close if SL placement fails. Hard rule: no position open without SL.

---

## Parameter History

### atr_stop_mult (stop distance multiplier)
| Date | Value | Reason |
|------|-------|--------|
| Jul 2026 (launch) | 1.5 | Default |
| **Aug 3** | **3.0** | **Stops too tight, ETH 0.7% hit by noise** |

### rr_ratio (risk-reward)
| Date | Value | Reason |
|------|-------|--------|
| Jul 2026 | 2.0 | Default |
| Jul 30 | 1.2 (V1.5) | Closer target for higher hit rate |
| **Aug 3** | **1.2 (V6.4, V11.0)** | **Same — closer target proven better** |

### min_abs_score (minimum signal score)
| Date | Value | Configs | Reason |
|------|-------|---------|--------|
| Jul 2026 | 3.0 | V1.0 | Default |
| Jul 2026 | 4.0 | Most | Standard |
| **Aug 3** | **7.0** | **V11.0** | **Only high-conviction** |
| **Aug 3** | **8.0** | **V11.0 (initial)** | **Too high, 0 signals → lowered to 7.0** |

### Bridge Threshold History
| Config | Jul 25 | Jul 29 | Jul 30 | Aug 1 | Aug 3 |
|--------|--------|--------|--------|-------|-------|
| V1.0 | 8.5 | 8.0 | 9.0 | 10.0 | 10.0 |
| V1.5 | — | — | 7.5 | 7.5 | 7.5 |
| V10.0 | — | — | — | 7.5 | 7.5 |
| V11.0 | — | — | — | — | 7.0 |

---

## Known Issues (Not Yet Fixed)

1. **Stale `.pyc` cache** — After code changes, resolution cron may use old bytecode. Force recompile.
2. **No market regime detection** — Configs don't adapt to bull/bear/sideways markets
3. **No trade journal** — Bot doesn't record why each trade was taken (which config, which pattern)
4. **No auto-stop** — Configs with 0% WR continue generating signals until manually disabled
5. **ATR timeframe mismatch** — 1h ATR used for 1h signals, but 4h ATR may be more stable
6. **No correlation filter** — Multiple correlated signals (e.g., 2 ETH-related coins) can open simultaneously