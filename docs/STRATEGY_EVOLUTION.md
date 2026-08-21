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

## 2026-08-13 to 2026-08-15: Major Reconfiguration

### Context
After analyzing 333 sent signals, clear performance hierarchy emerged. V1.0 (38.2% WR) and V6.3 (14.3% WR) were bleeding. V10.0 (90.9% WR) was underutilized at middle priority.

### Config Changes

| Action | Config | Why |
|--------|--------|-----|
| **DISABLED** | V1.0 | 38.2% WR, was the main culprit in 8-9.9 score range |
| **DISABLED** | V6.3 | 14.3% WR, EV=-3.50% per trade |
| **ENABLED** | V1.4 | Scanner-focused (scanner_hit_weight=2.0), atr_stop_mult=3.0, rr_ratio=1.5 |
| **ENABLED** | V1.5 | Conservative R:R (1.2), same formula as V1.0 |
| **ENABLED** | V3.6 | First V3.x that actually sends signals (bridge-added) |
| **ENABLED** | V6.4 | Flat Killer: tight stop, R:R 1.2, high vol (added to bridge) |

### Bridge Priority (Aug 15)

| # | Config | Threshold | Label |
|---|--------|-----------|-------|
| 1 | V10.0 | 7.0 | Chart Pattern Hunter (90.9% WR 🏆) |
| 2 | V20.0 | 5.0 | Time-of-Day |
| 3 | V19.0 | 5.0 | Ratio Arb |
| 4 | V18.0 | 6.0 | Mean Reversion |
| 5 | V17.0 | 6.0 | Liquidation |
| 6 | V16.0 | 6.0 | Vol Squeeze |
| 7 | V15.0 | 6.0 | Multi-TF |
| 8 | V6.4 | 7.0 | Flat Killer 🆕 |
| 9 | V3.6 | 7.0 | Bridge-Active ADX 🆕 |
| 10 | V1.5 | 7.0 | Conservative R:R 🆕 |
| 11 | V1.4 | 7.0 | Scanner-Focused |
| 12 | V14.0 | 7.0 | Pattern Discovery (BTC/ETH only) |

### ACTIVE_CONFIG Changed
V3.1 → **V1.4** (Telegram alerts now use scanner-focused formula)

### Alert System Improvements

| Issue | Fix |
|-------|-----|
| **Duplicate alerts every 15min** | 24h cooldown cache (file-backed JSON, survives restarts) |
| **Same signal re-alerted** | DB check: skip if unresolved signal already sent to bot |
| **UTF-8 garbage in logs** | Replaced emoji/multiplication chars with ASCII |
| **Resolved stats aggregated all configs** | Now shows per-config + all-configs, excludes disabled configs |

### Early Signal Detection (V1.4 Improvement)

**Problem:** V1.4 signals arrived 16h+ late — after the move already happened. The TA scanner only detected breakouts AFTER they occurred.

**Root Cause:** `_is_multi_source` required altFINS sources that are reactive. Volume threshold of 2.0x missed accumulation phases.

**Fix:** Added 3 new multi-source checks using own OHLCV data (no altFINS dependency):

| New Source | Description | Example Catch |
|------------|-------------|---------------|
| **own_volume ≥ 1.5x** | Our volume relative to 10-candle MA | Catches volume spikes from market_data table |
| **volume_accumulation** | Increasing volume over 3+ candles | Detects building pressure |
| **price_above_ema20 + vol ≥ 1.2x** | **Counts as 2 sources** | The pattern user identified: "above many MAs with volume" |

**Result:** For ICP, the setup was visible at Aug 14 10:00 (price above EMA20, volume 3.4x avg). This would have triggered the alert ~16 hours earlier than the previous signal.

## 2026-08-17: Deep Week Analysis — Major Reconfiguration (v2)

### Context
Last week (Aug 10-17): 92 trades, 42.9% WR, **net negative**. V16.0 bleeding (34 trades, EV=-0.71%), V3.6 terrible (0% WR, EV=-4.79%).

### Key Findings

| Finding | Data | Action |
|---------|------|--------|
| **V14.0 whitelist broken** | 106 signals since Aug 9, ZERO for BTC/ETH | Added symbol_whitelist check to `passes_filters()` — now enforced at scoring level |
| **V16.0 bleeding** | 34 trades, 44.1% WR, EV=-0.71% | Raised bridge threshold 6.0→7.0 |
| **V3.6 terrible** | 0% WR, EV=-4.79% | DISABLED |
| **Score 12+ profitable** | 57.1% WR, EV=+4.12% | Raised MAX_SCORE_CAP 11.0→15.0 |
| **R:R 1.0-1.4 sweet spot** | 53.8% WR, EV=+1.00% | All configs lowered to 1.0-1.4 |
| **V1.4 only positive EV** | 21 trades, EV=+0.95% | Stays active, rr lowered to 1.3 |

### V14.0 Rebuild (Validated Precursors)
Removed dependency on 5% move triggers. Now uses validated OOS precursors:
- **ATR expansion** (effect size 0.85 ETH / 0.80 BTC)
- **Volume Z-Score / Ratio** (effect size 0.78 / 0.68)
- **BB Width expansion** (effect size 0.81 / 0.66)
- Added `bb_squeeze_min=0.3` for Bollinger Band squeeze detection
- Whitelist now enforced at **scoring level** (not just bridge)

### Config Changes

| Action | Config | Detail |
|--------|--------|--------|
| **DISABLED** | V3.6 | 0% WR, EV=-4.79% |
| **FIXED** | V14.0 | Whitelist enforced at scoring, BB squeeze added |
| **RAISED** | V16.0 | Bridge threshold 6.0→7.0 |
| **RAISED** | V15/17/18 | Bridge threshold 6.0→7.0 |
| **LOWERED** | All active | R:R to 1.0-1.4 (V1.4: 1.3, V10: 1.3, V16: 1.3) |
| **RAISED** | MAX_SCORE_CAP | 11.0→15.0 (12+ scores profitable) |
| **REMOVED** | V3.6, V6.3 | From bridge priority (disabled) |

### Known Issues (Updated)
1. **Stale `.pyc` cache** — After code changes, resolution cron may use old bytecode. Force recompile.
2. **Market regime detection** — Bridge checks BTC EMA20 trend. LONG blocked in downtrend, SHORT blocked in uptrend.
3. **V1.4 timing still altFINS-dependent** — New OHLCV-based sources help but altFINS trend_score is still the main source.
4. **ATR timeframe mismatch** — 1h ATR used for 1h signals, but 4h ATR may be more stable.
5. **No correlation filter** — Multiple correlated signals (e.g., 2 ETH-related coins) can open simultaneously.
6. **V14.x only trades BTC/ETH** — Whitelist enforced at scoring level, not just bridge.
7. **Liquidation WS snapshot empty in quiet markets** — Falls back to Binance taker ratio / L/S ratio (always available).

---

## 2026-08-20 to 2026-08-21: Liquidation Overhaul + Bridge Robustness

### Context
Three parallel workstreams:
1. New V14.x precursor strategies (BB squeeze, ATR expansion, multi-precursor filter)
2. New V22.x liquidation-driven strategies (LONG + SHORT)
3. Replace Coinglass (paid, "upgrade required") with free Binance liquidation data
4. Fix bridge bugs (SHORT never sent, stale entry prices, scan failure orphaned signals)

### V14.x Precursor Strategy — Complete

| Feature | Weight | What it detects |
|---------|:------:|-----------------|
| BB Squeeze | 2.0 | Low vol → explosion (BB width < 20th percentile + expanding) |
| ATR Expansion | 1.5 | Volatility acceleration (current vs 60-period avg) |
| Volume Divergence | 2.0-3.0 | Price/volume disagreement |
| BB Position | 0.5-1.0 | Close near band extremes |
| Liquidation Pressure | 1.5-2.0 | Squeeze risk from positioning |
| Multi-precursor filter | — | min_precursors=2 (reduce false positives) |
| Dynamic ATR | — | Relaxes min_atr_pct 30% in low-vol regimes |

**Two variants:**
- **V14.0** — LONG bias, 7.0 bridge threshold, BTC/ETH whitelist
- **V14.1** — SHORT bias, 7.0 bridge threshold, BTC/ETH whitelist

### Historical Validation (2021-2026, hourly)

| Config | BTC hit rate (≥3%) | ETH hit rate (≥3%) | Avg score |
|--------|:-----------------:|:-----------------:|:---------:|
| V14.0 | 26% | **41%** | 5.4 |
| V14.1 | 26% | **39%** | 5.7 |

**Best year:** 2021 — 71% on ETH, 59% on BTC. Bull market = textbook precursors.

### V22.x Liquidation-Driven Strategy

| Aspect | V22.0 LONG | V22.1 SHORT |
|--------|-----------|------------|
| Triggers when | Long/Short ratio < 0.7 (short-heavy) | Long/Short ratio > 1.5 (long-heavy) |
| Liq weight | 5.0x (primary driver) | 5.0x (primary driver) |
| Confirmation | Volume spike (3.0x) | ATR expansion (1.5x) |
| Symbol scope | **All symbols** (no whitelist) | **All symbols** (no whitelist) |
| Stop loss | 3.0 ATR | 3.5 ATR |

### Liquidation Data Source Migration

| Period | Source | Cost | Data |
|--------|--------|:----:|------|
| Before Aug 21 | Coinglass API (open-api-v4.coinglass.com) | Upgrade required | Empty |
| **After Aug 21** 🆕 | **Binance !forceOrder@arr WS** | **Free** | Real liquidation events |
| **After Aug 21** 🆕 | **Binance takerlongshortRatio** | **Free** | Orderflow buy/sell ratio |
| **After Aug 21** 🆕 | **Binance globalLongShortAccountRatio** | **Free** | L/S positioning proxy |

**Architecture:**
1. WS daemon (persistent, auto-restart every 5 min by cron) listens to `!forceOrder@arr`
2. Accumulates events in rolling 1h window, writes JSON snapshot to `data/liquidation_snapshot.json`
3. Scanner reads snapshot → taker ratio → L/S ratio (fallback chain)
4. All endpoints public — **no API key needed**
5. Removed: `coinglass_liquidations.py`, `binance_liquidations.py`. Added: `binance_liq_ws.py`

### Bridge Bugs Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| **No SHORT signals ever sent** | SQL used `composite_score >= ?` — SHORT scores are negative | Changed to `ABS(composite_score)` — 344 SHORT signals now eligible |
| **Signals sent hours late** | Scan failed → script `exit 1` before bridge ran | Bridge always runs, even on partial scan. Script always exits 0. |
| **Target ≤ entry for LONG** | Target/stop computed from last_close, entry from live price | Target/stop computed from same live price as entry |
| **Degraded trades executed** | No check for effective R:R at send time | Live mark price check: reject if eff_RR < 0.8 or setup already completed |
| **Transient HTTP failures** | No retry on timeout/503 | 3 retries with exponential backoff |
| **Coinglass slowed scan** | ~100 HTTP calls per run, all "upgrade required" | Circuit-breaker after first reply. Then replaced entirely. |
| **Price cache missing** | Each of 8 signals fetched its own live price (8 API calls) | 5-second price cache per symbol |

### Cron Schedule Changes

| Job | Before | After | Why |
|-----|--------|-------|-----|
| edge-scan | Every 15 min | **Every 5 min** | Faster detection → narrower price gap |
| webhook-bridge | Every 30 min | **Every 5 min** | Backup failsafe if scan bridge fails |
| liq-daemon-ensure | — | **Every 5 min** | New: restart WS daemon if it dies |

### Edge Scan Script Hardened

```bash
# Before: scan failed → exit 1 → bridge NEVER ran
if [ $EXIT_CODE -ne 0 ]; then
  echo "FAILED"
  exit 1  # ← bridge never reached
fi

# After: bridge ALWAYS runs, even on partial failure
OUTPUT=$(timeout 300 python -m src.cli.main edge scan 2>&1)
# ...bridge runs regardless of EXIT_CODE...
if [ $EXIT_CODE -eq 0 ]; then
  exit 0
else
  echo "partial failure"
  exit 0  # ← always clean exit
fi
```

### Key Gains

| Metric | Before | After |
|--------|--------|-------|
| Signal → send gap | Up to **117 min** | **~0-5 min** |
| SHORT signals sent | **0** (all silently blocked) | Now eligible via ABS(score) |
| Degraded trades | Sent blindly | Rejected if effective R:R < 0.8 |
| Scan failure impact | Signals orphaned for hours | Bridge runs anyway |
| Liquidation data cost | $? (Coinglass plan upgrade) | **Free** (Binance public) |
| Liquidation coverage | BTC/ETH only | All symbols |
| API key required? | Yes (Coinglass) | **No** (Binance) |

### Disabled Configs (Unchanged from Aug 17)
V1.0 (38.2% WR), V2.0, V3.0 (43.8%), V3.6 (0% WR), V6.3 (14.3% WR), V7.0, V7.7

### New Configs (Aug 20-21)

| Config | Created | Bridge Level | Purpose |
|--------|:------:|:------------:|---------|
| V14.0 | Aug 17 | 7.0 (Tier 2) | Precursor-based LONG, BTC/ETH |
| V14.1 | Aug 21 | 7.0 (Tier 2) | Precursor-based SHORT, BTC/ETH |
| V22.0 | Aug 21 | 7.0 (Tier 4) | Liquidation-driven LONG, all symbols |
| V22.1 | Aug 21 | 7.0 (Tier 4) | Liquidation-driven SHORT, all symbols |