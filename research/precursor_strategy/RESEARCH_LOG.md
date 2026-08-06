# Precursor Strategy — Research Log

> **Purpose:** Track every phase, finding, and decision in the precursor pattern
> research pipeline. This log is the long-term value — it lets the next iteration
> build on the last instead of restarting blind.
>
> **Location:** `research/precursor_strategy/`
> **Last updated:** 2026-08-05

---

## Current Status

| Phase | Status | Deliverable |
|-------|--------|-------------|
| Phase 1 — Event Detection | ✅ Complete | events.csv (15 events) |
| Phase 2 — Precursor Features | ⏳ Pending | precursor_features.csv |
| Phase 3 — Precursor Mining | ⏳ Pending | Ranked candidate list |
| Phase 4 — OOS Validation | ⏳ Pending | Validated precursor list |
| Phase 5 — Strategy / Backtest | ⏳ Pending | Backtest report |
| Phase 6 — Paper Trading | ⏳ Pending | Paper trade log |
| Phase 7 — Continuous Loop | ⏳ Pending | Auto-refresh cron |

---

## Phase 1 — Event Detection

**Parameters:**
- Timeframe: 1h
- Window: 24h (24 × 1h candles)
- Threshold: 5% (using intra-window high/low)
- Lookback: 365 days
- Symbols: BTCUSDT, ETHUSDT

**Results:**
- BTC: 2 events (1 UP, 1 DOWN)
- ETH: 13 events (7 UP, 6 DOWN)
- Total: 15 events

**Notes:**
- Low event count reflects the current low-volatility regime for BTC
- Many events show large intra-window swings that partially recovered
- ETH is significantly more volatile than BTC in this period
- The 15m timeframe may yield more events (W=16, 4h window)

**Decision:** Proceed with 15 events. If Phase 3 shows insufficient data,
we'll revisit the timeframe or threshold.

---

## Config Evolution

### V14.0 — Precursor Pattern Strategy (preliminary, from quick analysis)

**Parameters:** rsi_momentum_weight=3.0, medium_term_trend_weight=2.0,
atr_stop_mult=3.0, rr_ratio=1.2, min_atr_pct=0.5, min_volume_relative=0.5

**Bridge priority:** 1st (threshold 5.0)

**Status:** ⚠️ Preliminary — built from quick analysis before the full
research pipeline was defined. The real V14.0 will be updated after Phase 5
delivers a statistically validated precursor set.

**How to update:** Run `python research/precursor_strategy/phase1_event_detection.py`
to re-detect events with updated parameters.

---

## Known Issues

- Small event count (15) limits statistical power
- Current market is low-volatility regime — may need to expand lookback
- No control set yet (Phase 3 requirement)
- No visual chart feature extraction yet (Phase 2b)