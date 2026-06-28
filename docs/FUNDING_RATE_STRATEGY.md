# Funding Rate Mean-Reversion Strategy — Predefined Rules

## Overview

Three distinct trade setups derived from Binance perpetual futures funding rate dynamics. 
All entries are scored via the edge scanner (V8.0) and resolved via the existing target/stop 
resolution pipeline.

---

## Strategy 1: Extreme Funding Normalization (V8.0 Primary)

**Concept:** When funding is extremely negative (shorts paying longs), the market is 
positioned for a squeeze. As funding normalizes toward zero, the squeeze materializes as 
price moves in the opposite direction of the extreme.

### Entry Conditions (ALL must pass)

LONG setup (negative funding):
- `funding_rate < -0.006` (extreme negative, shorts paying)
- `funding_rate > previous_rate` (normalizing — less negative than 1h ago)
- `funding_momentum > 0` (rising toward zero)
- Price: no fresh 15m lower-low for 2 consecutive candles
- OI: declining or flat (shorts covering, not adding)

SHORT setup (positive funding):
- `funding_rate > +0.006` (extreme positive, longs paying)
- `funding_rate < previous_rate` (normalizing — less positive)
- Price: no fresh 15m higher-high for 2 candles
- OI: declining or flat

### Exit Rules

| Condition | Action |
|-----------|--------|
| Funding reaches -0.001 to +0.001 (near zero) | Close 100% |
| Funding flips sign (negative → positive) | Close 100% immediately |
| New funding interval detected (e.g., 4h → 1h) | Close 50%, tighten stop |
| Price hits ATR stop (atr_stop_mult=1.0) | Close all |
| 24h elapsed | Close all |

### Position Sizing

Funding-based adjustment (multiplier applied to base 1% risk):
- `|funding| > 0.02` (extreme): 0.5% risk
- `|funding| 0.01-0.02` (high): 1.0% risk
- `|funding| < 0.01` (moderate): 0.25% risk

### Interval Scaling

Funding cost per day depends on interval:
```
1h funding × 24 = daily cost
4h funding × 6  = daily cost
8h funding × 3  = daily cost
```

V8.0 scores are automatically scaled: **1h funding is 8× more expensive than 8h at the same rate**.

---

## Strategy 2: Pre-Funding Dip Capture

**Concept:** 15-45 minutes before funding settlement, the price tends to dip as 
traders close positions to avoid funding payment. This creates a short-term 
reversal opportunity.

### Entry Conditions

- Funding is extreme negative (`< -0.006`)
- Time to next funding: **15-45 minutes**
- Price has dropped >1% in the last 30 minutes (pre-funding dip in progress)
- No major support level broken

### Entry

Go LONG at the dip, 15-45 min before funding timestamp.

### Exit

| Condition | Action |
|-----------|--------|
| **Funding timestamp hits** | Close 100% (collect funding payment) |
| Price recovers >0.5% from entry | Close 50%, hold rest through funding |
| Price breaks 2% below entry | Stop loss — close all |
| 5 minutes before next funding | Close all remaining |

### Position Sizing

- If `|funding| > 0.01` (very expensive): 1.5% risk (higher conviction — bigger dip expected)
- If `|funding| 0.006-0.01`: 0.75% risk
- If `|funding| < 0.006`: No trade (not extreme enough)

### Pattern Validation (LABUSDT Jun 2026)

```
Date        Pre-30min  Funding Candle  Pattern
──────────────────────────────────────────────
Jun 27 18:00  -5.07%     -1.77%       Dip → continued drop → recovery
Jun 28 02:00  +1.94%     -8.40%       Rise → crash at settlement
Jun 24 18:00  -0.10%     -0.11%       Flat → flat
```

The -5.07% dip at Jun 27 18:00 is the classic pattern: price drops hard 30min 
before funding, shorts get squeezed out at settlement, then price recovers.

---

## Strategy 3: Interval Switch Reversal

**Concept:** When Binance changes the funding interval of a symbol (e.g., 4h → 1h), 
it signals a volatility event. This is the strongest signal because the exchange 
is intervening to accelerate mean-reversion.

### Entry Conditions

- **Interval decreased** detected (e.g., 8h → 4h, 4h → 1h)
- Funding is extreme in the direction opposite to the intended trade
- Price is at a significant deviation from the 24h VWAP

### Entry

- If interval decreased AND funding extremely negative → LONG (squeeze incoming)
- If interval decreased AND funding extremely positive → SHORT (dump incoming)

### Exit

| Condition | Action |
|-----------|--------|
| Funding normalizes to < 0.003 | Close 100% |
| Interval reverts to original | Close 100% — signal exhausted |
| 3 funding periods elapsed | Close 100% |

### Real Example (LABUSDT Jun 24)

```
16:00 UTC — funding hits -2.000% (4h interval)
16:00-17:00 UTC — Binance switches LAB to 1h funding ← THIS IS THE SIGNAL
17:00 UTC onward — funding starts normalizing: -0.575%, -0.457%, -0.309%...
Price recovered from ~$13.70 → $17.85 (+30%)
```

The interval switch was the **exact peak signal**. The -2.0% was the maximum extreme,
and the switch to 1h forced mean-reversion.

---

## Integration with Edge Scanner

### New ScoringConfig Fields

| Field | Default | Purpose |
|-------|---------|---------|
| `funding_interval_weight` | 0.0 | Bonus when interval is shorter (more frequent = more expensive) |
| `pre_funding_dip_weight` | 0.0 | Score boost 15-45 min before funding when rate is extreme |
| `interval_switch_weight` | 0.0 | Score boost when interval decreases (volatility event) |

### V8.0 Score Calculation (Updated)

```
total_score = 
    trend_score × trend_weight (0.3) 
    + funding_rate_score × funding_rate_weight (5.0)
    + funding_momentum_score × funding_momentum_weight (5.0, was 3.0)
    + oi_change_score × oi_change_weight (2.0)
    + interval_bonus × funding_interval_weight (2.0)
    + dip_bonus × pre_funding_dip_weight (3.0)
```

### Resolution Time (Horizon)

| Strategy | Horizon | Rationale |
|----------|---------|-----------|
| Funding Normalization | 24h | Allows time for full squeeze |
| Pre-Funding Dip | **1 funding interval** | Must close at/after funding timestamp |
| Interval Switch | 3 funding periods | Signal decays after 3 cycles |

---

## Execution Notes

1. **Always use mark price** for PnL and liquidation checks (Binance uses mark price 
   for funding settlement, not last price).

2. **Funding payouts occur at exact UTC timestamps** (00:00, 04:00, 08:00 for 4h; 
   every hour on the hour for 1h). Must hold through the exact second to receive 
   the funding payment.

3. **Limit orders only** — maker fee tier is essential for strategies that collect 
   funding (the fee delta eats the profit otherwise).

4. **Cross-exchange check**: If funding is extreme on Binance but normal on Bybit/OKX,
   the deviation may be exchange-specific, not market-wide. Lower conviction.
