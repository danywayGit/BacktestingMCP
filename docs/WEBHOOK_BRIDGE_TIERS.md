# Webhook Bridge — Priority Tiers & Rotation System

> **Purpose:** Reference for rebuilding the bridge priority + rotation system on a new machine.
> **Last updated:** 2026-08-17

---

## Overview

The webhook bridge (`src/edge_scanner/webhook_bridge.py`) selects signals from the edge scanner DB and sends them to the Trading-WebHook-Bot for execution on Binance (TestNet). It uses a **4-tier priority queue** to decide which configs get to send signals.

## Key Concepts

### Config Status
- **`active`** — Generates Telegram alerts + scans + DB logging
- **`enabled`** — Scans + DB logging only
- **`disabled`** — Records only, never sends to bridge

### Bridge Flow (per run, every 15 min)
1. Get pending signals from DB per config (score >= config threshold)
2. Iterate `CONFIG_PRIORITY` top-down
3. For each config: validate signal (entry/stop/target, slippage, regime, whitelist, blocklist, Binance Futures availability)
4. **Config diversity:** max 1 signal per config per batch
5. **Symbol dedup:** no duplicate symbols per batch
6. Stop when `MAX_SIGNALS_PER_BATCH` reached

## The 4 Tiers

```
CONFIG_PRIORITY = [
    # ── TIER 1: Best EV (backtested) — always checked first ──
    ("3.1", 7.0, "V3.1 ADX Trend"),         # EV=+7.05%
    ("4.1", 7.0, "V4.1 Breakout+Vol"),       # EV=+5.55%
    ("5.1", 7.0, "V5.1 AI-focused"),         # EV=+5.17%
    ("6.1", 7.0, "V6.1 Breakout Momentum"),  # EV=+4.87%
    ("2.1", 7.0, "V2.1 MT Alignment"),       # EV=+3.65%
    ("4.0", 7.0, "V4.0 TR/ATR Breakout"),    # EV=+2.92%

    # ── TIER 2: Solid EV configs (max 2 V1.x) ──
    ("1.4", 7.0, "V1.4 Scanner-Focused"),    # EV=+2.23% (ACTIVE)
    ("1.5", 7.0, "V1.5 Conservative R:R"),   # best execution EV
    ("2.2", 7.0, "V2.2 Soft MT Alignment"),  # EV=+1.92%
    ("8.0", 7.0, "V8.0 Funding Rate"),       # EV=+1.76%
    ("6.0", 7.0, "V6.0 Pullback"),           # EV=+1.70%
    ("3.2", 7.0, "V3.2 Soft ADX"),           # EV=+1.46%
    ("5.2", 7.0, "V5.2 Balanced DeFi/AI"),   # EV=+1.09%
    ("6.2", 7.0, "V6.2 Pullback Strat"),     # EV=+0.72%
    ("14.0", 7.0, "V14.0 Precursor BTC/ETH"), # EV=+0.45%
    ("5.0", 7.0, "V5.0 DEFI-focused"),       # EV=+0.05%

    # ── TIER 3: Negative EV but keep for data continuity ──
    ("10.0", 7.0, "V10.0 Chart Patterns"),
    ("12.0", 7.0, "V12.0 Optimized Pro V2"),
    ("16.0", 7.0, "V16.0 Vol Squeeze"),
    ("11.0", 7.0, "V11.0 Optimized Pro"),
    ("13.0", 7.0, "V13.0 Auto-Evolved"),

    # ── TIER 4: Rotating configs (daily round-robin) ──
    ("1.1", 7.0, "V1.1 Volume-Weighted"),
    ("1.2", 7.0, "V1.2 Signal-Focused"),
    ("1.3", 7.0, "V1.3 On-Chain"),
    ("7.2", 7.0, "V7.2 Filtered Quality"),
    ("7.5", 7.0, "V7.5 LLM Quality Gate"),
    ("7.6", 7.0, "V7.6 LLM Evolved"),
    ("7.8", 7.0, "V7.8 LLM Evolved v2"),
    ("3.3", 7.0, "V3.3 LLM ADX"),
    ("3.5", 7.0, "V3.5 LLM ADX v2"),
    ("6.4", 7.0, "V6.4 Flat Killer"),

    # Special purpose (not in rotation)
    ("20.0", 5.0, "V20.0 Time-of-Day"),
    ("19.0", 5.0, "V19.0 Ratio Arb"),
    ("18.0", 7.0, "V18.0 Mean Reversion"),
    ("17.0", 7.0, "V17.0 Liquidation"),
    ("15.0", 7.0, "V15.0 Multi-TF"),
    ("9.0", 7.0, "V9.0 Vol Imbalance"),
]
```

## How Rotation Works

### Purpose
Tier 4 configs are moved into the priority rotation so they get a chance to accumulate execution data. Without rotation, they'd never send signals because higher-priority configs fill the batch first.

### Mechanism
```python
# In select_signals() function:
from datetime import datetime, timezone
_rot_offset = datetime.now(timezone.utc).day % 7  # 0-6, changes daily
```

The rotation offset is based on **day of month % 7**. This means:
- Day 1, 8, 15, 22, 29 → offset 0
- Day 2, 9, 16, 23, 30 → offset 1
- ...etc

**Currently the rotation is implemented as a simple offset variable.** A more complete implementation would rotate the Tier 4 list before iterating:

```python
# Full rotation implementation (recommended for rebuild):
def _rotate_tier4(priority_list, offset):
    """Rotate the Tier-4 segment of the priority list by `offset` positions."""
    # Find Tier 4 start index (first config after '6.4 Flat Killer' marker)
    # Or: pass tier4 list separately and rotate it
    tier4 = [c for c in priority_list if c[0] in TIER4_VERSIONS]
    other = [c for c in priority_list if c[0] not in TIER4_VERSIONS]
    rotated = tier4[offset:] + tier4[:offset]  # rotate left by offset
    return other + rotated
```

### Rotation Design Notes
- `day % 7` gives a 7-day cycle covering all 7 rotating configs in a week
- Using `day` (not hour) means the rotation is stable within a day
- Configs NOT in the rotation (Tier 1-3, special purpose) always keep their priority
- Rotation only affects which Tier 4 config gets a chance when there's batch room left

## Constants

```python
MAX_SIGNALS_PER_BATCH = 8       # Matches the 8 concurrent trade slots on bot
MAX_SCORE_CAP = 15.0            # 12+ scores have 57.1% WR, EV=+4.12%
EXCLUDED_SYMBOLS = {"BTWUSDT", "EULUSDT", "EIGENUSDT", "MORPHOUSDT", "DGBUSDT"}
MAX_SLIPPAGE_PCT = 0.5          # Max price difference from entry before skip
```

## Signal Validation Chain (order matters)

For each candidate signal, ALL checks must pass in this order:

| # | Check | Skip reason |
|---|-------|-------------|
| 1 | Config not disabled | "config is disabled" |
| 2 | Symbol in whitelist (if set) | "not in config whitelist" |
| 3 | Market regime filter (BTC) | "REGIME BLOCKED" |
| 4 | Time-of-day filter | "TIME BLOCKED" |
| 5 | Symbol not in EXCLUDED_SYMBOLS | "excluded symbol (0% WR)" |
| 6 | Score capped at MAX_SCORE_CAP | "score capped" |
| 7 | On Binance Futures | "not on Binance Futures" |
| 8 | Actively TRADING (not PENDING_TRADING) | "not actively TRADING" |
| 9 | On TestNet (if ACCOUNT_TYPE=TestNet) | "not on TestNet" |
| 10 | Config diversity (not already selected) | "config already contributed" |
| 11 | Hard validation: entry/stop/target valid | "REJECTED" |
| 12 | Slippage check (mark price vs entry) | "SKIPPED (slippage)" |

## Config EV Source (how tiers were ranked)

The EV values come from the `edge_signals` table in `data/crypto.db`:

```sql
SELECT config_version,
       SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as w,
       SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) as l,
       AVG(CASE WHEN outcome='WIN' THEN forward_return_pct END) as aw,
       AVG(CASE WHEN outcome='LOSS' THEN forward_return_pct END) as al
FROM edge_signals
WHERE status='RESOLVED' AND outcome IN ('WIN','LOSS')
GROUP BY config_version
HAVING (w+l) >= 20
ORDER BY EV DESC;
```

EV = `(w/(w+l) * aw) + (l/(w+l) * al)`

**Important:** Re-run this ranking periodically (every 1-2 weeks) and update the tier assignment based on fresh EV data.

## Rebuild Checklist (new machine)

1. **Clone repos:**
   ```bash
   git clone https://github.com/danywayGit/BacktestingMCP.git
   git clone https://github.com/danywayGit/Trading-WebHook-Bot.git
   ```

2. **Python venv:**
   ```bash
   cd BacktestingMCP && python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Verify bridge imports:**
   ```bash
   cd BacktestingMCP
   python -c "from src.edge_scanner.webhook_bridge import CONFIG_PRIORITY, MAX_SIGNALS_PER_BATCH; print(len(CONFIG_PRIORITY), 'configs, batch=', MAX_SIGNALS_PER_BATCH)"
   ```

4. **Test selection logic (dry run):**
   ```bash
   python -c "from src.edge_scanner.webhook_bridge import select_signals; print(select_signals())"
   ```

5. **Verify rotation offset:**
   ```bash
   python -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).day % 7)"
   ```

6. **Check DB exists:** `data/crypto.db` with populated `edge_signals` table

7. **Cron setup:**
   ```
   */15 * * * * cd /home/hermes/BacktestingMCP && . .env && venv/bin/python -m src.cli.main edge bridge
   ```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Config never sends signals | Not in CONFIG_PRIORITY | Add to list |
| Batch fills with 1 config | Diversity check disabled | Verify `selected_configs` logic |
| Rotation not happening | `_rot_offset` unused | Implement full `_rotate_tier4()` |
| Config disabled still sends | status not checked | Verify ALL_CONFIGS lookup |
| Symbols rejected | Not on Futures/TestNet | Check exchangeInfo status |

## Related Docs
- `docs/ARCHITECTURE.md` — System architecture overview
- `docs/STRATEGY_EVOLUTION.md` — Strategy change log & reasoning
- `src/edge_scanner/webhook_bridge.py` — The bridge implementation
