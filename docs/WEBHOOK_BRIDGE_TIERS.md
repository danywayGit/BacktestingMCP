# Webhook Bridge — Priority Tiers & Rotation System

> **Purpose:** Reference for rebuilding the bridge priority + rotation system on a new machine.
> **Last updated:** 2026-08-21

---

## Overview

The webhook bridge (`src/edge_scanner/webhook_bridge.py`) selects signals from the edge scanner DB and sends them to the Trading-WebHook-Bot for execution on Binance (TestNet). It uses a **5-tier priority queue** (46 configs) to decide which configs get to send signals.

## Key Concepts

### Config Status
- **`active`** — Generates Telegram alerts + scans + DB logging
- **`enabled`** — Scans + DB logging only
- **`disabled`** — Records only, never sends to bridge

### Bridge Flow (per run, every 5 min)
1. Get pending signals from DB per config (ABS(score) >= config threshold — supports BOTH LONG and SHORT)
2. Iterate `CONFIG_PRIORITY` top-down (46 configs)
3. For each config: validate signal (entry/stop/target, effective R:R, regime, whitelist, blocklist, Binance Futures availability)
4. **Live R:R validation:** fetches Binance mark price and rejects if effective R:R < 0.8 (setup degraded)
5. **Config diversity:** max 1 signal per config per batch
6. **Symbol dedup:** no duplicate symbols per batch
7. **HTTP retry:** up to 3 attempts with backoff on failure
8. Stop when `MAX_SIGNALS_PER_BATCH` reached

## The 5 Tiers

```python
CONFIG_PRIORITY = [
    # ── TIER 1: Best EV (backtested) ──
    ("3.1", 7.0, "V3.1 ADX Trend"),         # EV=+7.05%
    ("4.1", 7.0, "V4.1 Breakout+Vol"),       # EV=+5.55%
    ("5.1", 7.0, "V5.1 AI-focused"),         # EV=+5.17%
    ("6.1", 7.0, "V6.1 Breakout Momentum"),  # EV=+4.87%
    ("2.1", 7.0, "V2.1 MT Alignment"),       # EV=+3.65%
    ("4.0", 7.0, "V4.0 TR/ATR Breakout"),    # EV=+2.92%

    # ── TIER 2: Solid EV ──
    ("1.4", 7.0, "V1.4 Scanner-Focused"),    # EV=+2.23% (ACTIVE)
    ("1.5", 7.0, "V1.5 Conservative R:R"),   # best execution EV
    ("2.2", 7.0, "V2.2 Soft MT Alignment"),  # EV=+1.92%
    ("8.0", 7.0, "V8.0 Funding Rate"),       # EV=+1.76%
    ("6.0", 7.0, "V6.0 Pullback"),           # EV=+1.70%
    ("3.2", 7.0, "V3.2 Soft ADX"),           # EV=+1.46%
    ("5.2", 7.0, "V5.2 Balanced DeFi/AI"),   # EV=+1.09%
    ("6.2", 7.0, "V6.2 Pullback Strat"),     # EV=+0.72%
    ("14.0", 7.0, "V14.0 Precursor BTC/ETH"), # EV=+0.45% (min_precursors=2)
    ("14.1", 7.0, "V14.1 Precursor SHORT BTC/ETH"), # SHORT-focus variant
    ("5.0", 7.0, "V5.0 DEFI-focused"),       # EV=+0.05%

    # ── TIER 3: Negative EV / special ──
    ("10.0", 7.0, "V10.0 Chart Patterns"),   # 90.9% executed WR
    ("12.0", 7.0, "V12.0 Optimized Pro V2"),
    ("16.0", 7.0, "V16.0 Vol Squeeze"),      # Most execution data
    ("11.0", 7.0, "V11.0 Optimized Pro"),
    ("13.0", 7.0, "V13.0 Auto-Evolved"),

    # ── TIER 4: Liquidation-driven (new, multi-symbol) ──
    ("22.0", 7.0, "V22.0 Liquidation LONG"),   # Short squeeze detection
    ("22.1", 7.0, "V22.1 Liquidation SHORT"),  # Long squeeze detection

    # ── TIER 5: Rotating configs (data collection) ──
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
Tier 5 configs are rotated daily via `day % 7` offset so they get a chance to accumulate execution data. Without rotation, higher-priority configs always fill the batch first.

## Constants

```python
MAX_SIGNALS_PER_BATCH = 8       # Matches the 8 concurrent trade slots
MAX_SCORE_CAP = 15.0            # 12+ scores have 57.1% WR, EV=+4.12%
EXCLUDED_SYMBOLS = {"BTWUSDT", "EULUSDT", "EIGENUSDT", "MORPHOUSDT", "DGBUSDT"}
MAX_SLIPPAGE_PCT = 0.5          # Max price diff from entry before skip
MIN_EFFECTIVE_RR = 0.8          # Reject if live R:R < floor at send time
HTTP_RETRIES = 3                # Retry on timeout/503
HTTP_RETRY_DELAY = 1.0          # Base backoff in seconds
```

## Signal Validation Chain (order matters)

| # | Check | Skip reason |
|---|-------|-------------|
| 1 | Hard validation: entry/stop/target valid | "REJECTED" |
| 2 | **Live R:R check (reject if degraded)** 🆕 | **"DEGRADED (eff_RR < 0.8 or setup completed)"** |
| 3 | Config not disabled | "config is disabled" |
| 4 | Symbol in whitelist (if set) | "not in config whitelist" |
| 5 | Market regime filter (BTC) | "REGIME BLOCKED" |
| 6 | Time-of-day filter | "TIME BLOCKED" |
| 7 | Symbol not in EXCLUDED_SYMBOLS | "excluded symbol (0% WR)" |
| 8 | Score capped at MAX_SCORE_CAP | "score capped" |
| 9 | On Binance Futures | "not on Binance Futures" |
| 10 | Actively TRADING (not PENDING_TRADING) | "not actively TRADING" |
| 11 | On TestNet (if ACCOUNT_TYPE=TestNet) | "not on TestNet" |
| 12 | Config diversity (not already selected) | "config already contributed" |

## SHORT Signal Support

Previously, the bridge SQL used `composite_score >= ?` which **excluded** SHORT signals (stored as negative scores). Fixed by using `ABS(composite_score)` — both LONG and SHORT signals now compete fairly.

- LONG: score >= min_score
- SHORT: abs(score) >= min_score
- Direction is preserved and validated correctly (stop < entry < target for LONG, inverted for SHORT)

## Live Effective R:R Validation

At send time, the bridge fetches the Binance mark price and checks the **actual R:R** you'd get entering at current price with the original TP/SL levels:

- **LONG:** risk = live - stop, reward = target - live
- **SHORT:** risk = stop - live, reward = live - target
- If reward <= 0 (price past target) → "setup already completed, skip"
- If risk <= 0 (price past stop) → "no risk left"
- If eff_RR < MIN_EFFECTIVE_RR (0.8) → "degraded, skip"

This prevents the stale-signal scenario: signal detected at candle close but price moved before execution, destroying the original R:R.

## HTTP Retry

If the webhook endpoint returns 503 or times out, up to 3 attempts are made with exponential backoff (1s, 2s, 3s). Transient failures no longer lose signals.

## Cron Setup

```cron
# Edge scanner (every 5 min) — runs scan + bridge in-process
*/5 * * * * cd /home/hermes/BacktestingMCP && bash ~/.hermes/scripts/edge_scan.sh

# Standalone bridge (every 5 min) — backup if scan fails
*/5 * * * * cd /home/hermes/BacktestingMCP && . .env && venv/bin/python -m src.edge_scanner.webhook_bridge

# Liquidation WS daemon keepalive (every 5 min)
*/5 * * * * bash ~/.hermes/scripts/liq_ws_daemon.sh
```

## Config EV Source (how tiers were ranked)

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
   pip install websockets  # for liquidation WS daemon
   ```

3. **Verify bridge imports:**
   ```bash
   python -c "from src.edge_scanner.webhook_bridge import CONFIG_PRIORITY, MAX_SIGNALS_PER_BATCH; print(len(CONFIG_PRIORITY), 'configs, batch=', MAX_SIGNALS_PER_BATCH)"
   ```

4. **Test selection logic (dry run):**
   ```bash
   python -c "from src.edge_scanner.webhook_bridge import select_signals; print(select_signals())"
   ```

5. **Start liquidation WS daemon:**
   ```bash
   bash ~/.hermes/scripts/liq_ws_daemon.sh
   ```

6. **Check DB exists:** `data/crypto.db` with populated `edge_signals` table

7. **Cron setup (see above)**

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Config never sends signals | Not in CONFIG_PRIORITY | Add to list |
| Batch fills with 1 config | Diversity check disabled | Verify `selected_configs` logic |
| SHORT signals never send | Score stored negative | Fixed: bridge uses ABS(score) |
| Signals sent hours late | Scan failed, bridge didn't run | Fixed: edge_scan.sh always runs bridge |
| "DEGRADED" for most signals | Price moved before bridge ran | Cron now every 5m, in-process bridge |
| Coinglass shows "upgrade required" | Replaced with Binance WS/taker | Now uses free Binance sources |

## Related Docs
- `docs/ARCHITECTURE.md` — System architecture overview
- `docs/STRATEGY_EVOLUTION.md` — Strategy change log & reasoning
- `src/edge_scanner/webhook_bridge.py` — The bridge implementation
- `src/integrations/binance_liq_ws.py` — Liquidation WS daemon + REST fallback