# Edge Scanner — Architecture

## Overview

Edge Scanner is a systematic crypto trading signal generation and validation
system. It discovers profitable entry/stop/target parameters through
walk-forward optimization, scores candidates across multiple signal sources,
logs for forward validation, and surfaces the best performers.

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                                     │
│                                                                           │
│  altFINS MCP ─────→ Screener (116+ candidates)                           │
│  Binance API ─────→ OHLCV (1h/4h/1d) + Funding rates                    │
│  CoinGecko  ─────→ Market data (MCap, volume, supply)                   │
│  Tokenomist ─────→ Burn events + Token unlocks                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          SCORING LAYER                                    │
│                                                                           │
│  composite.py ─────→ Multi-factor score per symbol                        │
│    ├─ trend_weight (1.0 — V1.4)                                          │
│    ├─ volume_relative_weight (1.0)                                        │
│    ├─ signal_feed_weight (1.0)                                            │
│    ├─ scanner_hit_weight (2.0 — V1.4, doubled)                           │
│    ├─ onchain_netflow_weight (1.0)                                        │
│    ├─ chart_pattern_weight (5.0 — V10.0)                                 │
│    └─ volume_divergence_weight (3.0 — V3.x)                              │
│                                                                           │
│  30+ configs (V1.0 → V20.0) ──→ Each produces scored signals             │
│  ACTIVE_CONFIG = V1.4 ──→ Triggers Telegram alerts                       │
│                                                                           │
│  Multi-source check now uses own OHLCV data (EMA20, volume rel to 10MA)  │
│  for earlier signal detection (no altFINS dependency)                     │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          SIGNAL LAYER                                     │
│                                                                           │
│  edge_signals DB ───→ PENDING signals (score, entry, stop, target)       │
│  Resolution (01/13/17 UTC) ──→ WIN/LOSS/FLAT by OHLCV check             │
│  Evolution engine ──→ z-test config performance                         │
│  LLM auto-evolver ──→ Generates new configs (V7.5, V7.6, etc.)          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          EXECUTION LAYER                                   │
│                                                                           │
│  webhook_bridge.py ──→ 11 config priority queue                          │
│    ├─ V10.0 (≥7.0)  Chart Pattern Hunter (90.9% WR 🏆)                   │
│    ├─ V20.0 (≥5.0)  Time-of-Day                                          │
│    ├─ V19.0 (≥5.0)  Ratio Arb                                            │
│    ├─ V18.0 (≥7.0)  Mean Reversion                                       │
│    ├─ V17.0 (≥7.0)  Liquidation                                          │
│    ├─ V16.0 (≥7.0)  Vol Squeeze                                          │
│    ├─ V15.0 (≥7.0)  Multi-TF                                             │
│    ├─ V6.4  (≥7.0)  Flat Killer                                          │
│    ├─ V1.5  (≥7.0)  Conservative R:R                                     │
│    ├─ V1.4  (≥7.0)  Scanner-Focused                                      │
│    └─ V14.0 (≥7.0)  Pattern Discovery (BTC/ETH only, precursors)         │
│                                                                           │
│  Dedup rules: 1 signal/symbol per batch, max 3/batch, open-position      │
│  Validation: entry > 0, stop < entry < target (LONG)                     │
│  Blocklist: BTWUSDT, EULUSDT, EIGENUSDT, MORPHOUSDT, DGBUSDT            │
│  Slippage: MAX_SLIPPAGE_PCT = 0.5% (price too far from entry → skip)    │
│                                                                           │
│  └──→ POST /webhook ──→ Trading-WebHook-Bot ──→ Binance Futures         │
│                           (TestNet)                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Config Versions (Active Only)

| Version | Strategy | WR | Threshold | Notes |
|---------|----------|----|-----------|-------|
| **V1.4** 🏆 | **Scanner-Focused (ACTIVE)** | **58.3%** | **7.0** | `scanner_hit_weight=2.0`, atr_stop=3.0, rr=1.5 |
| **V1.5** | **Conservative R:R** | **44.4%** | **7.0** | rr_ratio=1.2, same as V1.0 formula |
| V3.6 | Bridge-Active ADX | N/A | 7.0 | First V3.x to actually send signals |
| V6.4 | Flat Killer | N/A | 7.0 | Tight stop, high vol, rr=1.2 |
| **V10.0** 🥇 | **Chart Pattern Hunter** | **90.9%** | **7.0** | `chart_pattern_weight=5.0` |
| V14.0 | Pattern Discovery | N/A | 7.0 | BTC/ETH only, precursor-based |
| V16.0 | Vol Squeeze | 45.7% | 6.0 | BB squeeze breakout |
| V20.0 | Time-of-Day | N/A | 5.0 | Filters by active hours |

### Disabled Configs
V1.0 (38.2% WR), V2.0, V3.0 (43.8%), V6.3 (14.3% WR, EV=-3.50%), V7.0, V7.7

## Multi-Source Alert Check

Signals require ≥ 2 sources to trigger Telegram alert. Sources now include:

| Source | Threshold | Data Origin |
|--------|-----------|-------------|
| altFINS trend | ≥ 7 (LONG) / ≤ -7 (SHORT) | altFINS |
| altFINS signal feed | BULLISH/BEARISH | altFINS |
| altFINS volume | **≥ 1.5×** (was 2.0) | altFINS |
| **Own volume** 🆕 | **≥ 1.5× of 10MA** | **Our market_data** |
| **Volume accumulation** 🆕 | **Increasing over 3 candles** | **Our market_data** |
| **Price > EMA20 + vol ≥ 1.2×** 🆕 | **Counts as 2 sources** | **Our market_data** |
| Scanner hit | Any pattern triggered | TA scanner |
| Onchain netflow | > 0.05 (LONG) / < -0.05 (SHORT) | CoinGecko |

**Benefit:** Early signal detection — catches setups during accumulation phase, 16h+ before previous reactive signals.

## Alert Dedup

- **24h cooldown** per (symbol, config, direction) — file-backed JSON cache
- **Unresolved check** — skip if signal already sent to bot and still open
- **Once per signal** — no re-alerting until resolved

## Cron Schedule

| Job | Schedule | Purpose |
|-----|----------|---------|
| edge-scan | Every 15 min | Score symbols, log signals, send alerts |
| edge-track | 01/13/17 UTC | Resolve PENDING → WIN/LOSS/FLAT |
| webhook-bridge | Every 15 min | Send high-score signals to bot |
| daily-summary | 09:00 UTC | Telegram report |
| pattern-scan | 10:00 UTC | Chart pattern detection |
| evolution-check | 18:00 UTC | Config performance analysis |
| gem-scan | Monday 08:00 UTC | Weekly gem discovery |
| burn-tracker | Saturday 10:00 UTC | Token buyback/burn events |
| funding-poll | Every 15 min | Refresh funding rate cache |

## Key Files

| File | Purpose |
|------|---------|
| `src/cli/main.py` | CLI entry point (`edge scan`, `edge gems`, etc.) |
| `src/edge_scanner/scoring_config.py` | All 30+ config definitions |
| `src/edge_scanner/composite.py` | Multi-factor scoring engine + early signal detection |
| `src/edge_scanner/alerts.py` | Telegram alert formatting + dedup + multi-source check |
| `src/edge_scanner/store.py` | Signal logging + resolution |
| `src/edge_scanner/evolution.py` | Config comparison + z-test |
| `src/edge_scanner/gem_scanner.py` | CoinGecko gem discovery + social metrics |
| `src/edge_scanner/gem_metrics_store.py` | Gem social/developer metric snapshots for progression |
| `src/edge_scanner/webhook_bridge.py` | Signal → bot bridge |
| `src/edge_scanner/burn_tracker.py` | Burn event monitoring |

## Dependencies

- **Python 3.11+** — Core runtime
- **SQLite** — Database (no server needed)
- **altFINS MCP** — Chart pattern + screener data
- **Binance API** — OHLCV data, funding rates
- **CoinGecko API** — Market data, tokenomics
- **OpenRouter API** — LLM auto-evolver (optional)