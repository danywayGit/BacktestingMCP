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
│    ├─ volume_divergence_weight (3.0 — V3.x)                              │
│    ├─ bb_squeeze_weight (2.0 — V14.0)                                    │
│    ├─ atr_expansion_weight (1.5 — V14.0)                                 │
│    ├─ bb_position_weight (0.5-1.0 — V14.x)                               │
│    └─ liquidation_weight (1.5-5.0 — V14.x/V22.x)                        │
│                                                                           │
│  46+ configs (V1.0 → V22.1) ──→ Each produces scored signals             │
│  ACTIVE_CONFIG = V1.4 ──→ Triggers Telegram alerts                       │
│                                                                           │
│  Multi-source check uses own OHLCV data (EMA20, volume rel to 10MA)      │
│  for earlier signal detection                                             │
│                                                                           │
│  Liquidation data from Binance WS + takerlongshortRatio (free, no key)   │
│    - !forceOrder@arr stream (real liquidation events)                    │
│    - takerlongshortRatio REST (orderflow proxy)                          │
│    - globalLongShortAccountRatio (L/S positioning)                       │
│  (Coinglass removed — upgrade required)                                  │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          SIGNAL LAYER                                     │
│                                                                           │
│  edge_signals DB ───→ PENDING signals (score, entry, stop, target)       │
│  Resolution (02/13/17 UTC) ──→ WIN/LOSS/FLAT by OHLCV check             │
│  Evolution engine ──→ z-test config performance                         │
│  LLM auto-evolver ──→ Generates new configs (V7.5, V7.6, etc.)          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          EXECUTION LAYER                                   │
│                                                                           │
│  webhook_bridge.py ──→ 46 config priority queue (5 tiers)                │
│    ├─ TIER 1 (Best EV): V3.1, V4.1, V5.1, V6.1, V2.1, V4.0             │
│    ├─ TIER 2 (Solid):   V1.4, V1.5, V2.2, V8.0, V6.0, V3.2, V5.2,      │
│    │                      V6.2, V14.0, V14.1, V5.0                       │
│    ├─ TIER 3 (Special): V10.0, V12.0, V16.0, V11.0, V13.0               │
│    ├─ TIER 4 (Liq):     V22.0, V22.1 (liquidation-driven)               │
│    └─ TIER 5 (Rotation): 10 rotating + 6 special-purpose configs        │
│                                                                           │
│  SHORT signal support (ABS score) 🆕                                     │
│  Live R:R validation (reject if eff_RR < 0.8) 🆕                        │
│  HTTP retry (3 attempts) 🆕                                              │
│                                                                           │
│  Dedup rules: 1 signal/config per batch, 8/batch, open-position          │
│  Validation: entry > 0, stop < entry < target (LONG), inverted (SHORT)  │
│  Blocklist: BTWUSDT, EULUSDT, EIGENUSDT, MORPHOUSDT, DGBUSDT            │
│  Slippage: MAX_SLIPPAGE_PCT = 0.5%                                       │
│                                                                           │
│  └──→ POST /webhook ──→ Trading-WebHook-Bot ──→ Binance Futures         │
│                           (TestNet)                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Config Versions (Active Only)

| Version | Strategy | WR | Threshold | Notes |
|---------|----------|----|-----------|-------|
| **V1.4** 🏆 | **Scanner-Focused (ACTIVE)** | **58.3%** | **7.0** | `scanner_hit_weight=2.0`, atr_stop=3.0, rr=1.3 |
| **V1.5** | **Conservative R:R** | **44.4%** | **7.0** | rr_ratio=1.2, same as V1.0 formula |
| V6.4 | Flat Killer | N/A | 7.0 | Tight stop, high vol, rr=1.2 |
| **V10.0** 🥇 | **Chart Pattern Hunter** | **90.9%** | **7.0** | `chart_pattern_weight=5.0` |
| V14.0 | Precursor Pattern (LONG) | N/A | 7.0 | BTC/ETH, min_precursors=2, BB squeeze + ATR |
| V14.1 | Precursor Pattern (SHORT) | N/A | 7.0 | BTC/ETH, bear bias, SHORT-focused |
| V16.0 | Vol Squeeze | 45.7% | 7.0 | BB squeeze breakout |
| V20.0 | Time-of-Day | N/A | 5.0 | Filters by active hours |
| **V22.0** 🆕 | **Liquidation LONG** | **N/A** | **7.0** | **Short squeeze, 5x liq weight, multi-symbol** |
| **V22.1** 🆕 | **Liquidation SHORT** | **N/A** | **7.0** | **Long squeeze, 5x liq weight, multi-symbol** |

### Disabled Configs
V1.0 (38.2% WR), V2.0, V3.0 (43.8%), V3.6 (0% WR), V6.3 (14.3% WR), V7.0, V7.7

## Multi-Source Alert Check

Signals require ≥ 2 sources to trigger Telegram alert. Sources now include:

| Source | Threshold | Data Origin |
|--------|-----------|-------------|
| altFINS trend | ≥ 7 (LONG) / ≤ -7 (SHORT) | altFINS |
| altFINS signal feed | BULLISH/BEARISH | altFINS |
| altFINS volume | ≥ 1.5× | altFINS |
| **Own volume** | ≥ 1.5× of 10MA | Our market_data |
| **Volume accumulation** | Increasing over 3 candles | Our market_data |
| **Price > EMA20 + vol ≥ 1.2×** | Counts as 2 sources | Our market_data |
| Scanner hit | Any pattern triggered | TA scanner |
| Onchain netflow | > 0.05 (LONG) / < -0.05 (SHORT) | CoinGecko |

**Benefit:** Early signal detection — catches setups during accumulation phase, 16h+ before previous reactive signals.

## Alert Dedup

- **24h cooldown** per (symbol, config, direction) — file-backed JSON cache
- **Unresolved check** — skip if signal already sent to bot and still open
- **Once per signal** — no re-alerting until resolved

## Liquidation Data (New Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                    LIQUIDATION DATA FLOW                          │
│                                                                   │
│  1. !forceOrder@arr WS (best)                                    │
│     Binance public WebSocket ──→ daemon (persistent)              │
│     └──→ writes to data/liquidation_snapshot.json                │
│                                                                   │
│  2. takerlongshortRatio (fallback)                               │
│     Binance public REST ──→ buy/sell ratio + volume              │
│     ↳ No key needed, always available                            │
│                                                                   │
│  3. globalLongShortAccountRatio (final fallback)                 │
│     Binance public REST ──→ account positioning proxy            │
│                                                                   │
│  Daemon keepalive: cron every 5 min                              │
│  All free, no API keys                                           │
└─────────────────────────────────────────────────────────────────┘
```

## Cron Schedule (Updated Aug 21)

| Job | Schedule | Purpose |
|-----|----------|---------|
| edge-scan | Every 5 min | Score symbols, log signals, send alerts + bridge |
| edge-track | 02/13/17 UTC | Resolve PENDING → WIN/LOSS/FLAT |
| webhook-bridge | Every 5 min | Backup bridge if in-process one fails |
| liq-daemon-ensure | Every 5 min | Ensure WS liquidation daemon is alive |
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
| `src/edge_scanner/scoring_config.py` | All 46+ config definitions |
| `src/edge_scanner/composite.py` | Multi-factor scoring engine + precursor detection |
| `src/edge_scanner/alerts.py` | Telegram alert formatting + dedup + multi-source check |
| `src/edge_scanner/store.py` | Signal logging + resolution + entry price live fetch |
| `src/edge_scanner/evolution.py` | Config comparison + z-test |
| `src/edge_scanner/gem_scanner.py` | CoinGecko gem discovery + social metrics |
| `src/edge_scanner/gem_metrics_store.py` | Gem social/developer metric snapshots for progression |
| `src/edge_scanner/webhook_bridge.py` | Signal → bot bridge (46 configs, SHORT support, live R:R) |
| `src/edge_scanner/burn_tracker.py` | Burn event monitoring |
| `src/integrations/binance_liq_ws.py` | Liquidation WS daemon + REST fallback (free, no key) |

## Dependencies

- **Python 3.11+** — Core runtime
- **SQLite** — Database (no server needed)
- **altFINS MCP** — Chart pattern + screener data
- **Binance API** — OHLCV data, WS liquidation feed, taker ratio
- **CoinGecko API** — Market data, tokenomics
- **OpenRouter API** — LLM auto-evolver (optional)
- **websockets** — For Binance liquidation WS daemon