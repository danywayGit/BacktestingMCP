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
│    ├─ trend_weight (0.4)                                                  │
│    ├─ volume_relative_weight (0.2)                                        │
│    ├─ signal_feed_weight (0.3)                                            │
│    ├─ onchain_netflow_weight (0.1)                                        │
│    ├─ volume_divergence_weight (3.0)                                      │
│    ├─ volume_imbalance_weight (5.0 — V9.0)                                │
│    └─ funding_rate_weight (5.0 — V8.0)                                   │
│                                                                           │
│  28 configs (V1.0 → V9.0) ──→ Each produces scored signals               │
│  ACTIVE_CONFIG = V7.0 ──→ Triggers Telegram alerts                       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          SIGNAL LAYER                                     │
│                                                                           │
│  edge_signals DB ───→ PENDING signals (score, entry, stop, target)       │
│  Resolution (01/13/17 UTC) ──→ WIN/LOSS/FLAT by OHLCV check             │
│  Evolution engine ──→ z-test config performance                         │
│  LLM auto-evolver ──→ Generates new configs (V7.5, V7.6...)             │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          EXECUTION LAYER                                   │
│                                                                           │
│  webhook_bridge.py ──→ 5 config priority queue                           │
│    ├─ V7.0 (≥7.5)  Quality Gate                                          │
│    ├─ V6.2 (≥7.5)  Pullback                                              │
│    ├─ V4.1 (≥7.5)  Breakout                                              │
│    ├─ V9.0 (≥7.0)  Volume Imbalance                                      │
│    └─ V1.0 (≥8.5)  Baseline                                              │
│                                                                           │
│  Dedup rules: 1 signal/symbol, max 3/batch, open-position check         │
│  Validation: entry > 0, stop < entry < target (LONG)                     │
│                                                                           │
│  └──→ POST /webhook ──→ Trading-WebHook-Bot ──→ Binance Futures         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Config Versions

| Version | Strategy | WR | Status |
|---------|----------|----|--------|
| V1.0-V1.4 | Baseline variants | 50-52% | 🟢 Enabled |
| V2.0 | Multi-timeframe (hard gate) | 50.6% | ⚪ Disabled |
| V2.1-V2.2 | Multi-timeframe (soft bonus) | 55-60% | 🟢 Enabled |
| V3.0 | ADX trend (hard gate) | 43.8% | ⚪ Disabled |
| V3.1-V3.2 | ADX trend (soft bonus) | 57-59% | 🟢 Enabled |
| V4.0-V4.1 | Breakout intensity | 56-58% | 🟢 Enabled |
| V5.0-V5.2 | Coin-type specific | 55-65% | 🟢 Enabled |
| V6.0-V6.2 | Pullback/Breakout | 54-63% | 🟢 Enabled |
| **V7.0** | **Quality Gate (active)** | **50%** | 🟢 **Active** |
| V7.5-V7.7 | LLM-evolved (relaxed) | 20-24% | ⚪ Disabled |
| V8.0 | Funding rate mean-reversion | 44% | 🟢 Enabled |
| **V9.0** | **Volume Imbalance (NEW)** | **N/A** | 🟢 Enabled |

## Cron Schedule

| Job | Schedule | Purpose |
|-----|----------|---------|
| edge-scan | Every 30 min | Score symbols, log signals |
| edge-track | 01/13/17 UTC | Resolve PENDING → WIN/LOSS/FLAT |
| daily-summary | 09:00 UTC | Telegram report |
| pattern-scan | 10:00 UTC | Chart pattern detection |
| evolution-check | 18:00 UTC | Config performance analysis |
| gem-scan | Monday 08:00 UTC | Weekly gem discovery |
| burn-tracker | Saturday 10:00 UTC | Token buyback/burn events |
| webhook-bridge | Every 30 min | Send high-score signals to bot |
| funding-poll | Every 15 min | Refresh funding rate cache |

## Key Files

| File | Purpose |
|------|---------|
| `src/cli/main.py` | CLI entry point (`edge scan`, `edge gems`, etc.) |
| `src/edge_scanner/scoring_config.py` | All 28 config definitions |
| `src/edge_scanner/composite.py` | Multi-factor scoring engine |
| `src/edge_scanner/store.py` | Signal logging + resolution |
| `src/edge_scanner/evolution.py` | Config comparison + z-test |
| `src/edge_scanner/llm_evolver.py` | LLM-based config generation |
| `src/edge_scanner/gem_scanner.py` | CoinGecko gem discovery |
| `src/edge_scanner/webhook_bridge.py` | Signal → bot bridge |
| `src/edge_scanner/burn_tracker.py` | Burn event monitoring |
| `src/edge_scanner/patterns.py` | Chart pattern detection |
| `src/edge_scanner/volume_divergence.py` | Volume analysis functions |
| `src/integrations/binance_funding.py` | Funding rate API |
| `src/integrations/binance_symbols.py` | Binance listing filter |
| `docs/WEBHOOK_BRIDGE.md` | Webhook message format reference |
| `docs/INTEGRATION_PLAN.md` | Future integration roadmap |

## Dependencies

- **Python 3.11+** — Core runtime
- **SQLite** — Database (no server needed)
- **altFINS MCP** — Chart pattern + screener data
- **Binance API** — OHLCV data, funding rates
- **CoinGecko API** — Market data, tokenomics
- **OpenRouter API** — LLM auto-evolver (optional)