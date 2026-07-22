# Edge Scanner

Systematic crypto trading signal generation and validation system.
Discovers profitable entry/stop/target parameters through multi-config
scoring, validates forward performance, and executes on Binance Futures.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/danywayGit/BacktestingMCP.git
cd BacktestingMCP
bash setup/setup.sh

# 2. Edit .env with your API keys
nano .env

# 3. Test the CLI
source venv/bin/activate
python -m src.cli.main edge configs

# 4. Install cron jobs
bash setup/install_crons.sh
```

## Required API Keys

| Service | Purpose | Free? |
|---------|---------|-------|
| altFINS | Chart patterns + screener | Paid |
| Binance | OHLCV data, funding rates | ✅ Free |
| CoinGecko | Market data, tokenomics | ✅ Free (10-30 req/min) |
| Telegram | Alerts to your channel | ✅ Free |
| OpenRouter | LLM auto-evolver (optional) | Paid (~$0.50/mo) |

## CLI Commands

```bash
# Scan for signals
python -m src.cli.main edge scan --multi

# Daily summary
python -m src.cli.main edge daily-summary

# Gem discovery
python -m src.cli.main edge gems --pages 4 --top 20

# Config performance
python -m src.cli.main edge report --breakeven

# LLM auto-evolve
python -m src.cli.main edge auto-evolve

# Chart patterns
python -m src.cli.main edge patterns

# List configs
python -m src.cli.main edge configs
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system diagram.

Three layers:
1. **Input** — altFINS screener, Binance OHLCV, CoinGecko, Tokenomist
2. **Scoring** — 28 configs × multi-factor score → edge_signals DB
3. **Execution** — webhook_bridge → Trading-WebHook-Bot → Binance

## Repos

| Repo | Purpose |
|------|---------|
| [BacktestingMCP](https://github.com/danywayGit/BacktestingMCP) | Signal generation (this repo) |
| [Trading-WebHook-Bot](https://github.com/danywayGit/Trading-WebHook-Bot) | Signal execution |

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture + data flow |
| [WEBHOOK_BRIDGE.md](docs/WEBHOOK_BRIDGE.md) | Webhook message format |
| [INTEGRATION_PLAN.md](docs/INTEGRATION_PLAN.md) | Future roadmap |
| [FUNDING_RATE_STRATEGY.md](docs/FUNDING_RATE_STRATEGY.md) | Funding rate strategy rules |
| `.env.template` | Environment configuration template |
| `setup/setup.sh` | Automated setup script |
| `setup/install_crons.sh` | Cron job installer |