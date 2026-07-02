# Integration Plan — Edge Scanner → Trading-WebHook-Bot

## Architecture

```
BacktestingMCP                        Trading-WebHook-Bot
─────────────────                     ────────────────────
● altFINS scoring                     ● Flask webhook server
● Config evolution (V1-V8)            ● Binance order execution
● Signal discovery                    ● Risk management (loss limits, max trades)
● Telegram alerts ───→ signal ───→    ● Position management
  with R:R sizing                     ● Funding fee tracking (cron)
  and breakeven WR                    ● Dashboard + Telegram cmds
● Gem scanner (spot & futures) ────→  ● Optional: auto-execute gems
```

## Current Bridge

The edge scanner already sends signals to the user via Telegram. The webhook bot can be
extended to read these signals (or better, read directly from the database) and execute
trades.

### Data Flow Options

1. **Telegram Forwarding** (simplest): Forward the alert message to a bot that parses it.
2. **Database Polling** (recommended): The bot queries `edge_signals` for new PENDING
   signals with a specific config_version or source.
3. **Webhook Push** (future): Modify `alerts.py` to POST to an endpoint when a signal is
   generated.

## Gem Scanner Integration

The gem scanner (`src/edge_scanner/gem_scanner.py`) is now a standalone module that:
- Scans CoinGecko for Binance-listed coins (Spot OR Futures)
- Applies tokenomics filters: market cap, volume/mcap, ATH distance, supply dynamics,
  coin age (< 2 years)
- Outputs a ranked list of gems for 3-6 month holds

It can be run manually or via cron (already set up for weekly runs).

## Planned Execution-Side Improvements (in Trading-WebHook-Bot)

When you're ready to enhance the execution bot, consider:

### P0 (Critical)
- [ ] **Capital Tracking**: Track available USDT and used margin.
- [ ] **Position Sizing**: Use the R:R-adjusted recommendation from the alert.
- [ ] **Min R:R Validation**: Only take signals where the config's R:R >= user threshold.
- [ ] **Max Concurrent Positions**: Limit number of open trades.

### P1 (Important)
- [ ] **Breakeven-Aware Sizing**: Reduce size or skip if config's win rate < breakeven.
- [ ] **Funding Fee Tracking**: Subtract estimated funding from target for futures.
- [ ] **Correlation Filter**: Avoid highly correlated positions (e.g., BTC and ETH).
- [ ] **Trailing Stop**: Implement ATR-based trailing stop.

### P2 (Nice-to-have)
- [ ] **Dynamic Resolution Horizon**: Exit based on ATR or time, not fixed horizon.
- [ ] **Score-Based Position Sizing**: Scale size by signal score (e.g., 9.0 = 2x size of 7.0).
- [ ] **Multi-Timeframe Confirmation**: Require signal alignment on higher TF.
- [ ] **Volume Profile Activation**: Use volume/liquidity zones for entries/exits.

### P3 (Future)
- [ ] **Auto-Trading from Gem Scanner**: Execute buys on high-score gems with DCA.
- [ ] **ML-Based Filter**: Train model on past signal outcomes.
- [ ] **Portfolio Rebalancing**: Periodically rebalance to target allocations.

## Immediate Next Steps

1. **Test the gem scanner**: Run `python -m src.cli.main edge gems --pages 5 --start-page 3 --top 30`
2. **Review the cron job**: `gem-scan` runs Mondays at 08:00 UTC.
3. **When ready to trade**: Implement the P0/P1 items in Trading-WebHook-Bot.
4. **Optional**: Add a webhook endpoint in the bot to receive real-time signals.

## Notes

- The gem scanner is **not** a trading signal — it's a research tool for long-term holds.
- The edge scanner continues to run every 30 minutes for short-term swing signals.
- Both systems can feed the same execution bot with different strategies.