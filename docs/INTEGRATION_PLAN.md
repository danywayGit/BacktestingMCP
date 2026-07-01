# Integration Plan — Edge Scanner → Trading-WebHook-Bot

## Architecture

```
BacktestingMCP                        Trading-WebHook-Bot
─────────────────                     ────────────────────
● altFINS scoring / signal discovery  ● Flask webhook server
● Config evolution (21 configs)       ● Binance order execution
● Target/stop resolution              ● Risk management (loss limits, max trades)
● Funding rate tracking               ● Position sync (DB ↔ Exchange)
● Telegram alerts ──── signal ───→   ● Funding fee tracking (cron)
  with R:R & position sizing           ● Dashboard + Telegram cmds
  and breakeven status                 ● DCA Spot strategy
```

The two systems are independent repos. The bridge is the **Telegram alert** from BacktestingMCP — the edge scanner sends structured alerts with position sizing recommendations that the webhook bot can parse and execute.

## Current Bridge (signal data in alerts)

Each Telegram alert from BacktestingMCP contains:

| Field | Example | Source |
|-------|---------|--------|
| Symbol + Direction | `🟢 BTC — LONG` | altFINS scoring |
| Entry price | `$45000.00` | Current market price |
| Stop loss | `$44250.00` | ATR × atr_stop_mult |
| Target | `$46500.00` | Stop × rr_ratio |
| R:R | `1:2.0` | Config's rr_ratio |
| Position Risk | `1.0%` | Score tier × R:R adjustment |
| Config version | `V7.0` | Active config |
| Config breakeven status | 🟢 if WR > be_wr | `be_wr = 1 / (1 + rr)` |
| Funding rate | `-0.18%` | V8.0 module |

## Future Integration (Execution Layer Improvements)

These are improvements planned for Trading-WebHook-Bot to fully leverage the edge scanner data:

### 1. Position Sizing from Signal

The webhook bot should parse the alert's position risk percentage and:

- ✅ Validate against max position size per user
- ✅ Check capital availability
- ✅ Calculate exact contract quantity using current price
- ✅ Place limits order at signal entry

**Formula (already defined in BacktestingMCP alerts):**
```
position_risk = base_tier_pct × min(rr_ratio, 4.0) / 2.0
```

### 2. Minimum Risk-Reward Validation

The bot should reject signals where R:R is below a configurable threshold:

- Check `rr_ratio` from the signal
- Reject if `rr_ratio < MIN_RR` (configurable per user/strategy)
- Log rejection reason to `trades.db`

### 3. Breakeven-Aware Trading

Skip signals from config versions that are below breakeven:

- BacktestingMCP calculates `be_wr = 1 / (1 + rr)` for each config
- The bot checks if the signal's config has `WR ≥ be_wr`
- If below breakeven: either skip OR reduce position size (penalty factor)

**Penalty example:**
```
wr_ratio = actual_wr / be_wr       # e.g. 37.5 / 33.3 = 1.13 (above, no penalty)
if wr_ratio < 1.0:
    position_risk *= wr_ratio       # Reduce proportionally
```

### 4. Capital & Concurrency Tracking

The bot already tracks max concurrent trades (3 for Futures). Extend to:

- ✅ Per-symbol max positions (avoid over-concentration)
- ✅ Per-config max positions (don't overload one strategy)
- ✅ Capital allocation per strategy (% of total capital)
- ✅ Correlation check: skip signals correlated with existing open positions

### 5. Funding Fee Management

The bot already tracks funding fees via cron (`*/5 * * * *`). Extend to:

- ✅ Pause trading when funding fees exceed a threshold (uses V8.0 data)
- ✅ Factor funding cost into position sizing (funding eats into R:R)
- ✅ Auto-close before funding for pre-funding dip trades (Strategy 2 from FUNDING_RATE_STRATEGY.md)

**Formula:**
```
adjusted_rr = rr_ratio - (funding_cost_per_hour × expected_hold_hours / stop_distance)
```
If `adjusted_rr < MIN_RR`, skip the trade.

### 6. Webhook Format Enhancement

To pass all signal data, the webhook message sent from BacktestingMCP → Trading-WebHook-Bot needs additional fields beyond the current Telegram format:

```json
{
  "key": "sec_key",
  "type": "trading_bot",
  "msg": "Username: Danyway\nAccountType: Standard\nExchange: Binance\nStrategy: ManualTrading\nAction: OpenLong\nSymbol: BTCUSDT\nEntry: 45000\nStopLoss: 44250\nTakeProfit: 46500\nRiskPercent: 1.0\nRR: 2.0\nConfigVersion: 7.0\nBreakevenStatus: profitable"
}
```

### 7. User-Level Overrides

Allow per-user configuration overrides in Trading-WebHook-Bot:

| Setting | Default | Per-User Override |
|---------|---------|-------------------|
| `MIN_RR` | 1.5 | ✅ `UserConfig.rr_min` |
| `MAX_RISK_PCT` | 2.0% | ✅ `UserConfig.max_risk_pct` |
| `MAX_CONCURRENT` | 3 | ✅ `UserConfig.max_concurrent` |
| `ALLOW_BELOW_BE` | false | ✅ `UserConfig.allow_below_breakeven` |
| `FUNDING_PAUSE_THRESHOLD` | 0.6% | ✅ `UserConfig.funding_pause` |

### 8. Auto-Trading Mode (Future)

When the integration reaches production maturity, add an auto-trading cron in BacktestingMCP that directly calls the Trading-WebHook-Bot API endpoint with new high-confidence signals (score ≥ 9.0, profitable config, adequate capital).

## Priority Order

| Priority | Feature | Complexity | Impact |
|----------|---------|------------|--------|
| P0 | Min R:R validation | Low | Prevents bad trades |
| P0 | Capital + concurrency | Low | Safety net |
| P1 | R:R-adjusted sizing | Low | Optimal sizing |
| P1 | Breakeven check | Medium | Filters losing strategies |
| P2 | Funding fee adjustment | Medium | Realistic R:R |
| P2 | Webhook format upgrade | Medium | Structured data flow |
| P3 | User overrides | Medium | Flexibility |
| P3 | Auto-trading cron | High | Full automation |