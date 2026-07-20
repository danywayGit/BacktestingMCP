# Webhook Bridge — Signal Format Reference

## Message Format

Each signal sent to the Trading-WebHook-Bot is a newline-separated string
in the `msg` field of a JSON POST to `/webhook`.

### Required Fields

| Field | Example | Notes |
|-------|---------|-------|
| `Username` | `Danyway` | Bot user account |
| `AccountType` | `TestNet` | `TestNet` or `Standard` |
| `Exchange` | `Binance` | Always Binance |
| `Strategy` | `ManualTrading` | Currently only ManualTrading |
| `Action` | `OpenLong` | `OpenLong` or `OpenShort` |
| `Side` | `BUY` | `BUY` for Long, `SELL` for Short |
| `Symbol` | `CFXUSDT` | **Must end with USDT**. No plain symbol. |
| `Entry` | `0.04724` | Single price only (no comma lists) |
| `StopLoss` | `0.04663` | Must be below entry for LONG, above for SHORT |
| `TakeProfit` | `0.04844` | Must be above entry for LONG, below for SHORT |

### Optional Fields

| Field | Example | Notes |
|-------|---------|-------|
| `Score` | `10.0` | Edge scanner composite score |
| `ConfigVersion` | `1.0` | Config that generated the signal |

### Unused Fields (will warn but not fail)

`Exit`, `TakeProfit1`, `TakeProfit2`, `TakeProfit3` — not currently used.

## Symbol Format Rules

| ✅ Correct | ❌ Wrong |
|-----------|---------|
| `CFXUSDT` | `CFX` |
| `BTCUSDT` | `BTC` |
| `ETHUSDT` | `ETH/USDT` |

The edge scanner stores symbols as `CFX` (no USDT). The bridge
**appends `USDT`** automatically before sending.

## Entry Price Precision

Low-priced coins ($0.01-$1.00) need enough decimal places.

| Coin Price | Example Entry | Decimal Places |
|-----------|--------------|----------------|
| > $100 | 555.50 | 2 |
| $1-$100 | 0.4504 | 4 |
| $0-$1 | 0.04724 | 5 |
| < $0.01 | 0.0001947 | 7 |

## Stop Loss Constraints

| Constraint | Reason |
|-----------|--------|
| Stop ≠ Entry | "Order would immediately trigger" error |
| Stop must give ≥ 0.5% room | Too tight = immediate trigger risk |
| Risk % < 1.5% | Bot's max `risk_per_trade` |
| Stop closer than entry on LONG | Entry = 0.04724, Stop must be < 0.04724 |

## Take Profit Constraints

| Constraint | Reason |
|-----------|--------|
| TP must give ≥ 3% room on low-price coins | Price noise eats small targets |
| R:R ratio ≥ 1.5 | Bot's `min_risk_reward` |

## Edge Scanner → Bridge Priority

| Priority | Config | Min Score | Strategy Type |
|----------|--------|-----------|--------------|
| 1 | V7.0 | 7.5 | Quality Gate |
| 2 | V6.2 | 7.5 | Pullback |
| 3 | V4.1 | 7.5 | Breakout |
| 4 | V9.0 | 7.0 | Volume Imbalance |
| 5 | V1.0 | 8.5 | Baseline |

## Webhook JSON Payload

```json
{
  "key": "your_webhook_secret",
  "telegram_alert_type": "trading_bot",
  "msg": "Username: Danyway\nAccountType: TestNet\nExchange: Binance\nStrategy: ManualTrading\nAction: OpenLong\nSide: BUY\nSymbol: CFXUSDT\nEntry: 0.04724\nStopLoss: 0.04663\nTakeProfit: 0.04844\nScore: 10.0\nConfigVersion: 1.0"
}
```

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `Invalid symbol` | Symbol missing USDT suffix | Use `CFXUSDT` not `CFX` |
| `Order would immediately trigger` | Stop loss = entry price | Widen the stop |
| `Trade validation failed` | Risk > bot's max or R:R too low | Widen TP or narrow stop |