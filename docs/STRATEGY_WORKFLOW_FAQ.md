# Strategy Generation, Optimization & Evaluation — FAQ

---

## Q1: Do I need to start the MCP server to generate a strategy?

**No.** The MCP server is only for AI assistants (Claude Desktop, VS Code Copilot, etc.).
Generate strategies directly with the CLI:

```bash
python -m src.cli.main strategy create \
  --description "Buy when RSI < 30 and price > 200 EMA. Exit when RSI > 70. Long only." \
  --name RSIOversoldLong \
  --provider auto \
  --register
```

`--register` auto-adds the strategy to `STRATEGY_REGISTRY` so you can immediately reference it by name.

---

## Q2: Can I generate a strategy without the MCP server?

**Yes.** The CLI command above works standalone.
The generated file is saved to `src/strategies/generated/{strategy_name}.py`.

---

## Q3: Do I need to set up an Ollama model, Anthropic model, and API key?

You only need **one** AI provider. Provider auto-detection checks in this order:

| Priority | Provider | Requirement |
|----------|----------|-------------|
| 1st | OpenAI | `OPENAI_API_KEY` env var set |
| 2nd | Anthropic | `ANTHROPIC_API_KEY` env var set |
| 3rd | Ollama | Local Ollama service running with `codellama` |

**Default models:**
- OpenAI: `gpt-4-turbo-preview`
- Anthropic: `claude-3-5-sonnet-20241022`
- Ollama: `codellama`

Set your key before running:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # or OPENAI_API_KEY
```

Override model with `--model claude-opus-4` if needed.

---

## Q4: What are the mandatory inputs for the strategy description?

**Must include:**
- **Entry condition** — which indicators and thresholds (e.g., "RSI(14) < 30", "price crosses above 50-day SMA")
- **Exit condition** — what triggers the close (RSI > 70, MA cross, take-profit trigger, etc.)
- **Direction** — "long only", "short only", or "both" (defaults to both if omitted)

**Optional but recommended:**
- Indicator default periods (e.g., "RSI period 14, EMA period 200")
- Day filter: "only trade Monday to Friday"
- Hour filter: "only enter trades between 9h and 17h"
- Stop loss / take profit if you want non-defaults (e.g., "1.5% stop loss, 3% take profit")

**Do NOT need to describe:**
- Position sizing formula — inherited automatically from `BaseStrategy`
- Import statements or code structure

---

## Q5: Is risk management predefined or do I describe it?

**Predefined and inherited** from `BaseStrategy`. You get this for free with every generated strategy:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `stop_loss_pct` | 2.0% | Stop loss distance from entry |
| `take_profit_pct` | 4.0% | Take profit distance from entry |
| `risk_pct` | 1.0% | Max account equity risked per trade |
| Position cap | 20% | Hard max position size (equity fraction) |

Position size is auto-calculated: `(equity × risk_pct%) / stop_loss_distance%`, capped at 20%.

All three (`stop_loss_pct`, `take_profit_pct`, `risk_pct`) are also **optimizable parameters**.

If you want different defaults, mention them in the description: "Use 1.5% stop loss, 3% take profit".

---

## Q6: Do I need to describe which parameters to optimize?

**No.** Any class-level attribute on the generated strategy is automatically an optimizable parameter.
The AI generates them as class variables:

```python
class RSIOversoldLong(BaseStrategy):
    rsi_period = 14       # <- optimizable
    rsi_oversold = 30     # <- optimizable
    ema_period = 200      # <- optimizable
    stop_loss_pct = 2.0   # <- inherited, also optimizable
```

When you run `bt.optimize()` or the GPU optimizer, you define the ranges to sweep for each.

---

## Q7: What is the optimization optimizing for?

| Script | Objective |
|--------|-----------|
| `examples/02_gpu_optimization.py` (DCA) | Maximizes `total_return_pct` by default |
| `examples/03_ema_crossover.py` | Returns all metrics; you choose the best trade-off |
| `backtest optimize` CLI command | Configurable: `--objective sharpe_ratio` (default) |

**Available metrics to evaluate results on:**

| Metric | Key | Interpretation |
|--------|-----|----------------|
| Total return % | `total_return_pct` | Raw profit |
| Sharpe ratio | `sharpe_ratio` | Return per unit of risk (> 1.0 = good) |
| Sortino ratio | `sortino_ratio` | Sharpe focused on downside risk |
| Profit factor | `profit_factor` | Gross wins / gross losses (> 1.3 = good) |
| Max drawdown % | `max_drawdown_pct` | Worst peak-to-trough loss (lower = better) |
| Win rate % | `win_rate_pct` | % of winning trades |
| SQN | `sqn` | System Quality Number (> 2.0 = good) |
| Expectancy % | `expectancy_pct` | Expected return per trade |

**Recommendation:** Target **Sharpe ratio** or **SQN** rather than raw return %,
to avoid cherry-picking lucky parameter combinations that won't generalize.

---

## Q8: Should I specify my optimization filters (day, symbol, timeframe, direction, hours)?

**Yes — these are your most impactful levers.**

| Filter | How to apply |
|--------|-------------|
| **Symbol** | Pass `--symbol BTCUSDT` at backtest time |
| **Timeframe** | Pass `--timeframe 1h` (or `4h`, `1d`, etc.) |
| **Direction** | Write "long only" or "short only" in your description |
| **Day filter** | Write "only trade Monday to Friday" in description → sets `trading_days = [0,1,2,3,4]` |
| **Hour filter** | Write "only enter between 9am and 5pm" in description → sets `trading_hours = range(9,17)` |

> Note: direction/day/hour filters apply to **entries only** — open positions stay until their exit condition regardless.

For robustness, test on multiple symbols after optimizing on one.

---

## Q9: Full recommended workflow

### Phase 1 — Generate the strategy
```bash
python -m src.cli.main strategy create \
  --description "Buy when RSI(14) < 30 and price > EMA(200). \
    Exit when RSI > 70. Long only. Only trade Monday to Friday." \
  --name RSIOversoldLong --provider auto --register
```
→ Saved to `src/strategies/generated/RSIOversoldLong.py`
→ Registered as `rsioversoldlong`

### Phase 2 — Initial backtest (in-sample, training period)
```bash
python -m src.cli.main backtest run \
  --strategy rsioversoldlong --symbol BTCUSDT \
  --timeframe 1h --start 2018-01-01 --end 2022-12-31 --cash 10000
```
Check: `num_trades > 30`, `sharpe_ratio > 1.0`, `profit_factor > 1.3`

### Phase 3 — Optimize parameters (GPU or backtesting.py)
```bash
# Using backtesting.py optimizer (multi-objective, targets Sharpe by default)
python -m src.cli.main backtest optimize \
  --strategy rsioversoldlong --symbol BTCUSDT \
  --timeframe 1h --start 2018-01-01 --end 2022-12-31 \
  --objective sharpe_ratio

# Or GPU DCA sweep (~1,145 tests/sec on RTX 4090)
python examples/02_gpu_optimization.py
```

### Phase 4 — Walk-forward validation (overfitting check)
```bash
python -m src.cli.main backtest walk-forward \
  --strategy rsioversoldlong --symbol BTCUSDT \
  --timeframe 1h --start 2018-01-01 --end 2025-12-31 \
  --train-ratio 0.7
```
Reports train metrics vs test metrics side by side.
- Metrics hold up → robust signal ✓
- Metrics collapse → overfit; simplify or widen parameter ranges

### Phase 5 — Multi-symbol robustness test
```bash
python -m src.cli.main backtest multi-symbol \
  --strategy rsioversoldlong \
  --symbols BTCUSDT ETHUSDT BNBUSDT \
  --timeframe 1h --start 2023-01-01 --end 2025-12-31
```
A real edge should work on 2+ symbols.

### Phase 6 — Review stored results
```bash
python -m src.cli.main results list-results --strategy rsioversoldlong --limit 10
```

---

## Q10: How do I detect overfitting?

Use the built-in walk-forward command (Phase 4 above), plus these manual checks:

| Check | How |
|-------|-----|
| **Walk-forward split** | `backtest walk-forward --train-ratio 0.7` — see if metrics hold on unseen 30% |
| **SQN threshold** | SQN < 1.6 on test period = likely overfit |
| **Buy & Hold comparison** | `buy_hold_return_pct` is in every result. If strategy barely beats B&H on test set, it adds no value |
| **Parameter sensitivity** | Run `python tools/param_sensitivity.py` — large result swings from small param changes = curve-fitted |
| **Multi-symbol test** | Robust edge appears on ETH and BNB too, not just BTC |

**Key thresholds after walk-forward:**

| Metric | Train | Test (acceptable drop) |
|--------|-------|----------------------|
| Sharpe ratio | > 1.0 | > 0.6 (40% drop OK) |
| Profit factor | > 1.3 | > 1.1 |
| Win rate | > 50% | > 45% |
| Max drawdown | < 30% | < 45% |

---

## Quick Reference: What do you need before starting?

| Item | Required? | Notes |
|------|-----------|-------|
| MCP server running | No | Only for AI assistants |
| API key (OpenAI or Anthropic) | One of them | Or free local Ollama |
| Ollama installed | Only if no API key | `ollama pull codellama` |
| Strategy description (entry/exit/direction) | Yes | Core requirement |
| Describe risk management | No | Predefined: 2% SL, 4% TP, 1% risk |
| Describe parameters to optimize | No | AI generates them automatically |
| Specify symbol/timeframe | Yes, at backtest time | Not in the description |
| Walk-forward validation | Recommended | `backtest walk-forward` command |
