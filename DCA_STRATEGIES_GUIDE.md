

# DCA Strategy Backtesting Guide

This guide explains how to use the two DCA (Dollar Cost Averaging) strategies implemented in this system and compare their performance.

## 🎯 Overview

Two distinct DCA strategies have been implemented for cryptocurrency trading:

### **Strategy 1: Monthly DCA with Rebalancing**
- Regular scheduled buying with fixed monthly budget
- Smart rebalancing of unused capital
- Systematic approach with clear rules
- Good for disciplined, hands-off investing

### **Strategy 2: Signal-Based DCA**
- Opportunistic buying based on technical signals
- Dynamic position sizing
- More sophisticated entry/exit rules
- Good for active, data-driven investing

---

## 📋 Strategy Specifications

### **Strategy 1: Monthly DCA**

**Budget Allocation:**
- $600/month split across coins:
  - BTC: 50% ($300)
  - ETH: 25% ($150)
  - BNB: 15% ($90)
  - TRX: 10% ($60)
- Each coin's budget divided into 5 equal parts

**Buy Rules:**
- **Standard Buy**: After 2 consecutive red days → buy 1/5 of allocation
- **Oversold Buy**: Price down ≥10% from reference → buy 2/5 of allocation
- **Minimum**: Must execute at least 2 buys per month per coin

**Sell Rules:**
- Trigger when:
  - Price is ≥15% above 200-day EMA, AND
  - Profit is ≥10% from average cost
- Sell 15% of holdings
- Maximum 1 sell per week
- Always keep at least 20% of coins
- After selling: stop buying until price returns to 200-day EMA

**Monthly Rebalancing:**
When you have unused capital + new funds:
1. Total available = Unused budget + $600 + extra funds
2. Split 50/50:
   - 50% follows original ratio (50/25/15/10)
   - 50% goes proportionally to underused coins
3. Recalculate new 1/5 buy amounts

**Philosophy**: Patient accumulation + aggressive oversold buying + disciplined profit-taking + smart capital recycling

---

### **Strategy 2: Signal-Based DCA**

**Cash Management:**
- $600/month into single accumulated pool
- Split: 70% Active | 30% Reserve
- No scheduled buys - only on signals
- Allocation ratios: BTC 50% | ETH 25% | BNB 15% | TRX 10%

**Position Sizing (Dynamic):**
Base size depends on accumulated cash:
- $600-$1,200: 1/10 of coin allocation
- $1,200-$2,400: 1/15
- $2,400-$3,600: 1/20
- $3,600+: 1/25

Size bonuses:
- Price dropped 10-19% from last buy: +15%
- Price dropped 20%+ from last buy: +25%
- Signal score 8-9: ×1.3
- Signal score 10+: ×1.5

**Buy Signal System (6+ points required):**

Point allocation (max 10):
1. **Distance from 200 EMA** (1-3 pts):
   - -8% to -12%: 1 pt
   - -12% to -18%: 2 pts
   - <-18%: 3 pts

2. **RSI(14)** (1-2 pts):
   - 35-45: 1 pt
   - <35: 2 pts

3. **V-shape pattern** (2 pts):
   - Price in lowest 25% of last 20 readings
   - Starting to recover

4. **Price acceleration** (1 pt):
   - Worsened ≥3% in 2 days

5. **Volume spike** (1 pt):
   - Volume >150% of 20-day average

6. **Price vs last buy** (1 pt):
   - Current price ≥10% below last buy

Required conditions:
- Price must be below 200 EMA
- Either: 2 red days OR 1 red day with ≥5% drop

Signal interpretation:
- 6-7 points: Standard buy
- 8-9 points: Strong buy (1.3× size)
- 10+ points: Extreme buy (1.5× size, uses reserve pool)

**Sell Signal (all conditions required):**

Triggers:
1. **Distance from 200 EMA**:
   - Moderate overbought: +15-25%
   - Strong overbought: +25-40%
   - Extreme overbought: >40%

2. **Inverse V-shape**: Price in top 25% and declining

3. **RSI(14)** > 65

4. **Profit** ≥15% from average cost

5. **Holding period** ≥30 days

Sell amounts:
- Moderate: 10% of holdings (max every 2 weeks)
- Strong: 20% of holdings (max weekly)
- Extreme: 30% of holdings (max weekly)

Limits:
- Never sell >30% at once
- Always keep ≥20% of all coins
- After selling: stop buying until price within 5% of 200 EMA

**Philosophy**: Wait for exceptional opportunities + scale position size with signal strength + take profits at overbought extremes

---

## 🚀 Quick Start

### **1. Download Historical Data**

```bash
# Download data for all 4 cryptocurrencies
python download_dca_data.py
```

This downloads daily data for:
- BTCUSDT (Bitcoin)
- ETHUSDT (Ethereum)
- BNBUSDT (Binance Coin)
- TRXUSDT (Tron)

### **2. Run Strategy Comparison**

```bash
# Compare both strategies across all assets
python compare_dca_strategies.py
```

This will:
1. Load historical data for all symbols
2. Run both DCA strategies
3. Generate comprehensive comparison report
4. Save results to `backtest_results/dca_comparison_[timestamp].json`

### **3. Review Results**

The comparison report includes:
- Overall portfolio performance for each strategy
- Per-symbol breakdown
- Trading activity analysis
- Risk metrics (drawdowns)
- Key insights and winner determination

---

## 📊 Understanding the Output

### **Portfolio Performance Metrics**

```
Total Start Value:  $10,000.00
Total End Value:    $12,500.00
Total Return:       25.00%
Total Trades:       45
Avg Win Rate:       65.50%
Max Drawdown:       -8.50%
```

- **Total Return**: Overall percentage gain/loss
- **Total Trades**: Number of buy+sell transactions
- **Avg Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Largest peak-to-trough decline

### **Per-Symbol Performance**

Shows how each cryptocurrency performed under each strategy:
- Individual returns
- Number of trades
- Win rates

### **Key Insights**

Automatically generated insights showing:
- Which strategy had higher returns
- Trading frequency comparison
- Win rate comparison
- Risk comparison (drawdown)
- Capital efficiency (return per trade)

---

## 🛠️ Customization

### **Modify Time Period**

Edit `compare_dca_strategies.py`:

```python
comparison = DCAStrategyComparison(
    symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'TRXUSDT'],
    timeframe='1d',
    start_date='2023-01-01',  # Change this
    end_date='2024-12-31',     # Change this
    initial_cash=10000.0,
    monthly_contribution=600.0
)
```

### **Adjust Strategy Parameters**

**For Monthly DCA Strategy:**

In `src/strategies/dca_strategies.py`, modify `DCAMonthlyStrategy` class:

```python
class DCAMonthlyStrategy(BaseStrategy):
    monthly_budget = 600.0  # Change monthly budget
    red_days_trigger = 2    # Change red day requirement
    oversold_threshold = -10.0  # Change oversold %
    sell_percentage = 15.0  # Change sell %
    # ... etc
```

**For Signal-Based DCA:**

```python
class DCASignalStrategy(BaseStrategy):
    monthly_contribution = 600.0  # Change monthly amount
    min_signal_score = 6  # Change minimum signal score
    strong_signal_score = 8  # Change strong signal threshold
    # ... etc
```

### **Change Asset Allocation**

```python
self.allocation_ratios = {
    'BTCUSDT': 0.50,  # 50% to Bitcoin
    'ETHUSDT': 0.25,  # 25% to Ethereum
    'BNBUSDT': 0.15,  # 15% to BNB
    'TRXUSDT': 0.10   # 10% to Tron
}
```

---

## 📈 Advanced Usage

### **Run Single Strategy Only**

```python
from backtesting import Backtest
from src.strategies.dca_strategies import DCAMonthlyStrategy
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')  # Must have OHLCV columns

# Run backtest
bt = Backtest(data, DCAMonthlyStrategy, cash=10000, commission=0.001)
stats = bt.run(monthly_budget=600)

print(stats)
```

### **Optimize Strategy Parameters**

```python
# Optimize parameters
stats, heatmap = bt.optimize(
    monthly_budget=range(400, 800, 100),
    oversold_threshold=[-15, -10, -5],
    maximize='Return [%]',
    return_heatmap=True
)
```

### **Access Detailed Trade Data**

```python
# Get all trades
trades = stats._trades
print(trades)

# Filter winning trades
winning_trades = trades[trades['PnL'] > 0]
print(f"Winning trades: {len(winning_trades)}")

# Get equity curve
equity_curve = stats._equity_curve
print(equity_curve)
```

---

## 📝 Strategy Comparison Checklist

When comparing the strategies, consider:

✅ **Returns**: Which strategy made more money?
✅ **Risk**: Which had smaller drawdowns?
✅ **Consistency**: Which had better win rate?
✅ **Efficiency**: Which made more per trade?
✅ **Trading Frequency**: Which suits your style?
✅ **Market Conditions**: Which performed better in different conditions?

---

## 🔍 Interpreting Results

### **When Monthly DCA Performs Better:**
- Steady, consistent market trends
- Volatile but predictable cycles
- When discipline beats timing
- Lower transaction costs preferred

### **When Signal-Based DCA Performs Better:**
- Highly volatile markets with clear extremes
- Strong trend reversals
- When patience creates exceptional opportunities
- When quality beats quantity

---

## 💡 Tips for Best Results

1. **Test Multiple Time Periods**: Bull markets vs bear markets vs sideways
2. **Consider Transaction Costs**: More trades = more fees
3. **Match to Your Style**: Systematic vs opportunistic
4. **Monitor Performance**: Track live vs backtest differences
5. **Adjust Parameters**: Based on your risk tolerance

---

## 🐛 Troubleshooting

### **No Data Found Error**
```bash
# Re-download data
python download_dca_data.py
```

### **Import Errors**
```bash
# Ensure you're in project root
cd BacktestingMCP

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### **Memory Issues with Large Datasets**
- Use shorter time periods
- Reduce number of symbols
- Use higher timeframes (1d instead of 1h)

---

## 📚 Further Reading

- **Backtesting.py Documentation**: https://kernc.github.io/backtesting.py/
- **Technical Analysis Library**: https://technical-analysis-library-in-python.readthedocs.io/
- **DCA Strategy Research**: Search for academic papers on dollar-cost averaging

---

## 🤝 Contributing

To add new DCA variations:

1. Create new strategy class inheriting from `BaseStrategy`
2. Implement `init()` and `next()` methods
3. Add to `DCA_STRATEGY_REGISTRY` in `dca_strategies.py`
4. Update comparison script to include new strategy

---

## 📄 License

MIT License - feel free to use and modify these strategies for your own trading.

---

**⚠️ Disclaimer**: These strategies are for educational and backtesting purposes only. Past performance does not guarantee future results. Always do your own research and never invest more than you can afford to lose.
