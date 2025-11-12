# Strategy Parameter Format - Why Class Attributes?

## Question
Should strategies use `params` tuple or class attributes for parameters?

## Answer: Class Attributes! ✅

The AI-generated strategy now uses **class attributes** for parameters, not the `params` tuple format.

## Why Class Attributes?

### 1. **Optimizer Compatible** 🎯
```python
# Class attributes work with backtesting.py optimizer
stats = bt.optimize(
    rsi_period=[10, 14, 20],
    ema_period=[100, 200, 300],
    maximize='Sharpe Ratio'
)
```

### 2. **Auto-Detection** 🔍
```python
# System automatically extracts parameters
params = get_strategy_parameters('test_rsi_strategy')
# Returns: {'rsi_period': 14, 'ema_period': 200, ...}
```

### 3. **Easy to Override** 🔧
```python
# Can override when running backtest
bt = Backtest(df, TestRSIStrategy, cash=10000)
stats = bt.run(rsi_period=20, ema_period=100)
```

### 4. **More Pythonic** 📝
```python
# Clean and readable
class TestRSIStrategy(BaseStrategy):
    # Strategy parameters
    rsi_period = 14
    ema_period = 200
    rsi_oversold = 30
    rsi_overbought = 70
```

## Format Comparison

### ❌ Tuple Format (Don't Use)
```python
class MyStrategy(BaseStrategy):
    params = (
        ('rsi_period', 14),
        ('ema_period', 200),
    )
    
    def init(self):
        # Access with self.params.rsi_period
        self.rsi = self.I(calculate_rsi, self.data.Close, self.params.rsi_period)
```

**Problems:**
- More verbose
- Harder to read
- Still works but less common in modern code

### ✅ Class Attributes (Use This)
```python
class MyStrategy(BaseStrategy):
    # Strategy parameters
    rsi_period = 14
    ema_period = 200
    
    def init(self):
        # Access directly with self.rsi_period
        self.rsi = self.I(calculate_rsi, self.data.Close, self.rsi_period)
```

**Benefits:**
- Clean and concise
- Direct attribute access
- Works with optimizer
- Auto-detected by system

## Example: Optimization

See `optimize_example.py` for a complete example of how to optimize strategy parameters.

```bash
python optimize_example.py
```

This will:
1. Load BTC data
2. Test different parameter combinations
3. Find the best parameters for maximum Sharpe Ratio
4. Generate an HTML report with heatmaps

## Key Takeaway

✅ **Use class attributes for strategy parameters**
- They work with optimization
- They're automatically detected
- They're easier to read and modify
- The AI generator now produces this format by default

🔄 **AI Generator Updated**
The prompt for the AI strategy generator has been updated to explicitly instruct it to use class attributes, not the params tuple format.
