# Ollama Setup Guide for RTX 4090

Your RTX 4090 is perfect for running the best coding models! Here's the optimal setup:

## 🚀 Quick Start (Recommended)

### 1. Install Ollama
Download and install from: https://ollama.ai/

### 2. Start Ollama Server
```powershell
# In a PowerShell terminal (keep this running)
ollama serve
```

### 3. Pull Qwen3-Coder 30B (Best for RTX 4090)
```powershell
# In a NEW PowerShell terminal
ollama pull qwen3-coder:30b
```

This will download ~19GB, so it may take a few minutes.

### 4. Test It
```powershell
# In your project directory
python test_ollama_strategy.py
```

Choose option 1 (qwen3-coder:30b)

### 5. Generate Your First Strategy
```powershell
python -m src.cli.main strategy create `
  --description "Buy when RSI drops below 30 and price is above 50-day MA. Sell when RSI goes above 70." `
  --name "RSIOversoldStrategy" `
  --provider ollama `
  --model qwen3-coder:30b
```

## 📊 Performance Expectations (RTX 4090)

| Model | VRAM Usage | Speed | Quality |
|-------|-----------|-------|---------|
| qwen3-coder:30b | ~18GB | ~10-15 tokens/sec | ⭐⭐⭐⭐⭐ |
| codellama:34b | ~19GB | ~10-15 tokens/sec | ⭐⭐⭐⭐⭐ |
| deepseek-coder-v2:16b | ~10GB | ~20-30 tokens/sec | ⭐⭐⭐⭐ |

## 🎯 Which Model Should You Use?

### For Trading Strategies: **qwen3-coder:30b**
- Latest model (2024)
- Best at understanding complex logic
- Excellent code quality
- Your RTX 4090 handles it perfectly

### Alternative: **codellama:34b**
- More established, well-tested
- Slightly more conservative code
- Also excellent quality

### For Quick Iterations: **deepseek-coder-v2:16b**
- Faster generation
- Still very good quality
- Uses only ~10GB VRAM

## 🔧 Troubleshooting

### Ollama not found
```powershell
# Check if Ollama is installed
ollama --version

# If not found, download from https://ollama.ai/
```

### Model too slow
Your RTX 4090 should be fast. If it's slow:
1. Make sure GPU drivers are up to date
2. Close other GPU-intensive applications
3. Check GPU usage: `nvidia-smi`

### Out of memory
Even with 24GB VRAM, you might hit limits with 32B models if other apps are using GPU:
```powershell
# Try the 16B model instead
ollama pull deepseek-coder-v2:16b
```

### Connection errors
Make sure `ollama serve` is running in another terminal.

## 📝 Example Strategies to Generate

### Simple RSI Strategy
```powershell
python -m src.cli.main strategy create `
  --description "Buy when RSI(14) < 30. Sell when RSI > 70." `
  --name "SimpleRSIStrategy" `
  --provider ollama `
  --model qwen3-coder:30b
```

### MA Crossover
```powershell
python -m src.cli.main strategy create `
  --description "Buy when 10-day MA crosses above 30-day MA. Sell when it crosses below." `
  --name "MACrossStrategy" `
  --provider ollama `
  --model qwen3-coder:30b
```

### Bollinger Bands
```powershell
python -m src.cli.main strategy create `
  --description "Buy when price touches lower Bollinger Band (20, 2). Sell at middle band." `
  --name "BBMeanReversionStrategy" `
  --provider ollama `
  --model qwen3-coder:30b
```

### Complex DCA Strategy
```powershell
python -m src.cli.main strategy create `
  --description "DCA strategy: Buy with $500 when RSI < 35 and price is 10% below 200-day EMA. Sell 20% when price is 15% above 200-day EMA and RSI > 65. Keep minimum 50% of holdings." `
  --name "SmartDCAStrategy" `
  --provider ollama `
  --model qwen3-coder:30b
```

## 🎓 Tips for Better Strategies

1. **Be Specific**: Include exact numbers (RSI < 30, not "oversold")
2. **Define Both Entry and Exit**: Don't leave the AI guessing
3. **Mention Periods**: "14-period RSI" not just "RSI"
4. **Include Risk Management**: Stop loss, position sizing
5. **Test First**: Always backtest generated strategies

## 🔗 Next Steps

After generating a strategy:
1. Review the code in `src/strategies/generated/`
2. Test it with backtesting
3. Register it in `src/strategies/templates.py`
4. Run real backtests on multiple symbols

## 🆘 Need Help?

- Ollama docs: https://ollama.ai/docs
- Model library: https://ollama.ai/library
- Qwen2.5-Coder: https://ollama.ai/library/qwen2.5-coder
- CodeLlama: https://ollama.ai/library/codellama

Enjoy your RTX 4090 power! 🚀
