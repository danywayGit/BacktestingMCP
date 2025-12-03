# AI-Powered Strategy Generation

Generate trading strategies from natural language descriptions using AI.

## Supported AI Providers

- **OpenAI** (GPT-4, GPT-3.5-turbo)
- **Anthropic** (Claude 3.5 Sonnet, Claude 3 Opus)
- **Ollama** (Local models like CodeLlama, Mistral)

## Setup

### 1. Install AI Provider Package

```bash
# For OpenAI
pip install openai

# For Anthropic Claude
pip install anthropic

# For Ollama (local)
pip install ollama
```

### 2. Set API Keys

```bash
# For OpenAI
export OPENAI_API_KEY="sk-..."

# For Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# For Ollama - no API key needed, just install and run:
# https://ollama.ai/
```

## Usage

### CLI Command

```bash
# Basic usage with auto-registration (recommended)
python -m src.cli.main strategy create \
  --description "Buy when RSI drops below 30 and price is above 50-day MA. Sell when RSI goes above 70." \
  --name "RSIOversoldStrategy" \
  --register

# Without auto-registration
python -m src.cli.main strategy create \
  --description "MACD crossover with volume confirmation" \
  --name "MACDVolumeStrategy" \
  --provider openai

# Use specific model with registration
python -m src.cli.main strategy create \
  --description "Bollinger Bands breakout with ATR stop loss" \
  --name "BBBreakoutStrategy" \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --register

# Custom output location
python -m src.cli.main strategy create \
  --description "Moving average crossover with momentum filter" \
  --name "MACrossStrategy" \
  --output my_strategies/ma_cross.py
```

### Python API

```python
from src.ai.strategy_generator import StrategyGenerator

# Initialize generator
generator = StrategyGenerator(provider="openai")  # or "anthropic", "ollama", "auto"

# Generate strategy
result = generator.generate_strategy(
    description="Buy when RSI < 30 and price > 50-day MA. Sell when RSI > 70.",
    strategy_name="RSIOversoldStrategy"
)

print(f"Provider: {result['provider']}")
print(f"Model: {result['model']}")
print(f"Code:\n{result['code']}")

# Validate code
is_valid, error = generator.validate_strategy_code(result['code'])
if not is_valid:
    print(f"Validation error: {error}")

# Save to file
filepath = generator.save_strategy(result['code'], "RSIOversoldStrategy")
print(f"Saved to: {filepath}")
```

### Interactive Mode

```bash
python -m src.ai.strategy_generator
```

This will guide you through:
1. Entering strategy description
2. Naming the strategy
3. Choosing AI provider
4. Generating and previewing code
5. Saving the strategy

### MCP Tool

If using the MCP server:

```json
{
  "tool": "create_strategy_from_description",
  "arguments": {
    "description": "Buy when RSI drops below 30 and price is above 50-day MA",
    "strategy_name": "RSIOversoldStrategy",
    "auto_register": true,
    "provider": "auto"
  }
}
```

The `auto_register` option (default: true) automatically adds the strategy to STRATEGY_REGISTRY.

## Example Descriptions

Good strategy descriptions are:
- **Specific**: Include exact indicators and thresholds
- **Complete**: Define both entry and exit conditions
- **Clear**: Use simple, unambiguous language

### Examples:

```
"Buy when RSI(14) drops below 30 and price is above 50-day moving average. 
Sell when RSI goes above 70 or price crosses below the 50-day MA."

"Enter long when fast MA(10) crosses above slow MA(30). 
Exit when fast MA crosses below slow MA."

"Buy when price touches the lower Bollinger Band (20-period, 2 std dev). 
Sell when price reaches the middle band."

"Enter when MACD line crosses above signal line and volume is 1.5x the 20-day average. 
Exit when MACD crosses below signal."

"Buy after 2 consecutive red candles when RSI < 35. 
Sell at 15% profit or 5% loss."
```

## Generated Code Structure

The AI generates a complete strategy class:

```python
class YourStrategy(BaseStrategy):
    """
    Strategy description...
    """
    
    # Parameters
    rsi_period = 14
    rsi_oversold = 30
    ma_period = 50
    
    def init(self):
        """Initialize indicators."""
        self.rsi = self.I(calculate_rsi, self.data.Close, self.rsi_period)
        self.ma = self.I(calculate_sma, self.data.Close, self.ma_period)
    
    def next(self):
        """Trading logic."""
        if not self.should_trade():
            return
        
        # Entry logic
        if not self.position:
            if self.rsi[-1] < self.rsi_oversold and self.data.Close[-1] > self.ma[-1]:
                self.enter_long_position()
        
        # Exit logic
        else:
            if self.rsi[-1] > 70:
                self.position.close()
```

## Post-Generation Steps

1. **Review the code** - AI-generated code should always be reviewed
2. **Test thoroughly** - Run backtests on multiple symbols and timeframes
3. **Validate logic** - Ensure the strategy matches your intent
4. **Register it** - Add to `STRATEGY_REGISTRY` in `src/strategies/templates.py`
5. **Use it** - Run via CLI or MCP

### Registering Generated Strategy

Edit `src/strategies/templates.py`:

```python
# Add import
from .generated.your_strategy import YourStrategy

# Add to registry
STRATEGY_REGISTRY = {
    # ... existing strategies ...
    'your_strategy': YourStrategy,
}
```

## Tips for Best Results

1. **Be specific** - "RSI below 30" is better than "oversold"
2. **Include numbers** - "50-day MA" is better than "long-term MA"
3. **Define both entry and exit** - Don't leave the AI guessing
4. **Mention indicators by name** - RSI, MACD, Bollinger Bands, etc.
5. **Keep it simple** - Start with basic strategies, then iterate

## Troubleshooting

### "Import could not be resolved" errors

These are normal - the AI packages (openai, anthropic, ollama) are optional dependencies. Install the one you want to use:

```bash
pip install openai  # or anthropic, or ollama
```

### API Key errors

Make sure your environment variable is set:

```bash
# Check if set
echo $OPENAI_API_KEY  # Linux/Mac
echo %OPENAI_API_KEY%  # Windows CMD
$env:OPENAI_API_KEY   # Windows PowerShell
```

### Validation failures

The generated code might have syntax errors. Review and fix manually, or:
1. Try a different AI provider
2. Rephrase your description
3. Use a more powerful model (e.g., GPT-4 instead of GPT-3.5)

### Ollama connection errors

Make sure Ollama is running:
```bash
ollama serve
ollama pull codellama  # Download model first
```

## Cost Considerations

- **OpenAI GPT-4**: ~$0.01-0.03 per strategy
- **OpenAI GPT-3.5**: ~$0.001-0.003 per strategy
- **Anthropic Claude**: ~$0.01-0.02 per strategy
- **Ollama**: Free (local)

For development, consider using Ollama for unlimited free generation.

## Limitations

- Generated code should always be reviewed by a human
- Complex strategies may not be perfectly captured
- AI might misinterpret ambiguous descriptions
- Some advanced features may require manual coding

## Future Enhancements

- [ ] Auto-registration in STRATEGY_REGISTRY
- [ ] Strategy optimization via AI
- [ ] Multi-iteration refinement
- [ ] Strategy explanation from code
- [ ] Backtesting result analysis

## License

MIT License - Use responsibly and always review AI-generated code before trading.
