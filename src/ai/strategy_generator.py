"""
AI-powered strategy generation from natural language descriptions.

Supports multiple AI providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Local models via Ollama
"""

import os
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


STRATEGY_GENERATION_PROMPT = """You are an expert trading strategy developer. Generate a complete Python trading strategy class based on the user's description.

The strategy MUST:
1. Inherit from BaseStrategy
2. Have an init() method to initialize indicators
3. Have a next() method with trading logic
4. Use self.I() to wrap indicator calculations
5. Call self.should_trade() before trading
6. Use self.enter_long_position() and self.enter_short_position() for entries
7. Use self.position.close() for exits
8. Set appropriate class-level parameters

Available indicator functions you can use:
- calculate_rsi(close_prices, period=14)
- calculate_sma(close_prices, period)
- calculate_ema(close_prices, period)
- calculate_bbands(close_prices, period=20, std=2)
- calculate_macd(close_prices, fast=12, slow=26, signal=9)

Available data fields:
- self.data.Open, self.data.High, self.data.Low, self.data.Close, self.data.Volume

User Description: {description}

Generate ONLY the Python class code with no markdown formatting, no explanations. Just the raw Python code starting with 'class' and ending with the last method."""


class StrategyGenerator:
    """Generates trading strategies from natural language descriptions using AI."""
    
    def __init__(self, provider: str = "auto"):
        """
        Initialize strategy generator.
        
        Args:
            provider: AI provider to use ('openai', 'anthropic', 'ollama', or 'auto')
        """
        self.provider = provider
        self._setup_provider()
    
    def _setup_provider(self):
        """Setup AI provider based on available credentials."""
        if self.provider == "auto":
            # Auto-detect based on environment variables
            if os.getenv("OPENAI_API_KEY"):
                self.provider = "openai"
            elif os.getenv("ANTHROPIC_API_KEY"):
                self.provider = "anthropic"
            else:
                self.provider = "ollama"  # Fallback to local
        
        logger.info(f"Using AI provider: {self.provider}")
    
    def generate_strategy(
        self,
        description: str,
        strategy_name: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a trading strategy from natural language description.
        
        Args:
            description: Natural language description of the strategy
            strategy_name: Name for the generated strategy
            model: Specific model to use (optional)
        
        Returns:
            Dictionary with 'code', 'class_name', and 'explanation'
        """
        try:
            if self.provider == "openai":
                return self._generate_openai(description, strategy_name, model)
            elif self.provider == "anthropic":
                return self._generate_anthropic(description, strategy_name, model)
            elif self.provider == "ollama":
                return self._generate_ollama(description, strategy_name, model)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        
        except Exception as e:
            logger.error(f"Strategy generation failed: {e}")
            raise
    
    def _generate_openai(
        self,
        description: str,
        strategy_name: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate strategy using OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        client = openai.OpenAI(api_key=api_key)
        model = model or "gpt-4-turbo-preview"
        
        prompt = STRATEGY_GENERATION_PROMPT.format(description=description)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert Python developer specializing in trading algorithms."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        code = response.choices[0].message.content.strip()
        code = self._clean_code(code)
        code = self._ensure_class_name(code, strategy_name)
        
        return {
            "code": code,
            "class_name": strategy_name,
            "explanation": f"Generated using OpenAI {model}",
            "provider": "openai",
            "model": model
        }
    
    def _generate_anthropic(
        self,
        description: str,
        strategy_name: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate strategy using Anthropic Claude API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        client = anthropic.Anthropic(api_key=api_key)
        model = model or "claude-3-5-sonnet-20241022"
        
        prompt = STRATEGY_GENERATION_PROMPT.format(description=description)
        
        message = client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        code = message.content[0].text.strip()
        code = self._clean_code(code)
        code = self._ensure_class_name(code, strategy_name)
        
        return {
            "code": code,
            "class_name": strategy_name,
            "explanation": f"Generated using Anthropic {model}",
            "provider": "anthropic",
            "model": model
        }
    
    def _generate_ollama(
        self,
        description: str,
        strategy_name: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate strategy using local Ollama."""
        try:
            import ollama
        except ImportError:
            raise ImportError("Ollama package not installed. Run: pip install ollama")
        
        model = model or "codellama"
        
        prompt = STRATEGY_GENERATION_PROMPT.format(description=description)
        
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert Python developer specializing in trading algorithms."},
                {"role": "user", "content": prompt}
            ]
        )
        
        code = response['message']['content'].strip()
        code = self._clean_code(code)
        code = self._ensure_class_name(code, strategy_name)
        
        return {
            "code": code,
            "class_name": strategy_name,
            "explanation": f"Generated using Ollama {model}",
            "provider": "ollama",
            "model": model
        }
    
    def _clean_code(self, code: str) -> str:
        """Clean generated code by removing markdown formatting."""
        # Remove markdown code blocks
        code = re.sub(r'^```python\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
        
        # Remove any explanatory text before the class
        lines = code.split('\n')
        class_line_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith('class '):
                class_line_idx = i
                break
        
        if class_line_idx is not None:
            code = '\n'.join(lines[class_line_idx:])
        
        return code.strip()
    
    def _ensure_class_name(self, code: str, desired_name: str) -> str:
        """Ensure the class has the desired name."""
        # Find the class definition line
        match = re.search(r'class\s+\w+\s*\(', code)
        if match:
            # Replace with desired name
            code = re.sub(r'class\s+\w+\s*\(', f'class {desired_name}(', code, count=1)
        
        return code
    
    def validate_strategy_code(self, code: str) -> tuple[bool, str]:
        """
        Validate that the generated code is syntactically correct.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            compile(code, '<string>', 'exec')
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)
    
    def save_strategy(
        self,
        code: str,
        strategy_name: str,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Save generated strategy to a file.
        
        Args:
            code: Strategy code
            strategy_name: Name of the strategy
            output_dir: Directory to save to (default: src/strategies/generated/)
        
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "strategies" / "generated"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert strategy name to filename
        filename = strategy_name.lower().replace(' ', '_')
        if not filename.endswith('.py'):
            filename += '.py'
        
        filepath = output_dir / filename
        
        # Add imports at the top
        full_code = self._add_imports(code)
        
        with open(filepath, 'w') as f:
            f.write(full_code)
        
        logger.info(f"Strategy saved to: {filepath}")
        return filepath
    
    def _add_imports(self, code: str) -> str:
        """Add necessary imports to the strategy code."""
        imports = '''"""
AI-generated trading strategy.
"""

import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.lib import crossover

from ..core.backtesting_engine import BaseStrategy
from ..strategies.templates import (
    calculate_rsi,
    calculate_sma,
    calculate_bbands,
    calculate_macd
)


'''
        return imports + code


def generate_strategy_interactive():
    """Interactive CLI for strategy generation."""
    print("\n" + "=" * 70)
    print("AI-POWERED STRATEGY GENERATOR")
    print("=" * 70)
    
    # Get description
    print("\nDescribe your trading strategy in plain English:")
    print("Example: 'Buy when RSI drops below 30 and price is above 50-day MA'")
    description = input("\nYour strategy: ").strip()
    
    if not description:
        print("❌ No description provided")
        return
    
    # Get strategy name
    print("\nEnter a name for your strategy (e.g., 'RSIOversoldStrategy'):")
    strategy_name = input("Strategy name: ").strip()
    
    if not strategy_name:
        print("❌ No strategy name provided")
        return
    
    # Choose provider
    print("\nSelect AI provider:")
    print("  1. OpenAI (requires OPENAI_API_KEY)")
    print("  2. Anthropic Claude (requires ANTHROPIC_API_KEY)")
    print("  3. Ollama (local, requires Ollama running)")
    print("  4. Auto-detect")
    
    choice = input("\nChoice (1-4, default: 4): ").strip() or "4"
    
    provider_map = {"1": "openai", "2": "anthropic", "3": "ollama", "4": "auto"}
    provider = provider_map.get(choice, "auto")
    
    # Generate
    print(f"\n🤖 Generating strategy using {provider}...")
    
    try:
        generator = StrategyGenerator(provider=provider)
        result = generator.generate_strategy(description, strategy_name)
        
        print("\n✅ Strategy generated successfully!")
        print(f"Provider: {result['provider']}")
        print(f"Model: {result['model']}")
        
        # Validate
        is_valid, error = generator.validate_strategy_code(result['code'])
        if not is_valid:
            print(f"\n⚠️  Warning: Code validation failed: {error}")
        
        # Show preview
        print("\n" + "-" * 70)
        print("GENERATED CODE PREVIEW:")
        print("-" * 70)
        lines = result['code'].split('\n')
        for i, line in enumerate(lines[:30], 1):  # Show first 30 lines
            print(f"{i:3d}: {line}")
        
        if len(lines) > 30:
            print(f"... ({len(lines) - 30} more lines)")
        
        # Save
        save = input("\n💾 Save this strategy? (y/n): ").strip().lower()
        if save == 'y':
            filepath = generator.save_strategy(result['code'], strategy_name)
            print(f"\n✅ Strategy saved to: {filepath}")
            print("\nNext steps:")
            print(f"  1. Review and test the code in {filepath}")
            print(f"  2. Register it in src/strategies/templates.py STRATEGY_REGISTRY")
            print(f"  3. Run: python -m src.cli.main strategy list-strategies")
        else:
            print("\n❌ Strategy not saved")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Strategy generation failed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_strategy_interactive()
