"""
Demo: AI-Powered Strategy Generation

This script demonstrates how to generate trading strategies from natural language.
"""

from src.ai.strategy_generator import StrategyGenerator
import os


def demo_openai():
    """Demo using OpenAI."""
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping OpenAI demo - OPENAI_API_KEY not set")
        return
    
    print("\n" + "=" * 70)
    print("DEMO: OpenAI Strategy Generation")
    print("=" * 70)
    
    generator = StrategyGenerator(provider="openai")
    
    description = """
    Buy when RSI(14) drops below 30 and the price is above the 50-day moving average.
    Sell when RSI goes above 70 or price drops below the 50-day MA.
    """
    
    print(f"\nDescription: {description.strip()}")
    print("\n🤖 Generating strategy...")
    
    result = generator.generate_strategy(
        description=description,
        strategy_name="RSIOversoldStrategy"
    )
    
    print(f"\n✅ Generated using {result['provider']} - {result['model']}")
    print("\nCode preview (first 20 lines):")
    print("-" * 70)
    
    lines = result['code'].split('\n')
    for i, line in enumerate(lines[:20], 1):
        print(f"{i:3d}: {line}")
    
    if len(lines) > 20:
        print(f"... ({len(lines) - 20} more lines)")


def demo_anthropic():
    """Demo using Anthropic Claude."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Skipping Anthropic demo - ANTHROPIC_API_KEY not set")
        return
    
    print("\n" + "=" * 70)
    print("DEMO: Anthropic Claude Strategy Generation")
    print("=" * 70)
    
    generator = StrategyGenerator(provider="anthropic")
    
    description = """
    Enter long when the fast MA (10-period) crosses above the slow MA (30-period).
    Exit when the fast MA crosses back below the slow MA.
    """
    
    print(f"\nDescription: {description.strip()}")
    print("\n🤖 Generating strategy...")
    
    result = generator.generate_strategy(
        description=description,
        strategy_name="MACrossoverStrategy"
    )
    
    print(f"\n✅ Generated using {result['provider']} - {result['model']}")
    print("\nCode preview (first 20 lines):")
    print("-" * 70)
    
    lines = result['code'].split('\n')
    for i, line in enumerate(lines[:20], 1):
        print(f"{i:3d}: {line}")
    
    if len(lines) > 20:
        print(f"... ({len(lines) - 20} more lines)")


def demo_ollama():
    """Demo using Ollama (local)."""
    print("\n" + "=" * 70)
    print("DEMO: Ollama (Local) Strategy Generation")
    print("=" * 70)
    print("\n⚠️  Note: Requires Ollama to be installed and running")
    print("Install: https://ollama.ai/")
    print("Run: ollama serve && ollama pull codellama")
    
    try:
        generator = StrategyGenerator(provider="ollama")
        
        description = """
        Buy when price touches the lower Bollinger Band (20-period, 2 std dev).
        Sell when price reaches the middle Bollinger Band.
        """
        
        print(f"\nDescription: {description.strip()}")
        print("\n🤖 Generating strategy...")
        
        result = generator.generate_strategy(
            description=description,
            strategy_name="BBMeanReversionStrategy"
        )
        
        print(f"\n✅ Generated using {result['provider']} - {result['model']}")
        print("\nCode preview (first 20 lines):")
        print("-" * 70)
        
        lines = result['code'].split('\n')
        for i, line in enumerate(lines[:20], 1):
            print(f"{i:3d}: {line}")
        
        if len(lines) > 20:
            print(f"... ({len(lines) - 20} more lines)")
    
    except Exception as e:
        print(f"\n❌ Ollama demo failed: {e}")
        print("Make sure Ollama is installed and running.")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("AI-POWERED STRATEGY GENERATION DEMO")
    print("=" * 70)
    print("\nThis demo shows how to generate trading strategies from plain English.")
    print("\nSetup:")
    print("  1. Install AI package: pip install openai (or anthropic, or ollama)")
    print("  2. Set API key: export OPENAI_API_KEY='sk-...'")
    print("  3. Run this script")
    
    # Try each provider
    demo_openai()
    demo_anthropic()
    demo_ollama()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review generated strategies")
    print("  2. Test with: python -m src.cli.main strategy create --help")
    print("  3. See full docs: src/ai/README.md")


if __name__ == "__main__":
    main()
