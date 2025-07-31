"""
Quick installation and setup script for the backtesting system.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n📦 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.9+ is required")
        return False


def install_dependencies():
    """Install Python dependencies."""
    # Core dependencies that are most likely to work
    core_deps = [
        "pandas>=2.0.0",
        "numpy>=1.24.0", 
        "sqlalchemy>=2.0.0",
        "click>=8.0.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "rich>=13.0.0",
        "pyyaml>=6.0"
    ]
    
    # Optional dependencies that might need special handling
    optional_deps = [
        "ccxt>=4.0.0",
        "backtesting>=0.3.3", 
        "ta-lib>=0.4.25",
        "plotly>=5.15.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "streamlit>=1.25.0"
    ]
    
    print("📦 Installing core dependencies...")
    for dep in core_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️  Failed to install {dep}, continuing...")
    
    print("\n📦 Installing optional dependencies...")
    print("Note: Some of these may fail depending on your system configuration")
    
    for dep in optional_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️  Failed to install {dep}, this feature may not work")
    
    # Special handling for TA-Lib
    print("\n🔧 TA-Lib Installation Notes:")
    print("If TA-Lib installation failed, you may need to:")
    print("  Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
    print("  macOS: brew install ta-lib")
    print("  Linux: Install ta-lib development package first")


def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    directories = [
        "data",
        "logs", 
        "exports",
        "strategies/custom"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")


def create_env_file():
    """Create a sample .env file."""
    print("\n⚙️  Creating sample environment file...")
    
    env_content = """# Advanced Crypto Backtesting System Configuration

# Database
CRYPTO_DB_PATH=data/crypto.db

# Exchange settings
CRYPTO_EXCHANGE=binance

# Risk management
ACCOUNT_RISK_PCT=1.0
TRADING_COMMISSION=0.001

# MCP Server
MCP_PORT=8000

# Logging
LOG_LEVEL=INFO

# Optional: API keys for live data (leave empty for historical data only)
# BINANCE_API_KEY=
# BINANCE_SECRET_KEY=
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(env_content)
        print("✅ Created .env file")
    else:
        print("⚠️  .env file already exists, skipping")


def run_basic_tests():
    """Run basic functionality tests."""
    print("\n🧪 Running basic tests...")
    
    # Test 1: Import core modules
    try:
        sys.path.insert(0, "src")
        from config.settings import settings
        print("✅ Configuration loading works")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
    
    # Test 2: Database initialization
    try:
        from src.data.database import db
        print("✅ Database initialization works")
    except Exception as e:
        print(f"❌ Database test failed: {e}")
    
    # Test 3: Strategy templates
    try:
        from src.strategies.templates import list_available_strategies
        strategies = list_available_strategies()
        print(f"✅ Found {len(strategies)} strategy templates")
    except Exception as e:
        print(f"❌ Strategy templates test failed: {e}")
    
    # Test 4: Risk management
    try:
        from src.risk.risk_manager import RiskManager
        rm = RiskManager()
        print("✅ Risk management works")
    except Exception as e:
        print(f"❌ Risk management test failed: {e}")


def show_next_steps():
    """Show next steps for the user."""
    print("\n🎉 Setup Complete!")
    print("\n📋 Next Steps:")
    print("1. Download some sample data:")
    print("   python examples.py")
    print("   # or")
    print("   python -m src.cli.main data download --symbol BTC/USDT --timeframe 1h --start 2024-01-01 --end 2024-01-31")
    
    print("\n2. Run a sample backtest:")
    print("   python -m src.cli.main backtest run --strategy rsi_mean_reversion --symbol BTCUSDT --timeframe 1h --start 2024-01-01 --end 2024-01-31")
    
    print("\n3. Start the MCP server:")
    print("   python -m src.mcp.server")
    
    print("\n4. Explore the CLI:")
    print("   python -m src.cli.main --help")
    
    print("\n📚 Documentation:")
    print("   Check README.md for detailed usage instructions")
    print("   Look at examples.py for code examples")
    
    print("\n🐛 Troubleshooting:")
    print("   - If imports fail, ensure all dependencies are installed")
    print("   - If TA-Lib fails, install it separately (see notes above)")
    print("   - Check the .env file for configuration options")


def main():
    """Main setup function."""
    print("🚀 Advanced Crypto Backtesting System Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("❌ Please upgrade to Python 3.9 or later")
        return
    
    # Install dependencies
    install_dependencies()
    
    # Setup directories
    setup_directories()
    
    # Create environment file
    create_env_file()
    
    # Run basic tests
    run_basic_tests()
    
    # Show next steps
    show_next_steps()


if __name__ == "__main__":
    main()
