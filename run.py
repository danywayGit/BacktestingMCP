#!/usr/bin/env python3
"""
Launcher script for the Advanced Crypto Backtesting System.
This script ensures the virtual environment is activated before running the main application.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def is_venv_activated():
    """Check if virtual environment is activated."""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)


def venv_exists():
    """Check if virtual environment exists."""
    venv_path = Path("venv")
    if platform.system() == "Windows":
        return (venv_path / "Scripts" / "python.exe").exists()
    else:
        return (venv_path / "bin" / "python").exists()


def get_venv_python():
    """Get the virtual environment Python executable."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\python.exe"
    else:
        return "venv/bin/python"


def get_activation_command():
    """Get the virtual environment activation command."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"


def main():
    """Main launcher function."""
    script_name = sys.argv[0]
    print(f"🚀 Advanced Crypto Backtesting System Launcher")
    print("=" * 60)
    
    # Check if we're already in a virtual environment
    if is_venv_activated():
        print("✅ Virtual environment is activated")
        
        # Import and run the examples
        try:
            import examples
            examples.main()
        except ImportError as e:
            print(f"❌ Error importing examples: {e}")
            print("Make sure you're in the correct directory")
        except Exception as e:
            print(f"❌ Error running examples: {e}")
        return
    
    # Check if virtual environment exists
    if not venv_exists():
        print("❌ Virtual environment not found")
        print("\n📋 To set up the virtual environment:")
        print("1. Run: python setup_venv.py")
        print("2. Then run this script again")
        return
    
    # Virtual environment exists but not activated
    print("⚠️  Virtual environment found but not activated")
    print(f"\n📋 To activate manually:")
    activation_cmd = get_activation_command()
    print(f"   {activation_cmd}")
    
    if platform.system() == "Windows":
        print("   or simply run: activate.bat")
    else:
        print("   or simply run: ./activate.sh")
    
    # Ask if user wants to run in virtual environment
    response = input("\nDo you want to run in the virtual environment now? (Y/n): ").lower().strip()
    
    if response in ['', 'y', 'yes']:
        print("\n🔄 Running in virtual environment...")
        
        # Get virtual environment Python
        venv_python = get_venv_python()
        
        # Re-run this script with virtual environment Python
        try:
            # Run examples.py directly with venv Python
            result = subprocess.run([venv_python, "examples.py"], check=True)
            print("✅ Completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running in virtual environment: {e}")
        except FileNotFoundError:
            print(f"❌ Virtual environment Python not found: {venv_python}")
    else:
        print("👋 Exiting. Activate the virtual environment and try again.")


if __name__ == "__main__":
    main()
