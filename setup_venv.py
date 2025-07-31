#!/usr/bin/env python3
"""
Virtual environment setup script for the Advanced Crypto Backtesting System.
This script creates and configures a Python virtual environment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"\n📦 {description}")
    print(f"Running: {command}")
    
    try:
        if platform.system() == "Windows":
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(command.split(), check=check, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print(result.stdout.strip())
        else:
            print(f"⚠️  Warning: Command returned {result.returncode}")
            if result.stderr:
                print(f"stderr: {result.stderr.strip()}")
        
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print("✅ Python version is compatible (3.9+)")
        return True
    else:
        print("❌ Python 3.9+ is required")
        return False


def create_virtual_environment():
    """Create a Python virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print(f"⚠️  Virtual environment already exists at {venv_path}")
        response = input("Do you want to recreate it? (y/N): ").lower().strip()
        if response == 'y':
            print("🗑️  Removing existing virtual environment...")
            import shutil
            shutil.rmtree(venv_path)
        else:
            print("📁 Using existing virtual environment")
            return True
    
    print("🔨 Creating virtual environment...")
    success = run_command("python -m venv venv", "Creating virtual environment")
    
    if success:
        print(f"✅ Virtual environment created at {venv_path.absolute()}")
        return True
    else:
        print("❌ Failed to create virtual environment")
        return False


def get_activation_command():
    """Get the virtual environment activation command."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"


def get_python_executable():
    """Get the Python executable path in the virtual environment."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\python.exe"
    else:
        return "venv/bin/python"


def get_pip_executable():
    """Get the pip executable path in the virtual environment."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\pip.exe"
    else:
        return "venv/bin/pip"


def upgrade_pip():
    """Upgrade pip in the virtual environment."""
    pip_exe = get_pip_executable()
    return run_command(f"{pip_exe} install --upgrade pip", "Upgrading pip")


def install_dependencies():
    """Install Python dependencies in the virtual environment."""
    pip_exe = get_pip_executable()
    
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
    
    print("📦 Installing core dependencies...")
    for dep in core_deps:
        run_command(f"{pip_exe} install {dep}", f"Installing {dep}", check=False)
    
    # Try to install from requirements.txt if it exists
    if Path("requirements.txt").exists():
        print("\n📦 Installing from requirements.txt...")
        run_command(f"{pip_exe} install -r requirements.txt", "Installing from requirements.txt", check=False)
    
    # Optional dependencies
    optional_deps = [
        "ccxt>=4.0.0",
        "backtesting>=0.3.3", 
        "plotly>=5.15.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0"
    ]
    
    print("\n📦 Installing optional dependencies...")
    print("Note: Some of these may fail depending on your system configuration")
    
    for dep in optional_deps:
        run_command(f"{pip_exe} install {dep}", f"Installing {dep}", check=False)


def create_activation_scripts():
    """Create convenient activation scripts."""
    activation_cmd = get_activation_command()
    
    # Windows batch file
    if platform.system() == "Windows":
        with open("activate.bat", "w") as f:
            f.write(f"""@echo off
echo Activating virtual environment...
call {activation_cmd}
echo Virtual environment activated!
echo.
echo To deactivate, run: deactivate
echo.
cmd /k
""")
        print("✅ Created activate.bat for Windows")
    
    # Unix shell script
    with open("activate.sh", "w") as f:
        f.write(f"""#!/bin/bash
echo "Activating virtual environment..."
{activation_cmd}
echo "Virtual environment activated!"
echo ""
echo "To deactivate, run: deactivate"
echo ""
bash
""")
    
    # Make shell script executable on Unix
    if platform.system() != "Windows":
        os.chmod("activate.sh", 0o755)
        print("✅ Created activate.sh for Unix/Linux/macOS")


def create_vscode_settings():
    """Create VS Code settings for the project."""
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)
    
    python_exe = get_python_executable()
    
    # Settings for Python interpreter
    settings = {
        "python.pythonPath": python_exe,
        "python.terminal.activateEnvironment": True,
        "python.linting.enabled": True,
        "python.linting.pylintEnabled": False,
        "python.linting.flake8Enabled": True,
        "python.formatting.provider": "black",
        "python.sortImports.args": ["--profile", "black"],
        "files.associations": {
            "*.py": "python"
        },
        "terminal.integrated.env.windows": {
            "PYTHONPATH": "${workspaceFolder}/src"
        },
        "terminal.integrated.env.osx": {
            "PYTHONPATH": "${workspaceFolder}/src"
        },
        "terminal.integrated.env.linux": {
            "PYTHONPATH": "${workspaceFolder}/src"
        }
    }
    
    import json
    with open(vscode_dir / "settings.json", "w") as f:
        json.dump(settings, f, indent=2)
    
    print("✅ Created VS Code settings")


def run_basic_tests():
    """Run basic functionality tests."""
    print("\n🧪 Running basic tests...")
    python_exe = get_python_executable()
    
    # Test 1: Check virtual environment Python
    success = run_command(f"{python_exe} --version", "Testing virtual environment Python")
    if not success:
        return False
    
    # Test 2: Test imports
    test_script = '''
import sys
print(f"Python path: {sys.executable}")
print(f"Virtual environment: {'venv' in sys.executable}")

try:
    import sqlite3
    print("✅ sqlite3 available")
except ImportError:
    print("❌ sqlite3 not available")

try:
    import json
    print("✅ json available")
except ImportError:
    print("❌ json not available")

try:
    import os
    print("✅ os available")
except ImportError:
    print("❌ os not available")

# Test pandas if available
try:
    import pandas as pd
    print("✅ pandas available")
except ImportError:
    print("⚠️  pandas not available (install with pip)")

# Test project imports
sys.path.insert(0, "src")
try:
    from config.settings import settings
    print("✅ Project configuration available")
except Exception as e:
    print(f"❌ Project configuration failed: {e}")
'''
    
    # Write test script to temporary file
    with open("test_venv.py", "w") as f:
        f.write(test_script)
    
    try:
        success = run_command(f"{python_exe} test_venv.py", "Testing project imports")
        return success
    finally:
        # Clean up test file
        if Path("test_venv.py").exists():
            os.remove("test_venv.py")


def show_usage_instructions():
    """Show usage instructions."""
    activation_cmd = get_activation_command()
    python_exe = get_python_executable()
    pip_exe = get_pip_executable()
    
    print("\n🎉 Virtual Environment Setup Complete!")
    print("=" * 60)
    
    print("\n📋 How to use your virtual environment:")
    
    if platform.system() == "Windows":
        print(f"1. Activate the environment:")
        print(f"   {activation_cmd}")
        print(f"   # or simply run: activate.bat")
        print(f"")
        print(f"2. Deactivate when done:")
        print(f"   deactivate")
    else:
        print(f"1. Activate the environment:")
        print(f"   {activation_cmd}")
        print(f"   # or simply run: ./activate.sh")
        print(f"")
        print(f"2. Deactivate when done:")
        print(f"   deactivate")
    
    print(f"\n🐍 Python executable: {python_exe}")
    print(f"📦 Pip executable: {pip_exe}")
    
    print(f"\n🔧 Common commands:")
    print(f"   Install packages: {pip_exe} install <package>")
    print(f"   Run project: {python_exe} examples.py")
    print(f"   CLI help: {python_exe} -m src.cli.main --help")
    print(f"   Start MCP server: {python_exe} -m src.mcp.server")
    
    print(f"\n💡 Tips:")
    print(f"   - Always activate the environment before working on the project")
    print(f"   - VS Code should automatically detect the virtual environment")
    print(f"   - The environment is located in the 'venv' directory")
    print(f"   - You can delete 'venv' directory to start fresh")


def main():
    """Main setup function."""
    print("🚀 Advanced Crypto Backtesting System - Virtual Environment Setup")
    print("=" * 80)
    
    # Check Python version
    if not check_python_version():
        print("❌ Please upgrade to Python 3.9 or later")
        return False
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Failed to create virtual environment")
        return False
    
    # Upgrade pip
    if not upgrade_pip():
        print("⚠️  Failed to upgrade pip, continuing...")
    
    # Install dependencies
    install_dependencies()
    
    # Create activation scripts
    create_activation_scripts()
    
    # Create VS Code settings
    create_vscode_settings()
    
    # Run basic tests
    if not run_basic_tests():
        print("⚠️  Some tests failed, but setup is complete")
    
    # Show usage instructions
    show_usage_instructions()
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Setup completed successfully!")
    else:
        print("\n❌ Setup completed with errors")
    
    input("\nPress Enter to exit...")
