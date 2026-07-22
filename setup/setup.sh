#!/usr/bin/env bash
# =============================================================================
# Edge Scanner — Automated Setup
# =============================================================================
# This script installs all dependencies, creates the database, and validates
# the environment. Run ONCE after cloning the repo.
#
# Usage:
#   chmod +x setup/setup.sh
#   ./setup/setup.sh
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "=== Edge Scanner Setup ==="
echo "Repo dir: $REPO_DIR"
echo ""

# ── 1. Python virtual environment ──────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "[1/5] Creating Python virtual environment..."
    python3 -m venv venv
    echo "  ✅ venv created"
else
    echo "[1/5] Python virtual environment already exists ✅"
fi

source venv/bin/activate

# ── 2. Install dependencies ───────────────────────────────────────────────
echo "[2/5] Installing Python dependencies..."
pip install -q -r requirements.txt 2>&1 | tail -1
echo "  ✅ Dependencies installed"

# ── 3. Environment file ───────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    echo "[3/5] Creating .env from template..."
    cp .env.template .env
    echo "  ⚠️  EDIT .env with your API keys before running!"
    echo "  ⚠️  Required: ALTFINS_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID"
else
    echo "[3/5] .env file already exists ✅"
fi

# ── 4. Initialize database ────────────────────────────────────────────────
echo "[4/5] Initializing database..."
mkdir -p data
# Run a quick Python command to trigger DB creation
source .env 2>/dev/null || true
venv/bin/python -c "
from src.data.database import init_db
init_db()
print('  ✅ Database initialized')
" 2>&1 | grep -v "^INFO:" || echo "  ⚠️  Could not init DB (set up .env first)"
echo "  ✅ Database directory created"

# ── 5. Validate setup ─────────────────────────────────────────────────────
echo "[5/5] Validating setup..."
source .env 2>/dev/null || true
VALID=true

if [ -z "${ALTFINS_API_KEY:-}" ] || [ "$ALTFINS_API_KEY" = "your_altfins_api_key_here" ]; then
    echo "  ⚠️  ALTFINS_API_KEY not set — pattern scanning won't work"
    VALID=false
fi
if [ -z "${TELEGRAM_BOT_TOKEN:-}" ] || [ "$TELEGRAM_BOT_TOKEN" = "your_telegram_bot_token_here" ]; then
    echo "  ⚠️  TELEGRAM_BOT_TOKEN not set — alerts disabled"
    VALID=false
fi
if [ -z "${TELEGRAM_CHAT_ID:-}" ]; then
    echo "  ⚠️  TELEGRAM_CHAT_ID not set — alerts won't send"
    VALID=false
fi

if [ "$VALID" = true ]; then
    echo "  ✅ All required env vars set"
fi

# ── CLI test ──────────────────────────────────────────────────────────────
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Run: source venv/bin/activate"
echo "  3. Test: python -m src.cli.main edge configs"
echo "  4. Install crons: bash setup/install_crons.sh"
echo ""