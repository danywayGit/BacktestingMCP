#!/usr/bin/env bash
# =============================================================================
# Edge Scanner — Cron Job Installer
# =============================================================================
# Installs all scheduled tasks for the Edge Scanner system.
# Run AFTER setup.sh and after configuring .env.
#
# Usage:
#   bash setup/install_crons.sh
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "=== Edge Scanner — Cron Installer ==="
echo "Repo dir: $REPO_DIR"
echo ""

# Load env vars for cron commands
source .env 2>/dev/null || true

# ── Helper: install a cron job ─────────────────────────────────────────────
install_cron() {
    local schedule="$1"
    local name="$2"
    local command="$3"
    local comment="# EdgeScanner: $name"

    # Remove existing job with same name if any
    (crontab -l 2>/dev/null | grep -v "$comment") | crontab -

    # Add new job
    (crontab -l 2>/dev/null; echo "$schedule $comment $command") | crontab -
    echo "  ✅ $name — $schedule"
}

# ── 1. Edge scanner (every 30 min) ────────────────────────────────────────
install_cron \
    "*/30 * * * *" \
    "edge-scan" \
    "cd $REPO_DIR && venv/bin/python -m src.cli.main edge scan --multi 2>&1 | grep -v '^INFO:' >> data/scan.log"

# ── 2. Resolution batches (01:00, 13:00, 17:00 UTC) ──────────────────────
install_cron \
    "0 1,13,17 * * *" \
    "edge-track" \
    "cd $REPO_DIR && venv/bin/python -m src.cli.main edge track 2>&1 | grep -v '^INFO:' >> data/track.log"

# ── 3. Daily summary (09:00 UTC) ──────────────────────────────────────────
install_cron \
    "0 9 * * *" \
    "daily-summary" \
    "cd $REPO_DIR && venv/bin/python -m src.cli.main edge daily-summary 2>&1 | grep -v '^INFO:' | grep -v '^\\[GPU'"

# ── 4. Pattern scan (10:00 UTC) ───────────────────────────────────────────
install_cron \
    "0 10 * * *" \
    "pattern-scan" \
    "cd $REPO_DIR && venv/bin/python -m src.cli.main edge patterns --lookback 'last 24 hours' 2>&1 | grep -v '^INFO:'"

# ── 5. Evolution check (18:00 UTC) ────────────────────────────────────────
install_cron \
    "0 18 * * *" \
    "evolution-check" \
    "cd $REPO_DIR && venv/bin/python -m src.edge_scanner.evolution --no-dry-run 2>&1 | grep -v '^INFO:'"

# ── 6. Gem scan (Monday 08:00 UTC) ────────────────────────────────────────
install_cron \
    "0 8 * * 1" \
    "gem-scan" \
    "cd $REPO_DIR && venv/bin/python -m src.cli.main edge gems --pages 4 --start-page 3 --top 20 2>&1 | grep -v '^INFO:'"

# ── 7. Burn tracker (Saturday 10:00 UTC) ──────────────────────────────────
install_cron \
    "0 10 * * 6" \
    "burn-tracker" \
    "cd $REPO_DIR && venv/bin/python -c 'from src.edge_scanner.burn_tracker import run_burn_check; print(run_burn_check())' 2>&1 | grep -v '^INFO:'"

# ── 8. Webhook bridge (every 30 min, staggered) ──────────────────────────
install_cron \
    "*/30 * * * *" \
    "webhook-bridge" \
    "cd $REPO_DIR && venv/bin/python -m src.edge_scanner.webhook_bridge 2>&1 | grep -v '^INFO:'"

# ── 9. Funding poll (every 15 min) ────────────────────────────────────────
install_cron \
    "*/15 * * * *" \
    "funding-poll" \
    "cd $REPO_DIR && venv/bin/python -c 'from src.integrations.binance_funding import poll_all_funding; poll_all_funding([])' 2>&1 | grep -v '^INFO:'"

echo ""
echo "=== All crons installed ==="
echo "Review with: crontab -l"
echo "Remove all with: crontab -r"
echo ""