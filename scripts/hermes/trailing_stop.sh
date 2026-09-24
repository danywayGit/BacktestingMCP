#!/bin/bash
# Trailing-stop engine — runs every 3 min on Hermes.
# Reads open positions via the bot's /api/bybit_risk (and /api/positions for
# other exchanges), computes R, and POSTs MoveStopLoss webhooks to the bot
# when a stop should be trailed. Logs to a rotating file.
cd /home/hermes
# Load TRAILING_BOT_API_KEY from the gitignored Hermes .env (never in code/git).
set -a; . /home/hermes/.hermes/.env 2>/dev/null; set +a
OUT=$(timeout 50 python3 ~/.hermes/scripts/trailing_stop.py 2>/dev/null | grep -v CuPy)
echo "$OUT" >> ~/.hermes/logs/trailing_stop.log
# Only print (deliver) if something moved, so the cron is quiet otherwise.
echo "$OUT" | grep -q "SL move" && echo "$OUT"
