#!/usr/bin/env python3
"""Send the daily evolution report to Telegram."""
import json
import os
import sys
import urllib.request
from pathlib import Path

# Load .env manually
env_path = Path('/home/hermes/BacktestingMCP/.env')
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
chat_id = '-1001482338614'

if not bot_token:
    print('ERROR: No bot token')
    sys.exit(1)

# Import the evolution module
sys.path.insert(0, '/home/hermes/BacktestingMCP')
from src.edge_scanner.evolution import auto_evolve, analyze_configs

result = auto_evolve(dry_run=True)
report = result['report']
stats = analyze_configs()

# Enhanced report with analysis
additional = "\n📈 *Quick Analysis*\n"
if 'v1.0' in stats:
    v1 = stats['v1.0']
    additional += f"• v1.0 baseline: {v1.win_rate:.1f}% WR | {v1.flat_rate:.1f}% flat rate | {v1.non_flat_trades} directional trades ({v1.wins}W/{v1.losses}L/{v1.flats}F)\n"
    additional += f"• Quality score: {v1.signal_quality_score:.1f} — {v1.win_rate:.1f}% WR is below 50% breakeven\n"
if '7.0' in stats:
    v70 = stats['7.0']
    additional += f"• Active v7.0: {v70.win_rate:.1f}% WR | {v70.non_flat_trades} directional trades ({v70.wins}W/{v70.losses}L/{v70.flats}F) — too few to evaluate\n"
if 'v6.1' in stats:
    v61 = stats['v6.1']
    additional += f"• v6.1 (resistance breakout) lowest flat rate: {v61.flat_rate:.1f}% — {v61.win_rate:.1f}% WR over {v61.non_flat_trades} trades\n"
if 'v6.0' in stats:
    v60 = stats['v6.0']
    additional += f"• v6.0 (uptrend pullback): {v60.win_rate:.1f}% WR, {v60.flat_rate:.1f}% flat, {v60.avg_return_pct:.2f}% avg return — promising\n"
# Count configs with >10 non-flat trades
meaningful = [(v, c) for v, c in stats.items() if c.non_flat_trades >= 10]
additional += f"• {len(meaningful)} configs with ≥10 directional trades: "
meaningful.sort(key=lambda x: x[1].win_rate, reverse=True)
additional += ", ".join(f"{v}({c.win_rate:.0f}%WR)" for v, c in meaningful[:5])
additional += "\n• No config qualifies for promotion (v7.0 too few trades, v1.0 WR below active)\n"
additional += "_Next report: tomorrow 18:00 UTC_\n"

message = report + additional

url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
payload = json.dumps({
    "chat_id": chat_id,
    "text": message,
    "parse_mode": "Markdown"
}).encode('utf-8')

req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=15) as resp:
    data = json.load(resp)
    if data.get('ok'):
        print("SUCCESS")
    else:
        print(f"FAILED: {data}")
