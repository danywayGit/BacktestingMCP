#!/usr/bin/env python3
"""Daily evolution report generator - sends performance analysis to Telegram."""
import os
import sys
import json
import urllib.request
from datetime import datetime, timezone

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load env vars
if os.path.exists('.env'):
    for line in open('.env').readlines():
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

sys.path.insert(0, '.')
from src.edge_scanner.evolution import auto_evolve, analyze_configs
from src.edge_scanner.llm_evolver import auto_evolve_with_llm

# --- Run analysis ---
result = auto_evolve('data/crypto.db', dry_run=True)
stats = analyze_configs('data/crypto.db')

# Active config
active_version = result['active_config']
active = stats.get(active_version)

# Rank configs (min 20 trades)
from src.edge_scanner.evolution import rank_configs, MIN_NON_FLAT_TRADES
ranked = rank_configs(stats)
# Filter out the v-prefixed duplicates of the same family
filtered = []
seen_scores = set()
for c in ranked:
    key = round(c.composite_rank_score, 1)
    if key not in seen_scores:
        seen_scores.add(key)
        filtered.append(c)

# LLM evolution
llm_result = auto_evolve_with_llm('data/crypto.db', dry_run=True)

# Count pending
import sqlite3
conn = sqlite3.connect('data/crypto.db')
pending = conn.execute("SELECT COUNT(*) FROM edge_signals WHERE status = 'PENDING'").fetchone()[0]
resolved = conn.execute("SELECT COUNT(*) FROM edge_signals WHERE status = 'RESOLVED'").fetchone()[0]
total = conn.execute("SELECT COUNT(*) FROM edge_signals").fetchone()[0]
conn.close()

# --- Build report ---
lines = []

def bold(s): return f"<b>{s}</b>"
def code(s): return f"<code>{s}</code>"
def italic(s): return f"<i>{s}</i>"

lines.append(bold("Daily Evolution Report"))
lines.append(italic(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"))
lines.append("")

# Header
if active:
    lines.append(f"Active: {code(active_version)} — {active.win_rate:.1f}% WR | {active.flat_rate:.1f}% flat | {active.non_flat_trades} non-flat trades | PF {active.profit_factor:.2f}")
lines.append(f"📊 DB: {total:,} total | {pending:,} pending | {resolved:,} resolved")
lines.append("")

# Promotion status
lines.append(bold("Auto-Promotion"))
if result['action'] == 'promote':
    lines.append(f"  Candidate: {code(result['recommended_config'])} — statistically outperforms active (dry run)")
    if result.get('recommended_config') and result['recommended_config'] in stats:
        cc = stats[result['recommended_config']]
        lines.append(f"  {cc.win_rate:.1f}% WR vs {active.win_rate:.1f}% active | {cc.non_flat_trades} trades")
elif result.get('recommended_config'):
    lines.append(f"  Top candidate: {code(result['recommended_config'])} (not significant enough for promotion)")
else:
    lines.append("  No candidate qualifies")
lines.append("")

# Top 10 configs
lines.append(bold("Config Rankings (top 10)"))
lines.append("<pre>")
hdr = f"{'Config':<10} {'WR%':>6} {'Flat%':>7} {'Trades':>7} {'Quality':>8} {'PF':>6} {'Score':>7}"
lines.append(hdr)
lines.append("-" * len(hdr))
for cfg in filtered[:10]:
    marker = " ← ACTIVE" if cfg.config_version == active_version else ""
    pf = f"{cfg.profit_factor:.2f}" if cfg.profit_factor != float('inf') else "INF"
    row = f"{cfg.config_version:<10} {cfg.win_rate:>5.1f}% {cfg.flat_rate:>6.1f}% {cfg.non_flat_trades:>3}/{cfg.total_signals:<4} {cfg.signal_quality_score:>7.1f} {pf:>6} {cfg.composite_rank_score:>6.1f}{marker}"
    lines.append(row)
lines.append("</pre>")
lines.append(italic("* = active config | Min {0} trades".format(MIN_NON_FLAT_TRADES)))
lines.append("")

# Notable variants
lines.append(bold("Notable Variants"))
notable_variants = ['1.4', '1.0', '1.5', '7.0', '7.2', '7.5', '8.0', '6.0', '3.1', '14.0', 'v6.2']
for v in notable_variants:
    c = stats.get(v)
    if c and c.non_flat_trades >= 10:  # show even low-trade configs
        pf = f"{c.profit_factor:.2f}" if c.profit_factor != float('inf') else "INF"
        lines.append(f"  {code(v)}: WR={c.win_rate:.1f}% | Flat={c.flat_rate:.1f}% | {c.non_flat_trades} trades | PF={pf}")
lines.append("")

# LLM suggestion
if llm_result['action'] == 'llm_generate':
    lines.append(bold("LLM Evolution Suggestion"))
    lines.append(f"  {llm_result['reason']}")
    nc = llm_result.get('new_config')
    if nc:
        lines.append(f"  Key changes from {active_version}:")
        for k in ['min_abs_score', 'min_adx', 'min_rsi', 'max_rsi', 'min_atr_pct', 'rr_ratio', 'trend_weight']:
            if k in nc:
                old_val = None
                if hasattr(active, 'to_dict'):
                    old_val = active.to_dict().get(k)
                old_str = f" (was {old_val})" if old_val is not None else ""
                lines.append(f"    {k}: {nc[k]}{old_str}")
lines.append("")

lines.append(italic("Next report: tomorrow 18:00 UTC"))

report_text = "\n".join(lines)

print(report_text)
print("\n--- End Report ---", flush=True)

# --- Send to Telegram ---
bot_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
chat_id = -1001482338614

if not bot_token:
    print("ERROR: No TELEGRAM_BOT_TOKEN found")
    sys.exit(1)

url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
data = json.dumps({
    "chat_id": chat_id,
    "text": report_text,
    "parse_mode": "HTML"
}).encode('utf-8')

req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        response_data = json.load(resp)
        if response_data.get('ok'):
            print("\n[OK] Report sent to Telegram successfully!")
        else:
            print(f"\n[FAIL] Telegram API error: {response_data}")
            sys.exit(1)
except Exception as e:
    print(f"\n[FAIL] Telegram send failed: {e}")
    # Try without parse mode as fallback
    print("  Retrying without parse_mode...")
    data = json.dumps({
        "chat_id": chat_id,
        "text": report_text,
    }).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        response_data = json.load(resp)
        if response_data.get('ok'):
            print("  [OK] Sent (plain text)")
        else:
            print(f"  [FAIL] Still failed: {response_data}")
            sys.exit(1)