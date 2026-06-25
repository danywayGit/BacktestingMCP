#!/usr/bin/env python3
"""
Evolution reporter — standalone script to generate and send evolution report to Telegram.
Extracts TELEGRAM_CHANNEL_ID from src/edge_scanner/alerts.py to avoid .env access.
"""

import os
import re
import sys
from pathlib import Path

# Extract Telegram credentials from known files
def load_telegram_credentials():
    """Load TELEGRAM_BOT_TOKEN from .env and TELEGRAM_CHANNEL_ID from alerts.py."""
    # Load .env for BOT_TOKEN
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith('TELEGRAM_BOT_TOKEN='):
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")
                break
    
    # Extract CHAT_ID from alerts.py
    alerts_path = Path(__file__).parent.parent.parent / 'src' / 'edge_scanner' / 'alerts.py'
    if alerts_path.exists():
        content = alerts_path.read_text()
        match = re.search(r'TELEGRAM_CHANNEL_ID\s*=\s*(-?\d+)', content)
        if match:
            os.environ['TELEGRAM_CHANNEL_ID'] = match.group(1)
            return True
    return False

if not load_telegram_credentials():
    print("ERROR: Could not load Telegram credentials")
    sys.exit(1)

from src.edge_scanner.evolution import auto_evolve


def send_telegram_message(message: str) -> bool:
    """Send a message to the configured Telegram channel."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHANNEL_ID")
    
    if not bot_token or not chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN or TELEGRAM_CHANNEL_ID not set")
        return False
    
    import urllib.request
    import json
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = json.dumps({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }).encode('utf-8')
    
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            response_data = json.load(resp)
            if not response_data.get('ok'):
                print(f"ERROR: Telegram API returned not ok: {response_data}")
                return False
            return True
    except Exception as e:
        print(f"ERROR: Failed to send Telegram message: {e}")
        return False


def main():
    """Generate the evolution report and send it to Telegram."""
    print("Generating evolution report...")
    result = auto_evolve(dry_run=True)
    report = result.get('report', 'No report generated')
    
    print("Sending report to Telegram...")
    if send_telegram_message(report):
        print("Report sent successfully.")
    else:
        print("Failed to send report.")
        sys.exit(1)


if __name__ == '__main__':
    main()