"""
Multi-source Burn / Buyback Event Tracker (2026-08-27 rebuild).

Data layers (all probed live & verified):
  L1  On-chain ground truth   : balance-delta at verified burn addresses via public RPCs
                                (native: eth_getBalance / erc20: eth_call balanceOf)
  L2  CMC supply cross-check  : maxSupply, totalSupply, circulatingSupply deltas
                                (derived burn for capped coins; total-supply delta over time)
  L3  Binance announcements   : official burn/buyback announcements (quarterly BNB auto-burn)
  L4  News RSS                : Cointelegraph / Decrypt / CoinDesk keyword scan (buybacks)

Design principles:
  - No API keys required (all sources public / free)
  - Per-entry failure isolation: one dead RPC never kills the report
  - State file keeps baselines; first run = baseline, second run = deltas
  - Watchlist entries verified live before being enabled
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree

import httpx
import json
import logging
import re

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config" / "burn_watchlist.json"
STATE_PATH = BASE_DIR / "data" / "burn_state.json"

# ---- RPC helpers -----------------------------------------------------------

def rpc_call(rpc: str, method: str, params: List) -> Optional[str]:
    """Single JSON-RPC call; returns result or None on any failure."""
    try:
        resp = httpx.post(
            rpc,
            json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
            timeout=10,
        )
        data = resp.json()
        if "result" in data:
            return data["result"]
        logger.debug("RPC %s %s error: %s", rpc, method, data.get("error"))
    except Exception as e:
        logger.debug("RPC %s %s failed: %s", rpc, method, e)
    return None


def eth_get_balance(rpc: str, addr: str) -> Optional[int]:
    raw = rpc_call(rpc, "eth_getBalance", [addr, "latest"])
    return int(raw, 16) if raw else None


def erc20_balance_of(rpc: str, contract: str, addr: str) -> Optional[int]:
    # balanceOf(address) selector: 0x70a08231 + padded address
    data = "0x70a08231" + addr[2:].lower().rjust(64, "0")
    raw = rpc_call(rpc, "eth_call", [{"to": contract, "data": data}, "latest"])
    if not raw or raw == "0x":
        return None
    try:
        return int(raw, 16)
    except ValueError:
        return None


def normalize_balance(raw: Optional[int], decimals: int) -> Optional[float]:
    if raw is None:
        return None
    return int(raw) / (10 ** int(decimals))


# ---- L1: on-chain sampler --------------------------------------------------

def sample_onchain(entry: Dict) -> Optional[float]:
    """Return current burned balance for a watchlist entry, or None on failure."""
    if entry.get("mode") == "native":
        raw = eth_get_balance(entry["rpc"], entry["burn_addr"])
    else:  # erc20
        raw = erc20_balance_of(entry["rpc"], entry["contract"], entry["burn_addr"])
    return normalize_balance(raw, entry.get("decimals", 18))


# ---- L2: CMC supply cross-check --------------------------------------------

def fetch_cmc_supply(slug: str) -> Optional[Dict]:
    """maxSupply / totalSupply / circulatingSupply from CMC web API (no key)."""
    try:
        resp = httpx.get(
            f"https://api.coinmarketcap.com/data-api/v3/cryptocurrency/detail?slug={slug}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        stats = resp.json().get("data", {}).get("statistics", {})
        return {
            "max": stats.get("maxSupply"),
            "total": stats.get("totalSupply"),
            "circ": stats.get("circulatingSupply"),
        }
    except Exception as e:
        logger.debug("CMC %s failed: %s", slug, e)
        return None


# ---- L3: Binance announcements ---------------------------------------------

BINANCE_CMS = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
BURN_KEYWORDS = ("burn", "buyback", "buy back", "auto-burn")


def fetch_binance_announcements() -> List[Dict]:
    """Scan recent Binance announcement articles for burn/buyback mentions."""
    hits = []
    try:
        for page in (1, 2, 3):
            resp = httpx.get(
                BINANCE_CMS,
                params={"type": 1, "pageNo": page, "pageSize": 30},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            if data.get("code") != "000000":
                continue
            for art in data.get("data", {}).get("articles", []):
                title = art.get("title", "")
                tl = title.lower()
                if any(k in tl for k in BURN_KEYWORDS):
                    hits.append({
                        "title": title,
                        "date": art.get("releaseDate", ""),
                        "url": f"https://www.binance.com/en/support/announcement/{art.get('code', '')}",
                    })
    except Exception as e:
        logger.debug("Binance CMS failed: %s", e)
    return hits


# ---- L4: news RSS -----------------------------------------------------------

NEWS_FEEDS = [
    ("CoinDesk",  "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ("Decrypt",   "https://decrypt.co/feed"),
]


def scan_news_feed(name: str, url: str) -> List[Dict]:
    hits = []
    try:
        resp = httpx.get(url, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        if resp.status_code != 200:
            return hits
        root = ElementTree.fromstring(resp.text)
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            tl = title.lower()
            if any(k in tl for k in BURN_KEYWORDS):
                link = (item.findtext("link") or "").strip()
                hits.append({"title": title[:160], "url": link, "source": name})
    except Exception as e:
        logger.debug("Feed %s failed: %s", name, e)
    return hits


# ---- State handling ---------------------------------------------------------

def load_state() -> Dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"onchain": {}, "cmc": {}}


def save_state(state: Dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


# ---- Report formatting ------------------------------------------------------

def fmt_num(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    if abs(v) >= 1e12:
        return f"{v/1e12:,.2f}T"
    if abs(v) >= 1e9:
        return f"{v/1e9:,.2f}B"
    if abs(v) >= 1e6:
        return f"{v/1e6:,.2f}M"
    if abs(v) >= 1e3:
        return f"{v/1e3:,.2f}K"
    return f"{v:,.2f}"


def derived_burn_str(s: Dict) -> str:
    """max - total, shown only when the gap is meaningful (>1% of supply)."""
    if s.get("max") and s.get("total"):
        gap = s["max"] - s["total"]
        if gap > 0 and gap > 0.01 * s["total"]:
            return f" | derived burned {fmt_num(gap)}"
    return ""


def build_report(onchain_rows: List[str], cmc_rows: List[str],
                 anns: List[Dict], news: List[Dict], first_run: bool) -> str:
    d = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"🔥 *Burn / Buyback Tracker — {d}*", ""]

    if first_run:
        lines.append("_Baseline recorded — deltas start next run._")
        lines.append("")

    lines.append("📡 *On-chain (exact, since last run):*")
    lines.extend(onchain_rows or ["  _no data / rpc unreachable_"])

    lines.append("")
    lines.append("📊 *Supply cross-check (CMC):*")
    lines.extend(cmc_rows or ["  _no data_"])

    if anns:
        lines.append("")
        lines.append("📢 *Official announcements:*")
        for a in anns[:5]:
            lines.append(f"  • {a['title'][:80]}")
            if a.get("url"):
                lines.append(f"    {a['url']}")

    if news:
        lines.append("")
        lines.append("📰 *News (burn/buyback):*")
        for n in news[:6]:
            lines.append(f"  • [{n['source']}] {n['title']}")
            if n.get("url"):
                lines.append(f"    {n['url']}")

    lines.append("")
    lines.append("_Sources: on-chain RPCs, CMC, Binance CMS, RSS feeds | weekly_")
    return "\n".join(lines)


# ---- Main pipeline ----------------------------------------------------------

def run_burn_check() -> str:
    watchlist = json.loads(CONFIG_PATH.read_text())
    state = load_state()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    onchain_rows: List[str] = []
    cmc_rows: List[str] = []
    new_state = {"onchain": {}, "cmc": {}}

    first_run = not state.get("onchain")

    # L1 on-chain
    for entry in watchlist.get("onchain", []):
        sym = entry["symbol"]
        bal = sample_onchain(entry)
        prev = state.get("onchain", {}).get(sym, {}).get("balance")
        if bal is None:
            onchain_rows.append(f"  ⚠️ {sym}: RPC unreachable (kept baseline)")
            new_state["onchain"][sym] = {"balance": prev, "date": today}
            continue
        new_state["onchain"][sym] = {"balance": bal, "date": today}
        if prev is not None:
            delta = bal - prev
            if delta > 1e-9:
                onchain_rows.append(
                    f"  🔥 {sym}: +{fmt_num(delta)} burned since {state['onchain'][sym].get('date', 'last run')} (total {fmt_num(bal)})")
            else:
                onchain_rows.append(f"  • {sym}: no change (total {fmt_num(bal)})")
        else:
            onchain_rows.append(f"  📌 {sym}: baseline {fmt_num(bal)} (first sample)")

    # L2 CMC supply
    for entry in watchlist.get("supply_cmc", []):
        sym = entry["symbol"]
        s = fetch_cmc_supply(entry["slug"])
        if not s:
            cmc_rows.append(f"  ⚠️ {sym}: CMC unavailable")
            continue
        prev_total = state.get("cmc", {}).get(sym, {}).get("total")
        new_state["cmc"][sym] = {"total": s["total"], "max": s["max"], "circ": s["circ"], "date": today}
        derived = derived_burn_str(s)
        if prev_total is not None and s["total"] is not None:
            t_delta = s["total"] - prev_total
            if t_delta < -1:
                cmc_rows.append(f"  🔥 {sym}: supply {fmt_num(prev_total)} → {fmt_num(s['total'])} (-{fmt_num(-t_delta)} burned)")
            else:
                cmc_rows.append(f"  • {sym}: total {fmt_num(s['total'])}{derived}")
        else:
            cmc_rows.append(f"  📌 {sym}: baseline total {fmt_num(s['total'])}{derived}")

    # L3 Binance announcements
    anns = fetch_binance_announcements()

    # L4 news feeds
    news: List[Dict] = []
    for name, url in NEWS_FEEDS:
        news.extend(scan_news_feed(name, url))

    save_state(new_state)
    return build_report(onchain_rows, cmc_rows, anns, news, first_run)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run_burn_check())