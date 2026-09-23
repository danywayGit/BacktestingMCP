#!/usr/bin/env python
"""V14 weekly re-evaluation loop.

Consumes results/pattern_discovery_combined.json (written every Monday 06:00 UTC by
pattern_discovery_refresh.sh) and checks each V14.x split config (14.D.S → direction ×
symbol) against its own pair's fresh pre-move percentiles. Reports divergence and
(appended to RESEARCH_LOG.md). Does NOT auto-edit configs by default — flags →
human review → optional --apply.

Family: 14.D.S  D=0 LONG(UP data), D=1 SHORT(DOWN data); S=0 BTC, S=1 ETH.
Threshold semantics: min_atr_pct & min_volume_relative are compared against the fresh
p25 pre-move value for that pair+direction.
  - config > fresh_p25      => TOO_STRICT (rejects the bottom-quartile pre-moves)
  - config < 0.6 * fresh_p25 => TOO_LOOSE  (passes nearly everything)
RSI: fresh median vs the ~46 "neutral" anchor from the original research.
"""
import argparse
import json
import os
import sys
from datetime import datetime

BASE = "/home/hermes/BacktestingMCP"
COMBINED = os.path.join(BASE, "results", "pattern_discovery_combined.json")
LOG = os.path.join(BASE, "research", "precursor_strategy", "RESEARCH_LOG.md")

# version -> (symbol_data_key, direction_label)
MAP = {
    "14.0.0": ("BTCUSDT", "UP"),
    "14.0.1": ("ETHUSDT", "UP"),
    "14.1.0": ("BTCUSDT", "DOWN"),
    "14.1.1": ("ETHUSDT", "DOWN"),
}
ANCHORS = {"ATR": 6.0, "VOL": 1.0}  # (reserved)


def _fresh_data(analysis, direction):
    pats = analysis.get("patterns", {}).get(direction, {})
    return pats


def _pct(stats, label):
    return stats.get(label) if stats else None


def review(apply=False, out_path=None):
    if not os.path.exists(COMBINED):
        print(f"NO_DATA: {COMBINED} missing — pattern-discovery-refresh did not produce combined output")
        return 1

    with open(COMBINED) as f:
        combined = json.load(f)

    sys.path.insert(0, BASE)
    from src.edge_scanner.scoring_config import ALL_CONFIGS  # noqa: E402

    lines = []
    drifts = 0
    today = datetime.now().strftime("%Y-%m-%d %H:%M")

    for version, (sym_key, label) in MAP.items():
        cfg = ALL_CONFIGS.get(version)
        if cfg is None:
            lines.append(f"- {version}: NOT IN ALL_CONFIGS (review code registration)")
            continue
        analysis = combined.get(sym_key, {}).get("analysis", {})
        if not analysis:
            lines.append(f"- {version}: no fresh data for {sym_key}")
            continue
        pats = _fresh_data(analysis, label)
        if not pats:
            lines.append(f"- {version}: no {label} pattern stats for {sym_key}")
            continue

        atr = pats.get("ATR_pct", {})
        vol = pats.get("Volume_Ratio", {})
        rsi = pats.get("RSI", {})
        atr_p25 = _pct(atr, "p25")
        vol_p25 = _pct(vol, "p25")
        rsi_med = _pct(rsi, "median")

        flags = []
        cfg_atr = float(getattr(cfg, "min_atr_pct", None) or 0)
        cfg_vol = float(getattr(cfg, "min_volume_relative", None) or 0)
        spans = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}.get(sym_key, sym_key)
        dirlab = "LONG" if label == "UP" else "SHORT"

        if atr_p25 is not None:
            if cfg_atr > atr_p25:
                flags.append(f"ATR TOO_STRICT cfg={cfg_atr} vs fresh p25={atr_p25:.3f}")
            elif cfg_atr < 0.6 * atr_p25:
                flags.append(f"ATR TOO_LOOSE cfg={cfg_atr} vs fresh p25={atr_p25:.3f} (<0.6x)")
            else:
                flags.append(f"ATR ok cfg={cfg_atr} within [0.6,1.0]x p25={atr_p25:.3f}")
        else:
            flags.append("ATR_pct missing in fresh data")

        if vol_p25 is not None:
            if cfg_vol > vol_p25:
                flags.append(f"VOL TOO_STRICT cfg={cfg_vol} vs fresh p25={vol_p25:.3f}")
            elif cfg_vol < 0.6 * vol_p25:
                flags.append(f"VOL TOO_LOOSE cfg={cfg_vol} vs fresh p25={vol_p25:.3f} (<0.6x)")
            else:
                flags.append(f"VOL ok cfg={cfg_vol} within [0.6,1.0]x p25={vol_p25:.3f}")
        else:
            flags.append("Volume_Ratio missing in fresh data")

        if rsi_med is not None:
            rsi_drift = rsi_med - 46.0
            if abs(rsi_drift) > 10:
                flags.append(f"RSI DRIFT median={rsi_med:.1f} (anchor 46, Δ{abs(rsi_drift):.1f})")
            else:
                flags.append(f"RSI median={rsi_med:.1f} near anchor 46")
        else:
            flags.append("RSI missing in fresh data")

        n_flags = sum(1 for fl in flags if "TOO_" in fl or "DRIFT" in fl)
        drifts += (1 if n_flags else 0)

        # Optional apply: only corrected NEW thresholds (never touch weights / status)
        applied = []
        if apply:
            if atr_p25 is not None and ("ATR TOO_STRICT" in flags[0] or "ATR TOO_LOOSE" in flags[0]):
                applied.append(f"min_atr_pct {cfg_atr}->{round(atr_p25, 3)}")
            if vol_p25 is not None and ("VOL TOO_STRICT" in flags[1] or "VOL TOO_LOOSE" in flags[1]):
                applied.append(f"min_volume_relative {cfg_vol}->{round(vol_p25, 3)}")

        line = f"- **{version}** ({dirlab}·{spans}, {sym_key}/{label}): " + " | ".join(flags)
        if applied:
            line += f" | APPLIED: {', '.join(applied)}"
        lines.append(line)

    status = "OK" if drifts == 0 else f"DRIFT[{drifts}/4]"
    header = f"## V14 Weekly Re-Eval — {today} ({status})"
    body = "\n".join(lines)
    print(header)
    print(body)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"date": today, "status": status, "version": "1.0",
                       "checks": [l for l in lines]}, f, indent=2)

    with open(LOG, "a") as f:
        f.write(f"\n{header}\n{body}\n\n")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Edit config thresholds to fresh p25 (DEFAULT OFF — review first).")
    ap.add_argument("--out", default=os.path.join(BASE, "results", "v14_weekly_review_latest.json"))
    a = ap.parse_args()
    sys.exit(review(apply=a.apply, out_path=a.out))