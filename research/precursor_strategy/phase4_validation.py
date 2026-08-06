"""
Phase 4 — Out-of-Sample Validation
Tests whether precursors hold up on held-out data.
"""
import sys, csv, json
from pathlib import Path
import numpy as np
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
OUTPUT_DIR = Path(__file__).parent
FEATURES_FILE = OUTPUT_DIR / "precursor_features.csv"
EVENTS_FILE = OUTPUT_DIR / "events.csv"
TRAIN_SPLIT = 0.70

def load_features():
    rows = []
    with open(FEATURES_FILE) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def load_events():
    events = {}
    with open(EVENTS_FILE) as f:
        for row in csv.DictReader(f):
            events[row["event_id"]] = row
    return events

def cohen_d(vals1, vals2):
    if len(vals1) < 2 or len(vals2) < 2:
        return 0
    m1, m2 = np.mean(vals1), np.mean(vals2)
    s1, s2 = np.std(vals1, ddof=1), np.std(vals2, ddof=1)
    pooled = np.sqrt(((len(vals1)-1)*s1**2 + (len(vals2)-1)*s2**2) / (len(vals1)+len(vals2)-2))
    if pooled == 0: return 0
    return (m1 - m2) / pooled

def validate_precursors(rows, events):
    event_times = {eid: e["event_start_ts"] for eid, e in events.items()}
    sorted_events = sorted(event_times.items(), key=lambda x: x[1])
    split_idx = int(len(sorted_events) * TRAIN_SPLIT)
    train_ids = set(e[0] for e in sorted_events[:split_idx])
    test_ids = set(e[0] for e in sorted_events[split_idx:])
    print(f"  Train events: {len(train_ids)} | Test events: {len(test_ids)}")

    precursors = [("atr_pct","ATR %"),("bb_width_pct","BB Width %"),("volume_zscore","Vol Z-Score"),
                  ("volume_ratio","Vol Ratio"),("ll_ratio","LL Ratio"),("hh_ratio","HH Ratio"),("rsi_level","RSI")]
    results = []

    for feat_key, feat_name in precursors:
        for sym in ["BTC", "ETH"]:
            for K in [60, 120, 240]:
                train_ev, test_ev, ctrl_vals = [], [], []
                for r in rows:
                    if r["lookback_K"] != str(K):
                        continue
                    try:
                        v = float(r[feat_key]) if r[feat_key] else 0
                    except:
                        continue
                    if r["is_control"] == "True":
                        ctrl_vals.append(v)
                    elif sym in r["event_id"] and r["event_id"] in train_ids:
                        train_ev.append(v)
                    elif sym in r["event_id"] and r["event_id"] in test_ids:
                        test_ev.append(v)

                if len(train_ev) < 3 or len(test_ev) < 3 or len(ctrl_vals) < 3:
                    continue

                train_d = cohen_d(train_ev, ctrl_vals)
                test_d = cohen_d(test_ev, ctrl_vals)

                results.append({
                    "symbol": sym, "feature": feat_name, "K": K,
                    "train_d": round(train_d, 3), "test_d": round(test_d, 3),
                    "train_n": len(train_ev), "test_n": len(test_ev), "ctrl_n": len(ctrl_vals),
                    "train_med": round(np.median(train_ev), 4),
                    "test_med": round(np.median(test_ev), 4),
                    "ctrl_med": round(np.median(ctrl_vals), 4),
                    "validated": abs(test_d) > 0.3,
                })
    return results

if __name__ == "__main__":
    print("\n  ── Phase 4: Out-of-Sample Validation ──")
    rows = load_features()
    events = load_events()
    print(f"  Loaded {len(rows)} feature rows, {len(events)} events")

    results = validate_precursors(rows, events)

    validated = [r for r in results if r["validated"]]
    failed = [r for r in results if not r["validated"] and abs(r["train_d"]) > 0.3]

    print(f"\n{'='*70}")
    print(f"  VALIDATED PRECURSORS (|test_d| > 0.3)")
    print('='*70)
    if validated:
        print(f"  {'Sym':4} {'Feature':14} {'K':4} {'Train d':8} {'Test d':8} {'TrainMed':8} {'TestMed':8} {'CtrlMed':8}")
        for r in sorted(validated, key=lambda x: abs(x["test_d"]), reverse=True):
            print(f"  {r['symbol']:4} {r['feature']:14} {r['K']:4} {r['train_d']:8.2f} {r['test_d']:8.2f} {r['train_med']:8.2f} {r['test_med']:8.2f} {r['ctrl_med']:8.2f}")
    else:
        print("  ⚠️ No precursors survived out-of-sample validation.")

    if failed:
        print(f"\n❌ Failed OOS (strong in-sample, weak OOS):")
        for r in sorted(failed, key=lambda x: abs(x["train_d"]-x["test_d"]), reverse=True)[:5]:
            print(f"  {r['symbol']:4} {r['feature']:14} K={r['K']:3} train_d={r['train_d']:.2f} → test_d={r['test_d']:.2f}")

    with open(OUTPUT_DIR / "phase4_validation.json", "w") as f:
        json.dump({"results": results}, f, indent=2, default=str)
    print(f"\n  Full report: {OUTPUT_DIR / 'phase4_validation.json'}")
    print(f"  [HUMAN CHECKPOINT]")
    print()