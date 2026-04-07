#!/usr/bin/env python3
"""Analyze adaptive-v2 vs adaptive-v1 vs baseline-211 benchmark results.

Reads per-request metrics JSON files from results/ directory,
computes cross-seed averages, and prints comparison tables.

Usage: python3 analyze.py <results_dir>
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import statistics


WORKLOAD_ORDER = ["fm1", "fm2a", "fm2b", "fm3", "fm4", "fm5", "fm6"]

WORKLOAD_NAMES = {
    "fm1": "FM-1: Prefix Pile-On",
    "fm2a": "FM-2a: Groups > Instances",
    "fm2b": "FM-2b: Groups < Instances",
    "fm3": "FM-3: Burst Absorption",
    "fm4": "FM-4: Multi-Regime Phased",
    "fm5": "FM-5: Short Output (Classification)",
    "fm6": "FM-6: Cold Traffic Under KV Pressure",
}

POLICY_ORDER = ["baseline-211", "adaptive-v1", "adaptive-v2", "adaptive-v3"]


def load_metrics(filepath):
    """Load summary metrics from BLIS JSON output."""
    with open(filepath) as f:
        data = json.load(f)
    if "e2e_mean_ms" in data:
        return {
            "e2e_mean_ms": data["e2e_mean_ms"],
            "e2e_p99_ms": data["e2e_p99_ms"],
            "ttft_mean_ms": data.get("ttft_mean_ms", 0),
            "ttft_p99_ms": data.get("ttft_p99_ms", 0),
        }
    return None


def parse_filename(filename):
    """Parse: {workload}_{policy}_seed{seed}.json"""
    stem = Path(filename).stem
    parts = stem.rsplit("_seed", 1)
    if len(parts) != 2:
        return None, None, None
    seed = int(parts[1])
    prefix = parts[0]
    for wl in ["fm2a", "fm2b", "fm1", "fm3", "fm4", "fm5", "fm6"]:
        if prefix.startswith(wl + "_"):
            policy = prefix[len(wl) + 1:]
            return wl, policy, seed
    return None, None, None


def avg_metrics(seed_metrics):
    """Compute cross-seed average of metrics. Requires at least 1 seed."""
    if not seed_metrics:
        return None
    return {
        "e2e_mean": statistics.mean([m["e2e_mean_ms"] for m in seed_metrics.values()]),
        "e2e_p99": statistics.mean([m["e2e_p99_ms"] for m in seed_metrics.values()]),
        "ttft_mean": statistics.mean([m["ttft_mean_ms"] for m in seed_metrics.values()]),
        "ttft_p99": statistics.mean([m["ttft_p99_ms"] for m in seed_metrics.values()]),
        "seeds": len(seed_metrics),
    }


def pct_change(baseline, other):
    """Positive = other is better (lower latency)."""
    if baseline == 0:
        return 0.0
    return (baseline - other) / baseline * 100


def print_comparison(data, label_a, label_b, title):
    """Print comparison table between two policies."""
    print("=" * 130)
    print(f"  {title}")
    print(f"  (positive % = {label_b} is better / lower latency)")
    print("=" * 130)
    print()

    hdr = (f"{'Workload':<35} "
           f"{'E2E Mean':>10} {'':>8} "
           f"{'E2E P99':>10} {'':>8} "
           f"{'TTFT Mean':>10} {'':>8} "
           f"{'TTFT P99':>10} {'':>8}")
    sub = (f"{'':35} "
           f"{'BL / AD':>10} {'chg':>8} "
           f"{'BL / AD':>10} {'chg':>8} "
           f"{'BL / AD':>10} {'chg':>8} "
           f"{'BL / AD':>10} {'chg':>8}")
    print(hdr)
    print(sub)
    print("-" * 130)

    wins = []

    for wl in WORKLOAD_ORDER:
        if wl not in data:
            continue
        if label_a not in data[wl] or label_b not in data[wl]:
            continue

        bl = avg_metrics(data[wl][label_a])
        ad = avg_metrics(data[wl][label_b])
        if bl is None or ad is None:
            continue

        chg_e2e_mean = pct_change(bl["e2e_mean"], ad["e2e_mean"])
        chg_e2e_p99 = pct_change(bl["e2e_p99"], ad["e2e_p99"])
        chg_ttft_mean = pct_change(bl["ttft_mean"], ad["ttft_mean"])
        chg_ttft_p99 = pct_change(bl["ttft_p99"], ad["ttft_p99"])

        name = WORKLOAD_NAMES.get(wl, wl)

        def fmt_pair(bl_val, ad_val, chg):
            flag = " *" if chg >= 15 else ""
            return f"{bl_val:>5.0f}/{ad_val:<5.0f} {chg:>+6.1f}%{flag}"

        line = (f"{name:<35} "
                f"{fmt_pair(bl['e2e_mean'], ad['e2e_mean'], chg_e2e_mean):>19} "
                f"{fmt_pair(bl['e2e_p99'], ad['e2e_p99'], chg_e2e_p99):>19} "
                f"{fmt_pair(bl['ttft_mean'], ad['ttft_mean'], chg_ttft_mean):>19} "
                f"{fmt_pair(bl['ttft_p99'], ad['ttft_p99'], chg_ttft_p99):>19}")
        print(line)

        best = max(chg_e2e_mean, chg_e2e_p99, chg_ttft_mean, chg_ttft_p99)
        if best >= 15:
            metrics_won = []
            if chg_e2e_mean >= 15:
                metrics_won.append(f"E2E mean +{chg_e2e_mean:.1f}%")
            if chg_e2e_p99 >= 15:
                metrics_won.append(f"E2E P99 +{chg_e2e_p99:.1f}%")
            if chg_ttft_mean >= 15:
                metrics_won.append(f"TTFT mean +{chg_ttft_mean:.1f}%")
            if chg_ttft_p99 >= 15:
                metrics_won.append(f"TTFT P99 +{chg_ttft_p99:.1f}%")
            wins.append((name, metrics_won))

    print("-" * 130)
    print()
    print("  * = wins by >= 15%")
    print(f"  Seeds per config: 3 | Values in ms ({label_a} / {label_b})")
    print()

    if wins:
        print("  Notable wins (>=15%):")
        for name, metrics in wins:
            print(f"    {name}")
            for m in metrics:
                print(f"      -> {m}")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]

    # Collect: {workload: {policy: {seed: metrics}}}
    data = defaultdict(lambda: defaultdict(dict))

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        wl, policy, seed = parse_filename(fname)
        if wl is None:
            continue
        metrics = load_metrics(os.path.join(results_dir, fname))
        if metrics:
            data[wl][policy][seed] = metrics

    if not data:
        print("No results found!")
        sys.exit(1)

    # Table 1: baseline-211 vs adaptive-v2 (load-aware)
    print_comparison(data, "baseline-211", "adaptive-v2",
                     "Adaptive-v2 (load-aware) vs Baseline 2:1:1")

    # Table 2: baseline-211 vs adaptive-v3 (inflight-requests)
    print_comparison(data, "baseline-211", "adaptive-v3",
                     "Adaptive-v3 (inflight-requests) vs Baseline 2:1:1")

    # Table 3: adaptive-v2 vs adaptive-v3 (stale threshold vs synchronous signal)
    print_comparison(data, "adaptive-v2", "adaptive-v3",
                     "Adaptive-v3 (synchronous) vs Adaptive-v2 (threshold-capped)")

    # Table 4: baseline-211 vs adaptive-v1 (both degraded by PR #960)
    print_comparison(data, "baseline-211", "adaptive-v1",
                     "Adaptive-v1 (queue-depth) vs Baseline 2:1:1")

    # Summary
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print()
    print("  PR #960 context: queue-depth now reads QueueDepth only (GIE parity).")
    print("  adaptive-v2: load-aware scorer (threshold-capped, periodic signal)")
    print("  adaptive-v3: inflight-requests scorer (synchronous, zero staleness)")
    print("  Key question: does synchronous feedback (v3) beat threshold cap (v2)?")
    print()


if __name__ == "__main__":
    main()
