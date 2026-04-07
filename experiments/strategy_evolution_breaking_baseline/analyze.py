#!/usr/bin/env python3
"""Analyze breaking-baseline experiment results.

Reads per-request metrics JSON files from results/ directory,
computes E2E and TTFT statistics per (workload, policy, seed),
then prints comparison tables showing where 2:1:1 breaks.

Usage: python3 analyze.py <results_dir>
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import statistics


def load_metrics(filepath):
    """Load metrics from BLIS JSON output (top-level summary fields)."""
    with open(filepath) as f:
        data = json.load(f)

    # Use top-level summary fields directly
    if "e2e_mean_ms" in data:
        return {
            "e2e_mean_ms": data["e2e_mean_ms"],
            "e2e_p99_ms": data["e2e_p99_ms"],
            "ttft_mean_ms": data.get("ttft_mean_ms", 0),
            "ttft_p99_ms": data.get("ttft_p99_ms", 0),
            "completed": data.get("completed_requests", 0),
        }

    return None


def parse_filename(filename):
    """Parse results filename: {workload}_{policy}_seed{seed}.json"""
    stem = Path(filename).stem
    parts = stem.rsplit("_seed", 1)
    if len(parts) != 2:
        return None, None, None
    seed = int(parts[1])
    prefix = parts[0]
    # Known workloads
    for wl in ["fm6", "fm5", "fm2a", "fm2b", "fm1", "fm3", "fm4"]:
        if prefix.startswith(wl + "_"):
            policy = prefix[len(wl) + 1:]
            return wl, policy, seed
    return None, None, None


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]

    # Collect all results: {workload: {policy: {seed: metrics}}}
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

    # Print tables per workload
    wl_names = {
        "fm1": "FM-1: Prefix Pile-On (Dominant Group)",
        "fm2a": "FM-2a: Groups > Instances (7 groups / 4 inst)",
        "fm2b": "FM-2b: Groups < Instances (2 groups / 4 inst)",
        "fm3": "FM-3: Burst Absorption",
        "fm4": "FM-4: Multi-Regime Phased",
        "fm5": "FM-5: Short Output + Many Groups (Classification)",
        "fm6": "FM-6: Cold Traffic Under KV Pressure",
    }

    baseline_key = "baseline-211"

    for wl in ["fm1", "fm2a", "fm2b", "fm3", "fm4", "fm5", "fm6"]:
        if wl not in data:
            continue

        print(f"\n{'='*80}")
        print(f"  {wl_names.get(wl, wl)}")
        print(f"{'='*80}")

        policies = data[wl]

        # Compute cross-seed averages
        avg = {}
        for policy, seeds in policies.items():
            e2e_means = [m["e2e_mean_ms"] for m in seeds.values()]
            e2e_p99s = [m["e2e_p99_ms"] for m in seeds.values()]
            ttft_means = [m["ttft_mean_ms"] for m in seeds.values()]
            ttft_p99s = [m["ttft_p99_ms"] for m in seeds.values()]

            avg[policy] = {
                "e2e_mean": statistics.mean(e2e_means),
                "e2e_p99": statistics.mean(e2e_p99s),
                "ttft_mean": statistics.mean(ttft_means),
                "ttft_p99": statistics.mean(ttft_p99s),
                "e2e_mean_std": statistics.stdev(e2e_means) if len(e2e_means) > 1 else 0,
                "e2e_p99_std": statistics.stdev(e2e_p99s) if len(e2e_p99s) > 1 else 0,
                "seeds": len(seeds),
            }

        # Print comparison table
        print(f"\n{'Policy':<16} {'E2E Mean':>12} {'E2E P99':>12} {'TTFT Mean':>12} {'TTFT P99':>12} {'Seeds':>6}")
        print("-" * 76)

        for policy in ["baseline-211", "adaptive", "lb-only", "no-kvu", "ppc-heavy", "qd-heavy"]:
            if policy not in avg:
                continue
            a = avg[policy]
            print(f"{policy:<16} {a['e2e_mean']:>9.1f} ms {a['e2e_p99']:>9.1f} ms "
                  f"{a['ttft_mean']:>9.1f} ms {a['ttft_p99']:>9.1f} ms {a['seeds']:>6}")

        # Print improvement vs baseline
        if baseline_key in avg:
            bl = avg[baseline_key]
            print(f"\n  Improvement vs {baseline_key} (positive = alternative is better):")
            print(f"  {'Policy':<16} {'E2E Mean %':>12} {'E2E P99 %':>12} {'TTFT Mean %':>12} {'TTFT P99 %':>12}")
            print(f"  {'-'*68}")

            for policy in ["adaptive", "lb-only", "no-kvu", "ppc-heavy", "qd-heavy"]:
                if policy not in avg:
                    continue
                a = avg[policy]
                e2e_mean_pct = (bl["e2e_mean"] - a["e2e_mean"]) / bl["e2e_mean"] * 100
                e2e_p99_pct = (bl["e2e_p99"] - a["e2e_p99"]) / bl["e2e_p99"] * 100
                ttft_mean_pct = (bl["ttft_mean"] - a["ttft_mean"]) / bl["ttft_mean"] * 100 if bl["ttft_mean"] > 0 else 0
                ttft_p99_pct = (bl["ttft_p99"] - a["ttft_p99"]) / bl["ttft_p99"] * 100 if bl["ttft_p99"] > 0 else 0

                any_metric = max(e2e_mean_pct, e2e_p99_pct, ttft_mean_pct, ttft_p99_pct)
                flag = " *** BONUS" if any_metric >= 25 else \
                       " ***" if any_metric >= 15 else ""
                print(f"  {policy:<16} {e2e_mean_pct:>+11.1f}% {e2e_p99_pct:>+11.1f}% "
                      f"{ttft_mean_pct:>+11.1f}% {ttft_p99_pct:>+11.1f}%{flag}")

        # Per-seed detail for baseline
        if baseline_key in policies:
            print(f"\n  Per-seed detail ({baseline_key}):")
            for seed in sorted(policies[baseline_key].keys()):
                m = policies[baseline_key][seed]
                print(f"    seed={seed}: E2E mean={m['e2e_mean_ms']:.1f}ms "
                      f"P99={m['e2e_p99_ms']:.1f}ms  TTFT mean={m['ttft_mean_ms']:.1f}ms "
                      f"P99={m['ttft_p99_ms']:.1f}ms  completed={m['completed']}")

    # Summary: which FMs showed >= 15% gap?
    print(f"\n{'='*80}")
    print("  SUMMARY: Failure Modes Where 2:1:1 Loses >= 15% (Bonus: >= 25%)")
    print(f"{'='*80}")
    found = False
    for wl in ["fm1", "fm2a", "fm2b", "fm3", "fm4", "fm5", "fm6"]:
        if wl not in data or baseline_key not in data[wl]:
            continue

        bl_seeds = data[wl][baseline_key]
        bl_e2e_mean = statistics.mean([m["e2e_mean_ms"] for m in bl_seeds.values()])
        bl_e2e_p99 = statistics.mean([m["e2e_p99_ms"] for m in bl_seeds.values()])
        bl_ttft_mean = statistics.mean([m["ttft_mean_ms"] for m in bl_seeds.values()])
        bl_ttft_p99 = statistics.mean([m["ttft_p99_ms"] for m in bl_seeds.values()])

        for policy in ["adaptive", "lb-only", "no-kvu", "ppc-heavy", "qd-heavy"]:
            if policy not in data[wl]:
                continue
            alt_seeds = data[wl][policy]
            alt_e2e_mean = statistics.mean([m["e2e_mean_ms"] for m in alt_seeds.values()])
            alt_e2e_p99 = statistics.mean([m["e2e_p99_ms"] for m in alt_seeds.values()])
            alt_ttft_mean = statistics.mean([m["ttft_mean_ms"] for m in alt_seeds.values()])
            alt_ttft_p99 = statistics.mean([m["ttft_p99_ms"] for m in alt_seeds.values()])

            # Positive = alternative is BETTER (lower latency)
            gap_e2e_mean = (bl_e2e_mean - alt_e2e_mean) / bl_e2e_mean * 100
            gap_e2e_p99 = (bl_e2e_p99 - alt_e2e_p99) / bl_e2e_p99 * 100
            gap_ttft_mean = (bl_ttft_mean - alt_ttft_mean) / bl_ttft_mean * 100 if bl_ttft_mean > 0 else 0
            gap_ttft_p99 = (bl_ttft_p99 - alt_ttft_p99) / bl_ttft_p99 * 100 if bl_ttft_p99 > 0 else 0

            # Only flag when alternative actually WINS (positive improvement)
            best_win = max(gap_e2e_mean, gap_e2e_p99, gap_ttft_mean, gap_ttft_p99)
            if best_win >= 15:
                found = True
                bonus = " ** BONUS **" if best_win >= 25 else ""
                wins = []
                if gap_e2e_mean >= 15:
                    wins.append(f"E2E mean {gap_e2e_mean:+.1f}%")
                if gap_e2e_p99 >= 15:
                    wins.append(f"E2E P99 {gap_e2e_p99:+.1f}%")
                if gap_ttft_mean >= 15:
                    wins.append(f"TTFT mean {gap_ttft_mean:+.1f}%")
                if gap_ttft_p99 >= 15:
                    wins.append(f"TTFT P99 {gap_ttft_p99:+.1f}%")
                print(f"  {wl_names.get(wl, wl)}")
                print(f"    {policy} beats baseline: {', '.join(wins)}{bonus}")

    if not found:
        print("  None found. The 2:1:1 baseline held across all failure modes.")
        print("  Consider: more aggressive workload parameters, different instance counts,")
        print("  constrained KV cache (--total-kv-blocks), or higher rates.")


if __name__ == "__main__":
    main()
