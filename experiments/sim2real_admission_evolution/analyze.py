#!/usr/bin/env python3
"""Analyze BLIS SLO attainment results for admission evolution experiment."""
import sys
import re
import os
from collections import defaultdict

def parse_result_file(filepath):
    """Parse a BLIS output file for SLO attainment and key metrics."""
    result = {
        "slo_attainment": {},  # {class: {target_ms: value}}
        "rejected": 0,
        "completed": 0,
        "e2e_by_class": {},    # {class: {mean, p99, n}}
        "ttft_by_class": {},
    }

    current_class = None
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()

            # Rejected requests
            m = re.search(r"Rejected Requests \(Admission\):\s+(\d+)", line)
            if m:
                result["rejected"] = int(m.group(1))

            # Completed requests (cluster aggregate in JSON)
            m = re.search(r'"completed_requests":\s+(\d+)', line)
            if m:
                result["completed"] = int(m.group(1))  # last match = cluster aggregate

            # SLO class header: "  critical:" or "--- SLO Class: critical ---"
            m = re.match(r"\s{2}(\w+):\s*$", line)
            if m and m.group(1) in ("critical", "standard", "sheddable", "batch", "background"):
                current_class = m.group(1)
                continue
            m = re.match(r"\s*--- SLO Class: (\w+) ---", line)
            if m:
                current_class = m.group(1)
                continue

            # SLO attainment: "    SLO_attainment(E2E<3000ms): 0.4942"
            m = re.match(r"\s+SLO_attainment\(E2E<(\d+)ms\):\s+([\d.]+)", line)
            if m and current_class:
                target_ms = int(m.group(1))
                value = float(m.group(2))
                if current_class not in result["slo_attainment"]:
                    result["slo_attainment"][current_class] = {}
                result["slo_attainment"][current_class][target_ms] = value

            # E2E latency: "    E2E:  mean=1234.56 p99=5678.90 (n=100)"
            m = re.match(r"\s+E2E:\s+mean=([\d.]+)\s+p99=([\d.]+)\s+\(n=(\d+)\)", line)
            if m and current_class:
                result["e2e_by_class"][current_class] = {
                    "mean": float(m.group(1)) / 1000,  # us -> ms
                    "p99": float(m.group(2)) / 1000,
                    "n": int(m.group(3)),
                }

            # TTFT latency
            m = re.match(r"\s+TTFT:\s+mean=([\d.]+)\s+p99=([\d.]+)\s+\(n=(\d+)\)", line)
            if m and current_class:
                result["ttft_by_class"][current_class] = {
                    "mean": float(m.group(1)) / 1000,
                    "p99": float(m.group(2)) / 1000,
                    "n": int(m.group(3)),
                }

    return result

def compare_results(baseline_files, treatment_files):
    """Compare baseline vs treatment SLO attainment."""
    # Group by workload
    baseline_by_wl = {}
    treatment_by_wl = {}

    for f in baseline_files:
        bn = os.path.basename(f)
        # Extract workload name: {iter}_{policy}_{workload}.txt
        parts = bn.rsplit("_", 1)  # split off .txt part
        wl = bn.split("gaie-legacy_")[-1].replace(".txt", "")
        baseline_by_wl[wl] = parse_result_file(f)

    for f in treatment_files:
        bn = os.path.basename(f)
        wl = bn.split("adaptive-admission_")[-1].replace(".txt", "")
        treatment_by_wl[wl] = parse_result_file(f)

    print("=" * 80)
    print("SLO ATTAINMENT COMPARISON: gaie-legacy (baseline) vs adaptive-admission (treatment)")
    print("=" * 80)

    for wl in sorted(set(baseline_by_wl.keys()) & set(treatment_by_wl.keys())):
        bl = baseline_by_wl[wl]
        tr = treatment_by_wl[wl]
        print(f"\n{'─' * 60}")
        print(f"Workload: {wl}")
        print(f"  Baseline: {bl['completed']} completed, {bl['rejected']} rejected")
        print(f"  Treatment: {tr['completed']} completed, {tr['rejected']} rejected")

        for cls in ["critical", "standard", "sheddable"]:
            if cls not in bl["slo_attainment"] or cls not in tr["slo_attainment"]:
                continue
            print(f"\n  [{cls.upper()}]")

            # E2E latency comparison
            if cls in bl["e2e_by_class"] and cls in tr["e2e_by_class"]:
                bl_e2e = bl["e2e_by_class"][cls]
                tr_e2e = tr["e2e_by_class"][cls]
                mean_delta = ((tr_e2e["mean"] - bl_e2e["mean"]) / bl_e2e["mean"] * 100) if bl_e2e["mean"] > 0 else 0
                print(f"    E2E mean: {bl_e2e['mean']:.0f}ms -> {tr_e2e['mean']:.0f}ms ({mean_delta:+.1f}%)")
                print(f"    E2E p99:  {bl_e2e['p99']:.0f}ms -> {tr_e2e['p99']:.0f}ms")

            # SLO attainment comparison
            for target_ms in sorted(bl["slo_attainment"][cls].keys()):
                bl_att = bl["slo_attainment"][cls].get(target_ms, 0)
                tr_att = tr["slo_attainment"][cls].get(target_ms, 0)
                abs_delta = tr_att - bl_att
                rel_delta = (abs_delta / bl_att * 100) if bl_att > 0 else float('inf')
                marker = " ***" if abs_delta >= 0.10 else (" **" if abs_delta >= 0.05 else "")
                print(f"    SLO<{target_ms}ms: {bl_att:.4f} -> {tr_att:.4f}  "
                      f"(abs: {abs_delta:+.4f}, rel: {rel_delta:+.1f}%){marker}")

    print(f"\n{'=' * 80}")
    print("Legend: *** = 10%+ absolute gain, ** = 5%+ absolute gain")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze.py <result_files...>")
        print("   or: python3 analyze.py --compare <baseline_iter> <treatment_iter> <results_dir>")
        sys.exit(1)

    if sys.argv[1] == "--compare":
        if len(sys.argv) < 5:
            print("Usage: python3 analyze.py --compare <baseline_iter> <treatment_iter> <results_dir>")
            sys.exit(1)
        bl_iter = sys.argv[2]
        tr_iter = sys.argv[3]
        results_dir = sys.argv[4]

        bl_files = sorted(f for f in [os.path.join(results_dir, fn) for fn in os.listdir(results_dir)]
                          if os.path.basename(f).startswith(f"{bl_iter}_gaie-legacy_"))
        tr_files = sorted(f for f in [os.path.join(results_dir, fn) for fn in os.listdir(results_dir)]
                          if os.path.basename(f).startswith(f"{tr_iter}_adaptive-admission_"))

        if not bl_files:
            print(f"No baseline files found for iter={bl_iter}")
            sys.exit(1)
        if not tr_files:
            print(f"No treatment files found for iter={tr_iter}")
            sys.exit(1)

        compare_results(bl_files, tr_files)
    else:
        # Just parse and dump individual files
        for filepath in sys.argv[1:]:
            result = parse_result_file(filepath)
            print(f"\n{'─' * 40}")
            print(f"File: {os.path.basename(filepath)}")
            print(f"  Completed: {result['completed']}, Rejected: {result['rejected']}")
            for cls in ["critical", "standard", "sheddable", "batch"]:
                if cls in result["slo_attainment"]:
                    print(f"  [{cls}] SLO attainment:")
                    for target_ms, val in sorted(result["slo_attainment"][cls].items()):
                        print(f"    E2E<{target_ms}ms: {val:.4f}")
                if cls in result["e2e_by_class"]:
                    e = result["e2e_by_class"][cls]
                    print(f"    E2E: mean={e['mean']:.0f}ms p99={e['p99']:.0f}ms (n={e['n']})")

if __name__ == "__main__":
    main()
