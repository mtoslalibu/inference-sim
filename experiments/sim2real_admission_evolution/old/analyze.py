#!/usr/bin/env python3
"""Analyze admission control experiment results from BLIS stdout.

Usage:
    python3 analyze.py results/
    python3 analyze.py results/ --compare gaie-legacy adaptive-admission
"""

import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_stdout(filepath):
    """Parse BLIS stdout for key metrics."""
    metrics = {}
    per_slo = {}

    with open(filepath) as f:
        content = f.read()

    # Cluster-level JSON metrics (last JSON block before Anomaly Counters)
    # Find responses_per_sec, e2e_mean_ms etc from the cluster aggregate block
    json_blocks = re.findall(r'\{[^{}]+\}', content, re.DOTALL)
    if json_blocks:
        last_block = json_blocks[-1]
        for pattern, key in [
            (r'"responses_per_sec":\s*([\d.]+)', "throughput_rps"),
            (r'"tokens_per_sec":\s*([\d.]+)', "throughput_tps"),
            (r'"e2e_mean_ms":\s*([\d.]+)', "e2e_mean_ms"),
            (r'"e2e_p99_ms":\s*([\d.]+)', "e2e_p99_ms"),
            (r'"ttft_mean_ms":\s*([\d.]+)', "ttft_mean_ms"),
            (r'"ttft_p99_ms":\s*([\d.]+)', "ttft_p99_ms"),
            (r'"completed_requests":\s*(\d+)', "completed"),
            (r'"injected_requests":\s*(\d+)', "injected"),
        ]:
            m = re.search(pattern, last_block)
            if m:
                val = m.group(1)
                metrics[key] = int(val) if val.isdigit() else float(val)

    # Anomaly counters
    m = re.search(r'Rejected Requests \(Admission\):\s*(\d+)', content)
    if m:
        metrics["rejected"] = int(m.group(1))

    m = re.search(r'Rejected Requests \(Routing\):\s*(\d+)', content)
    if m:
        metrics["routing_rejections"] = int(m.group(1))

    m = re.search(r'Deferred \(horizon-interrupted\):\s*(\d+)', content)
    if m:
        metrics["deferred"] = int(m.group(1))

    # Shed by tier
    shed_matches = re.findall(r'Shed \((\w+)\):\s*(\d+)', content)
    shed_by_tier = {}
    for tier, count in shed_matches:
        shed_by_tier[tier] = int(count)
    if shed_by_tier:
        metrics["shed_by_tier"] = shed_by_tier

    # Jain fairness
    jain_m = re.search(r'Jain Fairness Index:\s+([\d.]+)', content)
    if jain_m:
        metrics["jain_fairness"] = float(jain_m.group(1))

    # Per-SLO metrics (format: "  critical:\n    TTFT: mean=X p99=Y (n=Z)\n    E2E:  mean=X p99=Y (n=Z)")
    slo_pattern = r'  (\w+):\n\s+TTFT:\s+mean=([\d.]+)\s+p99=([\d.]+)\s+\(n=(\d+)\)\n\s+E2E:\s+mean=([\d.]+)\s+p99=([\d.]+)\s+\(n=(\d+)\)'
    for m in re.finditer(slo_pattern, content):
        slo_class = m.group(1)
        per_slo[slo_class] = {
            "ttft_mean_us": float(m.group(2)),
            "ttft_p99_us": float(m.group(3)),
            "ttft_n": int(m.group(4)),
            "e2e_mean_us": float(m.group(5)),
            "e2e_p99_us": float(m.group(6)),
            "e2e_n": int(m.group(7)),
        }

    return {"overall": metrics, "per_slo": per_slo}


def load_results(results_dir):
    """Load all results organized by policy/workload/seed."""
    data = defaultdict(lambda: defaultdict(dict))
    results_path = Path(results_dir)

    for policy_dir in sorted(results_path.iterdir()):
        if not policy_dir.is_dir():
            continue
        policy = policy_dir.name

        for workload_dir in sorted(policy_dir.iterdir()):
            if not workload_dir.is_dir():
                continue
            workload = workload_dir.name

            for seed_dir in sorted(workload_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                stdout_file = seed_dir / "stdout.txt"
                if stdout_file.exists() and stdout_file.stat().st_size > 0:
                    seed = seed_dir.name
                    data[policy][workload][seed] = parse_stdout(str(stdout_file))

    return data


def avg(vals):
    return sum(vals) / len(vals) if vals else 0


def print_summary(data, policies=None):
    """Print comparison tables."""
    if policies is None:
        policies = sorted(data.keys())

    workloads = set()
    for p in policies:
        workloads.update(data[p].keys())
    workloads = sorted(workloads)

    for workload in workloads:
        print(f"\n{'='*90}")
        print(f"  WORKLOAD: {workload}")
        print(f"{'='*90}")

        for policy in policies:
            if workload not in data[policy]:
                continue
            seeds = data[policy][workload]
            print(f"\n  --- {policy} ({len(seeds)} seeds) ---")

            all_metrics = [s["overall"] for s in seeds.values() if s["overall"]]
            if not all_metrics:
                print("    No data")
                continue

            # Overall metrics
            for key in ["completed", "injected", "rejected", "deferred",
                        "e2e_mean_ms", "e2e_p99_ms", "ttft_mean_ms", "ttft_p99_ms",
                        "throughput_rps", "jain_fairness"]:
                vals = [m[key] for m in all_metrics if key in m]
                if vals:
                    a = avg(vals)
                    if isinstance(vals[0], int):
                        print(f"    {key:30s}: {a:>10.0f}  (seeds: {vals})")
                    else:
                        print(f"    {key:30s}: {a:>10.2f}  (seeds: {vals})")

            # Per-SLO summary (averaged across seeds)
            slo_classes = set()
            for s in seeds.values():
                slo_classes.update(s.get("per_slo", {}).keys())

            if slo_classes:
                print(f"\n    Per-SLO (averaged across {len(seeds)} seeds):")
                print(f"    {'Class':12s} {'Completed':>10s} {'E2E_mean_ms':>12s} {'E2E_p99_ms':>12s} {'TTFT_mean_ms':>13s} {'TTFT_p99_ms':>12s}")
                for slo in sorted(slo_classes):
                    e2e_means = []
                    e2e_p99s = []
                    ttft_means = []
                    ttft_p99s = []
                    completeds = []
                    for s in seeds.values():
                        sd = s.get("per_slo", {}).get(slo, {})
                        if sd:
                            e2e_means.append(sd["e2e_mean_us"] / 1000)
                            e2e_p99s.append(sd["e2e_p99_us"] / 1000)
                            ttft_means.append(sd["ttft_mean_us"] / 1000)
                            ttft_p99s.append(sd["ttft_p99_us"] / 1000)
                            completeds.append(sd["e2e_n"])
                    if completeds:
                        print(f"    {slo:12s} {avg(completeds):>10.0f} {avg(e2e_means):>12.1f} {avg(e2e_p99s):>12.1f} {avg(ttft_means):>13.1f} {avg(ttft_p99s):>12.1f}")

            # Shed by tier
            shed = [m.get("shed_by_tier", {}) for m in all_metrics]
            if any(shed):
                tiers = set()
                for s in shed:
                    tiers.update(s.keys())
                shed_str = ", ".join(f"{t}: {avg([s.get(t, 0) for s in shed]):.0f}" for t in sorted(tiers))
                print(f"    Shed by tier (avg): {shed_str}")

        # Delta comparison between two policies
        if len(policies) == 2 and all(workload in data[p] for p in policies):
            p1, p2 = policies[0], policies[1]
            print(f"\n  --- DELTA: {p2} vs {p1} ---")
            m1 = [s["overall"] for s in data[p1][workload].values()]
            m2 = [s["overall"] for s in data[p2][workload].values()]

            for key in ["completed", "rejected", "e2e_mean_ms", "e2e_p99_ms",
                        "ttft_mean_ms", "ttft_p99_ms", "throughput_rps"]:
                v1 = [m[key] for m in m1 if key in m]
                v2 = [m[key] for m in m2 if key in m]
                if v1 and v2:
                    a1, a2 = avg(v1), avg(v2)
                    if a1 > 0:
                        delta_pct = (a2 - a1) / a1 * 100
                        print(f"    {key:30s}: {a1:>10.2f} -> {a2:>10.2f}  ({delta_pct:+.1f}%)")

            # Per-SLO deltas
            slo_classes = set()
            for s in data[p1][workload].values():
                slo_classes.update(s.get("per_slo", {}).keys())

            if slo_classes:
                print(f"\n    Per-SLO deltas (E2E P99 ms):")
                for slo in sorted(slo_classes):
                    v1 = [s["per_slo"][slo]["e2e_p99_us"]/1000 for s in data[p1][workload].values() if slo in s.get("per_slo", {})]
                    v2 = [s["per_slo"][slo]["e2e_p99_us"]/1000 for s in data[p2][workload].values() if slo in s.get("per_slo", {})]
                    if v1 and v2:
                        a1, a2 = avg(v1), avg(v2)
                        delta = (a2 - a1) / a1 * 100 if a1 > 0 else 0
                        print(f"      {slo:12s}: {a1:>10.1f} -> {a2:>10.1f}  ({delta:+.1f}%)")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir> [--compare policy1 policy2]")
        sys.exit(1)

    results_dir = sys.argv[1]
    data = load_results(results_dir)

    policies = None
    if "--compare" in sys.argv:
        idx = sys.argv.index("--compare")
        policies = sys.argv[idx+1:idx+3]

    print_summary(data, policies)


if __name__ == "__main__":
    main()
