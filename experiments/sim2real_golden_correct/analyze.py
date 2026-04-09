#!/usr/bin/env python3
"""Analyze sim2real_golden benchmark results.

Usage:
    python3 analyze.py <results_dir>              # Tables only
    python3 analyze.py <results_dir> --figures     # Tables + box plot figures
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import statistics


WORKLOAD_NAMES = {
    "workload_fm1_prefix_pileon": "FM-1: Prefix Pile-On",
    "workload_fm2a_groups_gt_instances": "FM-2a: Groups > Instances",
    "workload_fm2b_groups_lt_instances": "FM-2b: Groups < Instances",
    "workload_fm3_burst": "FM-3: Burst Absorption",
    "workload_fm4_multiregime": "FM-4: Multi-Regime Phased",
    "workload_fm5_short_output": "FM-5: Short Output (Classification)",
    "workload_fm6_cold_pressure": "FM-6: Cold Traffic Under KV Pressure",
    "workload_fm7_burst_heavy": "FM-7: Heavy Burst + Long Output",
    "workload_fm8_short_output_highrate": "FM-8: Short Output High Rate",
}

WORKLOAD_ORDER = list(WORKLOAD_NAMES.keys())

POLICY_ORDER = ["baseline-211", "baseline-322", "adaptive", "glia"]
POLICY_LABELS = {
    "baseline-211": "Baseline\n2:1:1",
    "baseline-322": "Baseline\n3:2:2",
    "adaptive": "Adaptive",
    "glia": "Glia\nHRA",
}
POLICY_COLORS = {
    "baseline-211": "#7f8c8d",   # gray
    "baseline-322": "#f39c12",   # orange
    "adaptive": "#e74c3c",       # red
    "glia": "#9b59b6",           # purple
}

METRIC_KEYS = [
    ("E2E Mean (ms)", "e2e_mean_ms"),
    ("E2E P99 (ms)", "e2e_p99_ms"),
    ("TTFT Mean (ms)", "ttft_mean_ms"),
    ("TTFT P99 (ms)", "ttft_p99_ms"),
]


def load_metrics(filepath):
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
    """Parse: workload_fm1_prefix_pileon_baseline-211_seed42.json"""
    stem = Path(filename).stem
    parts = stem.rsplit("_seed", 1)
    if len(parts) != 2:
        return None, None, None
    seed = int(parts[1])
    prefix = parts[0]

    # Try each known workload prefix (longest first to avoid partial matches)
    for wl in sorted(WORKLOAD_ORDER, key=len, reverse=True):
        if prefix.startswith(wl + "_"):
            policy = prefix[len(wl) + 1:]
            return wl, policy, seed
    return None, None, None


def load_all_data(results_dir):
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
    return data


def avg_metrics(seed_metrics):
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
    if baseline == 0:
        return 0.0
    return (baseline - other) / baseline * 100


def print_comparison(data, label_a, label_b, title):
    print("=" * 130)
    print(f"  {title}")
    print(f"  (positive % = {label_b} is better / lower latency)")
    print("=" * 130)
    print()

    hdr = (f"{'Workload':<40} "
           f"{'E2E Mean':>10} {'':>8} "
           f"{'E2E P99':>10} {'':>8} "
           f"{'TTFT Mean':>10} {'':>8} "
           f"{'TTFT P99':>10} {'':>8}")
    sub = (f"{'':40} "
           f"{'A / B':>10} {'chg':>8} "
           f"{'A / B':>10} {'chg':>8} "
           f"{'A / B':>10} {'chg':>8} "
           f"{'A / B':>10} {'chg':>8}")
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

        line = (f"{name:<40} "
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


def generate_figures(data, results_dir):
    """Generate per-workload box plot figures with % gain annotations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  WARNING: matplotlib not installed, skipping figures.")
        print("  Install with: pip install matplotlib")
        return

    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Find which policies have data
    all_policies = set()
    for wl_data in data.values():
        all_policies.update(wl_data.keys())
    policies = [p for p in POLICY_ORDER if p in all_policies]

    for wl in WORKLOAD_ORDER:
        if wl not in data:
            continue

        wl_name = WORKLOAD_NAMES.get(wl, wl)
        wl_data = data[wl]

        # Skip if baseline missing
        if "baseline-211" not in wl_data:
            continue

        # Collect per-seed values for each policy and metric
        policy_values = {}  # policy -> metric_key -> [values across seeds]
        for pol in policies:
            if pol not in wl_data:
                continue
            policy_values[pol] = {}
            for _, mkey in METRIC_KEYS:
                policy_values[pol][mkey] = [
                    m[mkey] for m in wl_data[pol].values()
                ]

        present_policies = [p for p in policies if p in policy_values]
        if not present_policies:
            continue

        # Compute baseline means for % gain calculation
        baseline_means = {}
        for _, mkey in METRIC_KEYS:
            vals = policy_values.get("baseline-211", {}).get(mkey, [])
            baseline_means[mkey] = statistics.mean(vals) if vals else 0

        # Create 2x2 figure: E2E Mean, E2E P99, TTFT Mean, TTFT P99
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(wl_name, fontsize=16, fontweight="bold", y=0.98)

        for ax_idx, (metric_label, mkey) in enumerate(METRIC_KEYS):
            ax = axes[ax_idx // 2][ax_idx % 2]

            # Gather box plot data
            box_data = []
            box_labels = []
            box_colors = []
            for pol in present_policies:
                vals = policy_values[pol].get(mkey, [])
                if vals:
                    box_data.append(vals)
                    box_labels.append(POLICY_LABELS.get(pol, pol))
                    box_colors.append(POLICY_COLORS.get(pol, "#95a5a6"))

            if not box_data:
                ax.set_visible(False)
                continue

            positions = list(range(1, len(box_data) + 1))
            bp = ax.boxplot(
                box_data,
                positions=positions,
                patch_artist=True,
                widths=0.6,
                showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="white",
                               markeredgecolor="black", markersize=6),
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
            )

            # Color the boxes
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Plot individual seed points
            for i, (vals, pol) in enumerate(zip(box_data, present_policies)):
                jitter = [positions[i] + (j - 1) * 0.08 for j in range(len(vals))]
                ax.scatter(jitter, vals, color="black", s=20, zorder=5, alpha=0.6)

            # Annotate % gain vs baseline on non-baseline boxes
            bl_mean = baseline_means[mkey]
            for i, pol in enumerate(present_policies):
                if pol == "baseline-211" or bl_mean == 0:
                    continue
                vals = policy_values[pol].get(mkey, [])
                if not vals:
                    continue
                pol_mean = statistics.mean(vals)
                gain = (bl_mean - pol_mean) / bl_mean * 100

                # Position annotation above the box
                y_max = max(vals)
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0] if ax.get_ylim()[1] != ax.get_ylim()[0] else 1

                # Color: green for positive (better), red for negative (worse)
                if gain >= 15:
                    color = "#27ae60"
                    weight = "bold"
                elif gain > 0:
                    color = "#27ae60"
                    weight = "normal"
                else:
                    color = "#c0392b"
                    weight = "bold"

                ax.annotate(
                    f"{gain:+.1f}%",
                    xy=(positions[i], y_max),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9, fontweight=weight, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=color, alpha=0.8),
                )

            ax.set_title(metric_label, fontsize=12, fontweight="bold")
            ax.set_ylabel("Latency (ms)")
            ax.set_xticks(positions)
            ax.set_xticklabels(box_labels, fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            ax.set_xlim(0.3, len(present_policies) + 0.7)

            # Adjust y-axis to leave room for annotations
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.2)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure
        wl_short = wl.replace("workload_", "")
        fig_path = os.path.join(figures_dir, f"{wl_short}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fig_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze.py <results_dir> [--figures]")
        sys.exit(1)

    results_dir = sys.argv[1]
    do_figures = "--figures" in sys.argv

    data = load_all_data(results_dir)

    if not data:
        print("No results found!")
        sys.exit(1)

    # Discover policies
    all_policies = set()
    for wl_data in data.values():
        all_policies.update(wl_data.keys())
    print(f"  Policies found: {sorted(all_policies)}")
    print()

    # Table 1: Adaptive vs Baseline 2:1:1
    if "adaptive" in all_policies:
        print_comparison(data, "baseline-211", "adaptive",
                         "Adaptive vs Baseline 2:1:1")

    # Table 2: Adaptive vs Baseline 3:2:2
    if "adaptive" in all_policies and "baseline-322" in all_policies:
        print_comparison(data, "baseline-322", "adaptive",
                         "Adaptive vs Baseline 3:2:2")

    # Table 3: Glia vs Baseline 2:1:1
    if "glia" in all_policies:
        print_comparison(data, "baseline-211", "glia",
                         "Glia HRA vs Baseline 2:1:1")

    # Table 4: Adaptive vs Glia
    if "adaptive" in all_policies and "glia" in all_policies:
        print_comparison(data, "glia", "adaptive",
                         "Adaptive vs Glia HRA")

    # Table 5: Baseline 3:2:2 vs 2:1:1
    if "baseline-322" in all_policies:
        print_comparison(data, "baseline-211", "baseline-322",
                         "Baseline 3:2:2 vs Baseline 2:1:1 (weight tuning help?)")

    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print()
    print(f"  Policies compared: {sorted(all_policies)}")
    print("  baseline-211: ppc:2, queue-depth:1, kv-utilization:1 (llm-d default)")
    print("  baseline-322: ppc:3, queue-depth:2, kv-utilization:2 (heavier weights)")
    print("  adaptive: 5-scorer regime-detection (ppc, la, ar, rr, kvu)")
    print("  glia: KV headroom-aware routing (no scorer pipeline)")
    print()

    # Generate figures if requested
    if do_figures:
        print("=" * 80)
        print("  GENERATING FIGURES")
        print("=" * 80)
        print()
        generate_figures(data, results_dir)
        print()


if __name__ == "__main__":
    main()
