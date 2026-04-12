#!/usr/bin/env python3
"""Generate SLO attainment comparison figures for admission evolution experiments."""

import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = "experiments/sim2real_admission_evolution/figures"

# All data extracted from results_tp/ runs (trained-physics only)
# Format: {model: {workload: {class: {slo_target: (baseline, iter11)}}}}
DATA = {
    "Qwen3-14B": {
        "W1: Sustained 1.5x\n(rate=110)": {
            "critical": {
                "5s": (0.0393, 0.3454),
                "8s": (0.2850, 0.9646),
                "10s": (0.5991, 0.9991),
                "12s": (0.8664, 1.0000),
                "15s": (0.9944, 1.0000),
            },
            "standard": {
                "5s": (0.0423, 0.3486),
                "8s": (0.2985, 0.9592),
                "10s": (0.6066, 1.0000),
                "12s": (0.8665, 1.0000),
                "15s": (0.9903, 1.0000),
            },
        },
        "W2: Burst 1x→2x\n(rate=219)": {
            "critical": {
                "5s": (0.0825, 0.3636),
                "8s": (0.4343, 0.8326),
                "10s": (0.6992, 0.9669),
                "12s": (0.8730, 0.9969),
                "15s": (0.9881, 1.0000),
            },
            "standard": {
                "5s": (0.0850, 0.3603),
                "8s": (0.4357, 0.7878),
                "10s": (0.6823, 0.9564),
                "12s": (0.8701, 0.9963),
                "15s": (0.9835, 1.0000),
            },
        },
        "W3: High Sheddable 2.9x\n(rate=210, 65% shed)": {
            "critical": {
                "5s": (0.0311, 0.2082),
                "8s": (0.2507, 0.8662),
                "10s": (0.5607, 0.9935),
                "12s": (0.8523, 1.0000),
                "15s": (0.9893, 1.0000),
            },
            "standard": {
                "5s": (0.0283, 0.2227),
                "8s": (0.2631, 0.8705),
                "10s": (0.5795, 0.9914),
                "12s": (0.8613, 1.0000),
                "15s": (0.9894, 1.0000),
            },
        },
    },
    "Qwen3-32B": {
        "W1: Sustained 1.8x\n(rate=30)": {
            "critical": {
                "5s": (0.0433, 0.0526),
                "8s": (0.4367, 0.5658),
                "10s": (0.7767, 0.8882),
                "12s": (0.9700, 0.9967),
                "15s": (1.0000, 1.0000),
            },
            "standard": {
                "5s": (0.0448, 0.0675),
                "8s": (0.4819, 0.5823),
                "10s": (0.8380, 0.9072),
                "12s": (0.9744, 0.9895),
                "15s": (1.0000, 1.0000),
            },
        },
        "W2: Burst 1x→2x\n(rate=51)": {
            "critical": {
                "5s": (0.0590, 0.0730),
                "8s": (0.4465, 0.5146),
                "10s": (0.7970, 0.8796),
                "12s": (0.9520, 0.9891),
                "15s": (1.0000, 1.0000),
            },
            "standard": {
                "5s": (0.0804, 0.1188),
                "8s": (0.5025, 0.5916),
                "10s": (0.8241, 0.8985),
                "12s": (0.9623, 0.9851),
                "15s": (0.9975, 1.0000),
            },
        },
        "W3: High Sheddable 4.4x\n(rate=75, 65% shed)": {
            "critical": {
                "5s": (0.0165, 0.0390),
                "8s": (0.0988, 0.4123),
                "10s": (0.2757, 0.7660),
                "12s": (0.5556, 0.9610),
                "15s": (0.7737, 1.0000),
            },
            "standard": {
                "5s": (0.0064, 0.0384),
                "8s": (0.1261, 0.4341),
                "10s": (0.3248, 0.8003),
                "12s": (0.5406, 0.9667),
                "15s": (0.7030, 0.9987),
            },
        },
    },
}

COLORS = {"baseline": "#d62728", "iter11": "#2ca02c"}


def plot_per_workload(model, workloads, filename):
    """One figure per model: 3 workloads x 2 classes = 6 subplots."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    fig.suptitle(f"{model} — SLO Attainment: Iter11 vs GAIE Legacy Baseline\n(trained-physics, 4x H100 TP=1)",
                 fontsize=14, fontweight="bold")

    classes = ["critical", "standard"]
    wk_names = list(workloads.keys())

    for col, wk_name in enumerate(wk_names):
        for row, cls in enumerate(classes):
            ax = axes[row][col]
            targets = list(workloads[wk_name][cls].keys())
            baselines = [workloads[wk_name][cls][t][0] * 100 for t in targets]
            iter11s = [workloads[wk_name][cls][t][1] * 100 for t in targets]

            x = np.arange(len(targets))
            width = 0.35

            bars_b = ax.bar(x - width / 2, baselines, width, label="GAIE Legacy",
                            color=COLORS["baseline"], alpha=0.85, edgecolor="black", linewidth=0.5)
            bars_i = ax.bar(x + width / 2, iter11s, width, label="Iter11 (ours)",
                            color=COLORS["iter11"], alpha=0.85, edgecolor="black", linewidth=0.5)

            # Add gain annotations
            for i, t in enumerate(targets):
                b, it = workloads[wk_name][cls][t]
                gain = (it - b) * 100
                if gain >= 5:
                    ax.annotate(f"+{gain:.0f}pp",
                                xy=(x[i] + width / 2, it * 100),
                                xytext=(0, 5), textcoords="offset points",
                                ha="center", fontsize=7, fontweight="bold", color="#1a5e1a")

            ax.set_xticks(x)
            ax.set_xticklabels([f"<{t}" for t in targets], fontsize=9)
            ax.set_ylim(0, 110)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.grid(axis="y", alpha=0.3)

            if row == 0:
                ax.set_title(wk_name, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{cls.capitalize()} — SLO Attainment (%)", fontsize=10)
            if row == 1:
                ax.set_xlabel("E2E Latency Target", fontsize=9)

            if row == 0 and col == 2:
                ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_gains_summary(filename):
    """Single figure: absolute gain (pp) at SLO<10s for both models, all workloads."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("SLO<10s Attainment Gain (pp) — Iter11 vs GAIE Legacy Baseline\n(trained-physics, 4x H100 TP=1)",
                 fontsize=13, fontweight="bold")

    for idx, model in enumerate(["Qwen3-14B", "Qwen3-32B"]):
        ax = axes[idx]
        workloads = DATA[model]
        wk_short = []
        crit_gains = []
        std_gains = []

        for wk_name, classes in workloads.items():
            short = wk_name.split("\n")[0]  # first line only
            wk_short.append(short)
            b_c, i_c = classes["critical"]["10s"]
            b_s, i_s = classes["standard"]["10s"]
            crit_gains.append((i_c - b_c) * 100)
            std_gains.append((i_s - b_s) * 100)

        x = np.arange(len(wk_short))
        width = 0.35

        ax.bar(x - width / 2, crit_gains, width, label="Critical",
               color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, std_gains, width, label="Standard",
               color="#ff7f0e", alpha=0.85, edgecolor="black", linewidth=0.5)

        # Annotate values
        for i in range(len(wk_short)):
            ax.text(x[i] - width / 2, crit_gains[i] + 1, f"+{crit_gains[i]:.0f}",
                    ha="center", fontsize=9, fontweight="bold")
            ax.text(x[i] + width / 2, std_gains[i] + 1, f"+{std_gains[i]:.0f}",
                    ha="center", fontsize=9, fontweight="bold")

        ax.axhline(y=40, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(len(wk_short) - 0.5, 41, "40pp target", fontsize=8, color="red", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(wk_short, fontsize=9)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_ylabel("Gain (percentage points)" if idx == 0 else "")
        ax.set_ylim(0, 55)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_gains_all_thresholds(filename):
    """Gain at multiple SLO thresholds for both models — shows where gains are biggest."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    fig.suptitle("Absolute SLO Attainment Gain (pp) by Threshold — Iter11 vs GAIE Legacy\n(Critical class, trained-physics, 4x H100 TP=1)",
                 fontsize=13, fontweight="bold")

    for row, model in enumerate(["Qwen3-14B", "Qwen3-32B"]):
        workloads = DATA[model]
        for col, (wk_name, classes) in enumerate(workloads.items()):
            ax = axes[row][col]
            targets = list(classes["critical"].keys())
            gains = [(classes["critical"][t][1] - classes["critical"][t][0]) * 100 for t in targets]

            colors = ["#d62728" if g < 20 else "#ff7f0e" if g < 40 else "#2ca02c" for g in gains]
            bars = ax.bar(range(len(targets)), gains, color=colors, alpha=0.85,
                          edgecolor="black", linewidth=0.5)

            for i, g in enumerate(gains):
                ax.text(i, g + 1, f"+{g:.0f}", ha="center", fontsize=9, fontweight="bold")

            ax.axhline(y=40, color="red", linestyle="--", alpha=0.5, linewidth=1)
            ax.set_xticks(range(len(targets)))
            ax.set_xticklabels([f"<{t}" for t in targets], fontsize=9)
            ax.set_ylim(0, 75)
            ax.grid(axis="y", alpha=0.3)

            if row == 0:
                ax.set_title(wk_name.split("\n")[0], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{model}\nGain (pp)", fontsize=10)
            if row == 1:
                ax.set_xlabel("E2E Latency Target", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    plot_per_workload("Qwen3-14B", DATA["Qwen3-14B"], "14b_slo_attainment.png")
    plot_per_workload("Qwen3-32B", DATA["Qwen3-32B"], "32b_slo_attainment.png")
    plot_gains_summary("gains_slo10s_summary.png")
    plot_gains_all_thresholds("gains_by_threshold.png")

    print("All figures generated.")
