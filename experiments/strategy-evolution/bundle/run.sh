#!/bin/bash
# Strategy Evolution — Reproducible Benchmark Bundle
#
# Compares two routing algorithms on two workloads:
#   1. Adaptive-CLB (oracle) — two-stage: cache triage -> load balance
#      Swaps adaptive_cache_lb.go into sim/routing.go (custom Route() method)
#      Policy: policy_oracle.yaml (precise-prefix-cache:2, queue-depth:1)
#
#   2. Baseline 2:1:1 — default weighted scorer pipeline (llm-d production config)
#      Uses unmodified sim/routing.go (standard scorer pipeline Route() method)
#      Policy: policy_baseline.yaml (prefix-affinity:2, kv-utilization:1, queue-depth:1)
#
# All three routing .go files have identical struct definitions (including cacheFn).
# Each Route() method uses only its corresponding signals:
#   - Baseline: scorer pipeline with prefix-affinity + kv-utilization + queue-depth
#   - Oracle:   ws.cacheFn (precise cache queries) + EffectiveLoad() directly
#
# Workloads:
#   W1: Prefix-Burst — 80% prefix-heavy (2 doc groups, gamma cv=2), 30% cold, rate=100, 6000 reqs
#   W2: GLIA-Stress  — 80% prefix-heavy (long prefixes, gamma cv=3), 20% background, rate=80, 5000 reqs
#
# Setup: qwen2.5-7b-instruct, H100, tp=1, blackbox latency, 4 instances
# Signal delays: 5s stale snapshots, 2s cache delay
# Seeds: 42, 123, 7 (averaged)
#
# Usage: cd <repo-root> && bash experiments/strategy-evolution/bundle/run.sh

set -euo pipefail
BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$BUNDLE_DIR/../../.."
cd "$REPO_ROOT"

RESULTS="$BUNDLE_DIR/results"
mkdir -p "$RESULTS"

SEEDS=(42 123 7)
COMMON="--model qwen/qwen2.5-7b-instruct --hardware H100 --tp 1 --latency-model blackbox --num-instances 4 --cache-signal-delay 2000000 --snapshot-refresh-interval 5000000"

W1="$BUNDLE_DIR/workload1_prefix_burst.yaml"
W2="$BUNDLE_DIR/workload2_glia_stress.yaml"

echo "============================================"
echo " Strategy Evolution — Reproducible Bundle"
echo " 2 algorithms x 2 workloads x 3 seeds"
echo "============================================"
echo ""

# --- Phase 1: Adaptive-CLB (oracle) ---
# Swaps adaptive_cache_lb.go into sim/routing.go to replace Route() method.
# Uses policy_oracle.yaml: precise-prefix-cache:2, queue-depth:1
# Route() bypasses scorer pipeline; uses cacheFn + EffectiveLoad() directly.
echo ">>> Building Adaptive-CLB (oracle)..."
cp sim/routing.go sim/routing.go.backup
cp "$BUNDLE_DIR/adaptive_cache_lb.go" sim/routing.go
go build -o blis main.go 2>/dev/null

for wl_name in w1 w2; do
    wl_file="$W1"; [ "$wl_name" = "w2" ] && wl_file="$W2"
    echo "  Running Adaptive-CLB on $wl_name..."
    for seed in "${SEEDS[@]}"; do
        ./blis run $COMMON \
            --policy-config "$BUNDLE_DIR/policy_oracle.yaml" \
            --workload-spec "$wl_file" --seed "$seed" \
            --metrics-path "$RESULTS/oracle_${wl_name}_s${seed}.json" 2>/dev/null >/dev/null
    done
done
echo "  Done."

# --- Phase 2: Baseline 2:1:1 (llm-d production config) ---
# Restores unmodified sim/routing.go (standard scorer pipeline Route() method).
# Uses policy_baseline.yaml: prefix-affinity:2, kv-utilization:1, queue-depth:1
# Route() invokes scorer pipeline: Σ clamp(s_i) × w_i, then argmax.
echo ">>> Building Baseline 2:1:1..."
mv sim/routing.go.backup sim/routing.go
go build -o blis main.go 2>/dev/null

for wl_name in w1 w2; do
    wl_file="$W1"; [ "$wl_name" = "w2" ] && wl_file="$W2"
    echo "  Running Baseline 2:1:1 on $wl_name..."
    for seed in "${SEEDS[@]}"; do
        ./blis run $COMMON \
            --policy-config "$BUNDLE_DIR/policy_baseline.yaml" \
            --workload-spec "$wl_file" --seed "$seed" \
            --metrics-path "$RESULTS/baseline_${wl_name}_s${seed}.json" 2>/dev/null >/dev/null
    done
done
echo "  Done."
echo ""

# --- Analysis ---
RESULTS="$RESULTS" python3 << 'PYEOF'
import json, statistics, os

RD = os.environ["RESULTS"]

def load_avg(pattern, seeds=[42, 123, 7]):
    results = {}
    for seed in seeds:
        path = pattern.format(seed=seed)
        d = json.load(open(path))
        for k, v in d.items():
            if isinstance(v, (int, float)):
                results.setdefault(k, []).append(v)
    return {k: (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0) for k, v in results.items()}

configs = {
    "Adaptive-CLB": "oracle_{wl}_s{seed}.json",
    "Baseline 2:1:1": "baseline_{wl}_s{seed}.json",
}

metrics = ['e2e_mean_ms', 'e2e_p99_ms', 'ttft_mean_ms', 'ttft_p99_ms', 'tokens_per_sec']
labels  = ['E2E Mean', 'E2E P99', 'TTFT Mean', 'TTFT P99', 'TPS']

for wl_name, wl_key in [
    ("W1: Prefix-Burst (bursty, 2 prefix groups + cold, rate=100, 6000 reqs)", "w1"),
    ("W2: GLIA-Stress (long prefixes, high cv=3, rate=80, 5000 reqs)", "w2"),
]:
    print(f"\n{'='*85}")
    print(f"  {wl_name}")
    print(f"  4 instances | H100 | qwen2.5-7b | blackbox | 5s snapshot | 2s cache delay")
    print(f"{'='*85}")
    print(f"{'Config':<18}", end="")
    for l in labels:
        print(f" {l:>10}", end="")
    print()
    print("-" * 85)

    ref = {}
    for name, pat in configs.items():
        full = f"{RD}/{pat}".replace("{wl}", wl_key)
        m = load_avg(full)
        ref[name] = m
        print(f"{name:<18}", end="")
        for k in metrics:
            print(f" {m[k][0]:>10.1f}", end="")
        print()

    print()
    oracle = ref["Adaptive-CLB"]
    bl = ref["Baseline 2:1:1"]
    print(f"  Adaptive-CLB vs Baseline 2:1:1:")
    for k, l in zip(metrics[:4], labels[:4]):
        delta = (oracle[k][0] - bl[k][0]) / bl[k][0] * 100
        flag = " <<<" if abs(delta) >= 30 else " <<" if abs(delta) >= 15 else " <" if abs(delta) >= 5 else ""
        print(f"    {l:<12}: {delta:+6.1f}%{flag}")
    print()

print()
print("=" * 85)
print("  <<< = 30%+ improvement   << = 15%+   < = 5%+")
print("=" * 85)
PYEOF
