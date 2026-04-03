#!/bin/bash
# Strategy Evolution — Final Benchmark
#
# Compares Adaptive-Cache-LB (evolved algorithm) against:
#   1. BLIS 3:2:2 baseline (precise-prefix-cache:3, queue-depth:2, kv-utilization:2)
#   2. GLIA HRA baseline (KV headroom allocation, no cache awareness)
#
# on two workloads designed to expose each baseline's weakness:
#   W1: Prefix-Burst — bursty prefix-heavy traffic with 2 document groups (breaks GLIA)
#   W2: GLIA-Stress  — long prefix sharing with high burstiness (breaks GLIA harder)
#
# Setup: qwen2.5-7b-instruct, H100, tp=1, blackbox, 4 instances
# Signal delays: 5s stale snapshots, 2s precise cache signals
# Seeds: 42, 123, 7
#
# Usage: ./run_benchmark.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

RESULTS="$SCRIPT_DIR/results"
mkdir -p "$RESULTS"

SEEDS=(42 123 7)
COMMON="--model qwen/qwen2.5-7b-instruct --hardware H100 --tp 1 --latency-model blackbox --num-instances 4 --routing-policy weighted --cache-signal-delay 2000000 --snapshot-refresh-interval 5000000"

W1="$SCRIPT_DIR/workload1_prefix_burst.yaml"
W2="$SCRIPT_DIR/workload2_glia_stress.yaml"

echo "============================================"
echo " Strategy Evolution — Final Benchmark"
echo " 3 algorithms × 2 workloads × 3 seeds"
echo "============================================"
echo ""

# --- Phase 1: Adaptive-Cache-LB (evolved algorithm) ---
echo ">>> Building Adaptive-Cache-LB..."
cp sim/routing.go sim/routing.go.backup
cp "$SCRIPT_DIR/iter3_adaptive_cache_lb.go" sim/routing.go
go build -o blis main.go 2>/dev/null

for wl_name in w1 w2; do
    wl_file="$W1"; [ "$wl_name" = "w2" ] && wl_file="$W2"
    echo "  Running Adaptive-CLB on $wl_name..."
    for seed in "${SEEDS[@]}"; do
        ./blis run $COMMON --routing-scorers "precise-prefix-cache:1,queue-depth:1" \
            --workload-spec "$wl_file" --seed "$seed" \
            --metrics-path "$RESULTS/aclb_${wl_name}_s${seed}.json" 2>/dev/null >/dev/null
    done
done
echo "  Done."

# --- Phase 2: BLIS 3:2:2 baseline ---
echo ">>> Building BLIS 3:2:2..."
mv sim/routing.go.backup sim/routing.go
go build -o blis main.go 2>/dev/null

for wl_name in w1 w2; do
    wl_file="$W1"; [ "$wl_name" = "w2" ] && wl_file="$W2"
    echo "  Running 3:2:2 on $wl_name..."
    for seed in "${SEEDS[@]}"; do
        ./blis run $COMMON --routing-scorers "precise-prefix-cache:3,queue-depth:2,kv-utilization:2" \
            --workload-spec "$wl_file" --seed "$seed" \
            --metrics-path "$RESULTS/blis322_${wl_name}_s${seed}.json" 2>/dev/null >/dev/null
    done
done
echo "  Done."

# --- Phase 3: GLIA baseline ---
echo ">>> Building GLIA..."
cp sim/routing.go sim/routing.go.backup
cp baseline_glia.go sim/routing.go
go build -o blis main.go 2>/dev/null

for wl_name in w1 w2; do
    wl_file="$W1"; [ "$wl_name" = "w2" ] && wl_file="$W2"
    echo "  Running GLIA on $wl_name..."
    for seed in "${SEEDS[@]}"; do
        ./blis run $COMMON --routing-scorers "queue-depth:1" \
            --workload-spec "$wl_file" --seed "$seed" \
            --metrics-path "$RESULTS/glia_${wl_name}_s${seed}.json" 2>/dev/null >/dev/null
    done
done

# Restore original
mv sim/routing.go.backup sim/routing.go
go build -o blis main.go 2>/dev/null
echo "  Done."
echo ""

# --- Analysis ---
python3 << 'PYEOF'
import json, statistics, os

RD = os.environ.get("RESULTS", "experiments/strategy-evolution/results")

def load_avg(pattern, seeds=[42, 123, 7]):
    results = {}
    for seed in seeds:
        d = json.load(open(pattern.format(seed=seed)))
        for k, v in d.items():
            if isinstance(v, (int, float)):
                results.setdefault(k, []).append(v)
    return {k: (statistics.mean(v), statistics.stdev(v) if len(v)>1 else 0) for k, v in results.items()}

configs = {
    "Adaptive-CLB": ("aclb_{wl}_s{seed}.json",),
    "BLIS 3:2:2":   ("blis322_{wl}_s{seed}.json",),
    "GLIA":         ("glia_{wl}_s{seed}.json",),
}

metrics = ['e2e_mean_ms', 'e2e_p99_ms', 'ttft_mean_ms', 'ttft_p99_ms', 'tokens_per_sec']
labels = ['E2E Mean', 'E2E P99', 'TTFT Mean', 'TTFT P99', 'TPS']

for wl_name, wl_key in [("W1: Prefix-Burst (bursty, prefix-heavy, 2 groups + cold)", "w1"),
                          ("W2: GLIA-Stress (long prefixes, high burstiness)", "w2")]:
    print(f"\n{'='*85}")
    print(f"  {wl_name}")
    print(f"  4 instances | H100 | qwen2.5-7b | blackbox | 5s stale | 2s cache delay")
    print(f"{'='*85}")
    print(f"{'Config':<16}", end="")
    for l in labels:
        print(f" {l:>10}", end="")
    print()
    print("-"*85)

    ref = {}
    for name, (pat,) in configs.items():
        full = f"{RD}/{pat}".replace("{wl}", wl_key)
        m = load_avg(full)
        ref[name] = m
        print(f"{name:<16}", end="")
        for k in metrics:
            print(f" {m[k][0]:>10.1f}", end="")
        print()

    print()
    aclb = ref["Adaptive-CLB"]
    for baseline in ["BLIS 3:2:2", "GLIA"]:
        bl = ref[baseline]
        print(f"  Adaptive-CLB vs {baseline}:")
        for k, l in zip(metrics[:4], labels[:4]):
            delta = (aclb[k][0] - bl[k][0]) / bl[k][0] * 100
            print(f"    {l:<12}: {delta:+6.1f}%", end="")
            if abs(delta) >= 30: print(" <<<")
            elif abs(delta) >= 15: print(" <<")
            elif abs(delta) >= 5: print(" <")
            else: print()
        print()

print("="*85)
print("  <<< = 30%+ improvement   << = 15%+   < = 5%+")
print("="*85)
PYEOF
