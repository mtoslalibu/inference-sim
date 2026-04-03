#!/bin/bash
# Strategy Evolution Benchmark Script
# Runs all baselines and candidate algorithms across both workloads and 3 seeds.
#
# Usage: ./benchmark.sh [routing-go-file]
#   If routing-go-file is provided, it replaces sim/routing.go before building.
#   Otherwise uses current sim/routing.go.

set -euo pipefail
cd "$(dirname "$0")/../.."

RESULTS_DIR="experiments/strategy-evolution/results"
mkdir -p "$RESULTS_DIR"

SEEDS=(42 123 7)
COMMON_FLAGS="--model qwen/qwen2.5-7b-instruct --hardware H100 --tp 1 --latency-model blackbox --num-instances 2 --routing-policy weighted --cache-signal-delay 2000000 --snapshot-refresh-interval 5000000"

# Workload 1: Prefix-heavy burst (rate=60, 4000 reqs, prefix=512)
W1_FLAGS="--workload-spec experiments/strategy-evolution/workload1_prefix_heavy.yaml"

# Workload 2: Load stress (rate=80, 5000 reqs, variable sizes, no prefix)
W2_FLAGS="--workload-spec experiments/strategy-evolution/workload2_load_stress.yaml"

extract_metrics() {
    local json_file="$1"
    python3 -c "
import json, sys
d = json.load(open('$json_file'))
print(f'e2e_mean={d[\"e2e_mean_ms\"]:.1f} e2e_p99={d[\"e2e_p99_ms\"]:.1f} ttft_mean={d[\"ttft_mean_ms\"]:.1f} ttft_p99={d[\"ttft_p99_ms\"]:.1f} tps={d[\"tokens_per_sec\"]:.0f}')
"
}

run_config() {
    local name="$1"
    local scorers="$2"
    local workload_name="$3"
    local workload_flags="$4"

    echo "=== $name | $workload_name ==="
    for seed in "${SEEDS[@]}"; do
        local out="$RESULTS_DIR/${name}_${workload_name}_s${seed}.json"
        ./blis run $COMMON_FLAGS --routing-scorers "$scorers" $workload_flags --seed "$seed" --metrics-path "$out" 2>/dev/null > /dev/null
        echo "  seed=$seed: $(extract_metrics "$out")"
    done

    # Compute mean across seeds
    python3 << PYEOF
import json, statistics
files = ['$RESULTS_DIR/${name}_${workload_name}_s42.json', '$RESULTS_DIR/${name}_${workload_name}_s123.json', '$RESULTS_DIR/${name}_${workload_name}_s7.json']
data = [json.load(open(f)) for f in files]
for metric in ['e2e_mean_ms', 'e2e_p99_ms', 'ttft_mean_ms', 'ttft_p99_ms', 'tokens_per_sec']:
    vals = [d[metric] for d in data]
    print(f'  AVG {metric}: {statistics.mean(vals):.1f} (std={statistics.stdev(vals):.1f})')
PYEOF
    echo ""
}

run_glia() {
    local workload_name="$1"
    local workload_flags="$2"

    echo "=== GLIA | $workload_name ==="

    # Swap in GLIA routing
    cp sim/routing.go sim/routing.go.bak
    cp baseline_glia.go sim/routing.go
    go build -o blis main.go 2>/dev/null

    for seed in "${SEEDS[@]}"; do
        local out="$RESULTS_DIR/glia_${workload_name}_s${seed}.json"
        # GLIA ignores --routing-scorers but we pass dummy to satisfy weighted policy
        ./blis run $COMMON_FLAGS --routing-scorers "queue-depth:1" $workload_flags --seed "$seed" --metrics-path "$out" 2>/dev/null > /dev/null
        echo "  seed=$seed: $(extract_metrics "$out")"
    done

    # Restore original routing
    mv sim/routing.go.bak sim/routing.go
    go build -o blis main.go 2>/dev/null

    python3 << PYEOF
import json, statistics
files = ['$RESULTS_DIR/glia_${workload_name}_s42.json', '$RESULTS_DIR/glia_${workload_name}_s123.json', '$RESULTS_DIR/glia_${workload_name}_s7.json']
data = [json.load(open(f)) for f in files]
for metric in ['e2e_mean_ms', 'e2e_p99_ms', 'ttft_mean_ms', 'ttft_p99_ms', 'tokens_per_sec']:
    vals = [d[metric] for d in data]
    print(f'  AVG {metric}: {statistics.mean(vals):.1f} (std={statistics.stdev(vals):.1f})')
PYEOF
    echo ""
}

echo "============================================"
echo "Strategy Evolution Benchmark"
echo "============================================"
echo ""

# Run BLIS 3:2:2 baseline
run_config "baseline_322" "precise-prefix-cache:3,queue-depth:2,kv-utilization:2" "w1_prefix" "$W1_FLAGS"
run_config "baseline_322" "precise-prefix-cache:3,queue-depth:2,kv-utilization:2" "w2_stress" "$W2_FLAGS"

# Run GLIA baseline
run_glia "w1_prefix" "$W1_FLAGS"
run_glia "w2_stress" "$W2_FLAGS"

echo "============================================"
echo "Benchmark Complete"
echo "============================================"
