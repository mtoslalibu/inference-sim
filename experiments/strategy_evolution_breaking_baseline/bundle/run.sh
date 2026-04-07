#!/usr/bin/env bash
# Adaptive vs Baseline 2:1:1: Self-Contained Benchmark
#
# Runs 2 policies (baseline-211, adaptive) across 7 workloads, 3 seeds each.
# Total: 2 x 7 x 3 = 42 runs.
#
# Usage:
#   cd experiments/strategy_evolution_breaking_baseline/bundle
#   bash run.sh
#
# Or from repo root:
#   bash experiments/strategy_evolution_breaking_baseline/bundle/run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

# Model and cluster config
MODEL="Qwen/Qwen3-32B"
HARDWARE="H100"
TP=1
LATENCY_MODEL="trained-physics"
NUM_INSTANCES=4
CACHE_SIGNAL_DELAY=2000000      # 2s
SNAPSHOT_REFRESH=5000000         # 5s

SEEDS="42 123 456"

# Two policies only
POLICIES="
baseline-211:policy_baseline_211.yaml
adaptive:policy_adaptive.yaml
"

# All 7 workloads
WORKLOADS="
fm1:workload_fm1_prefix_pileon.yaml
fm2a:workload_fm2a_groups_gt_instances.yaml
fm2b:workload_fm2b_groups_lt_instances.yaml
fm3:workload_fm3_burst.yaml
fm4:workload_fm4_multiregime.yaml
fm5:workload_fm5_short_output.yaml
fm6:workload_fm6_cold_pressure.yaml
"

echo "============================================================"
echo " Adaptive vs Baseline 2:1:1 Benchmark"
echo "============================================================"
echo ""
echo "Model:       $MODEL"
echo "Hardware:    $HARDWARE (TP=$TP)"
echo "Latency:     $LATENCY_MODEL"
echo "Instances:   $NUM_INSTANCES"
echo "Cache delay: ${CACHE_SIGNAL_DELAY}us (2s)"
echo "Snapshot:    ${SNAPSHOT_REFRESH}us (5s)"
echo "Seeds:       $SEEDS"
echo "Runs:        42 (2 policies x 7 workloads x 3 seeds)"
echo ""

# Build
echo "--- Building BLIS ---"
cd "$REPO_ROOT"
go build -o blis main.go
echo "Build complete."
echo ""

# Clean results
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

run_count=0
fail_count=0

for wl_entry in $WORKLOADS; do
    [ -z "$wl_entry" ] && continue
    wl_name="${wl_entry%%:*}"
    wl_yaml="${wl_entry#*:}"
    wl_path="$SCRIPT_DIR/$wl_yaml"

    echo "=== Workload: $wl_name ==="

    for pol_entry in $POLICIES; do
        [ -z "$pol_entry" ] && continue
        policy_name="${pol_entry%%:*}"
        policy_yaml="${pol_entry#*:}"
        policy_path="$SCRIPT_DIR/$policy_yaml"

        for seed in $SEEDS; do
            metrics_out="$RESULTS_DIR/${wl_name}_${policy_name}_seed${seed}.json"
            echo -n "  ${wl_name}/${policy_name} seed=$seed ... "

            cd "$REPO_ROOT"
            if ./blis run \
                --model "$MODEL" \
                --hardware "$HARDWARE" \
                --tp "$TP" \
                --latency-model "$LATENCY_MODEL" \
                --num-instances "$NUM_INSTANCES" \
                --workload-spec "$wl_path" \
                --policy-config "$policy_path" \
                --seed "$seed" \
                --cache-signal-delay "$CACHE_SIGNAL_DELAY" \
                --snapshot-refresh-interval "$SNAPSHOT_REFRESH" \
                --metrics-path "$metrics_out" \
                --log warn \
                > "${metrics_out%.json}.stdout" 2>&1; then
                echo "done"
                run_count=$((run_count + 1))
            else
                echo "FAILED (see ${metrics_out%.json}.stdout)"
                fail_count=$((fail_count + 1))
            fi
        done
    done
    echo ""
done

# Analyze
echo "--- Results ---"
echo ""
python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR"

echo ""
echo "============================================================"
echo " Benchmark complete: $run_count succeeded, $fail_count failed"
echo " Results in: $RESULTS_DIR/"
echo "============================================================"
