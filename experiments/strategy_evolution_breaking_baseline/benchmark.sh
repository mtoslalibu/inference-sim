#!/usr/bin/env bash
# Breaking Baseline 2:1:1: Diagnostic Benchmark
#
# Runs 5 policies across 5 workloads (4 failure modes, FM-2 has 2 variants),
# 3 seeds each. Total: 5 policies x 5 workloads x 3 seeds = 75 runs.
#
# Usage: cd <repo-root> && bash experiments/strategy_evolution_breaking_baseline/benchmark.sh
#
# Optional: pass a single FM to run:
#   bash benchmark.sh fm1
#   bash benchmark.sh fm2a
#   bash benchmark.sh fm3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

# Model and cluster config
MODEL="Qwen/Qwen3-32B"
HARDWARE="H100"
TP=1
LATENCY_MODEL="trained-physics"
CACHE_SIGNAL_DELAY=2000000      # 2s
SNAPSHOT_REFRESH=5000000         # 5s

SEEDS="42 123 456"

# Policies: "name:yaml_file" pairs
POLICIES="
baseline-211:policy_baseline_211.yaml
adaptive:policy_adaptive.yaml
lb-only:policy_lb_only.yaml
no-kvu:policy_no_kvu.yaml
ppc-heavy:policy_ppc_heavy.yaml
qd-heavy:policy_qd_heavy.yaml
"

# Workloads: "name:yaml_file:num_instances" triples
WORKLOADS="
fm1:workload_fm1_prefix_pileon.yaml:4
fm2a:workload_fm2a_groups_gt_instances.yaml:4
fm2b:workload_fm2b_groups_lt_instances.yaml:4
fm3:workload_fm3_burst.yaml:4
fm4:workload_fm4_multiregime.yaml:4
fm5:workload_fm5_short_output.yaml:4
fm6:workload_fm6_cold_pressure.yaml:4
"

# Filter to single FM if argument provided
FILTER="${1:-all}"

echo "============================================================"
echo " Breaking Baseline 2:1:1: Diagnostic Experiment"
echo "============================================================"
echo ""
echo "Model:       $MODEL"
echo "Hardware:    $HARDWARE"
echo "Latency:     $LATENCY_MODEL"
echo "Cache delay: ${CACHE_SIGNAL_DELAY}us (2s)"
echo "Snapshot:    ${SNAPSHOT_REFRESH}us (5s)"
echo "Seeds:       $SEEDS"
echo "Filter:      $FILTER"
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

run_blis() {
    local label="$1"
    local seed="$2"
    local policy_config="$3"
    local workload="$4"
    local num_instances="$5"
    local metrics_out="$6"

    echo -n "  $label seed=$seed instances=$num_instances ... "

    cd "$REPO_ROOT"
    if ./blis run \
        --model "$MODEL" \
        --hardware "$HARDWARE" \
        --tp "$TP" \
        --latency-model "$LATENCY_MODEL" \
        --num-instances "$num_instances" \
        --workload-spec "$workload" \
        --policy-config "$policy_config" \
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
        return 1
    fi
}

# Run all combinations
for wl_entry in $WORKLOADS; do
    [ -z "$wl_entry" ] && continue
    wl_name="${wl_entry%%:*}"
    rest="${wl_entry#*:}"
    wl_yaml="${rest%%:*}"
    num_instances="${rest#*:}"

    # Filter check
    if [ "$FILTER" != "all" ] && [ "$FILTER" != "$wl_name" ]; then
        continue
    fi

    wl_path="$SCRIPT_DIR/$wl_yaml"
    echo "=== Workload: $wl_name ($num_instances instances) ==="

    for pol_entry in $POLICIES; do
        [ -z "$pol_entry" ] && continue
        policy_name="${pol_entry%%:*}"
        policy_yaml="${pol_entry#*:}"
        policy_path="$SCRIPT_DIR/$policy_yaml"

        echo "  Policy: $policy_name"
        for seed in $SEEDS; do
            metrics_out="$RESULTS_DIR/${wl_name}_${policy_name}_seed${seed}.json"
            run_blis "${wl_name}/${policy_name}" "$seed" "$policy_path" "$wl_path" "$num_instances" "$metrics_out" || true
        done
    done
    echo ""
done

# Analyze
echo "--- Analysis ---"
python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR"

echo ""
echo "============================================================"
echo " Benchmark complete: $run_count succeeded, $fail_count failed"
echo " Results in: $RESULTS_DIR/"
echo "============================================================"
