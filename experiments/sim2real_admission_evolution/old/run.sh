#!/usr/bin/env bash
# run.sh — Execute admission control baseline and adaptive experiments
#
# Usage:
#   bash run.sh                    # Run all workloads, all seeds, both policies
#   bash run.sh baseline           # Run only baseline (gaie-legacy)
#   bash run.sh adaptive           # Run only adaptive-admission
#   bash run.sh w1                 # Run only workload 1
#
# Results are written to results/<policy>/<workload>/seed<N>/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BLIS="$REPO_ROOT/blis"
WORKLOAD_DIR="$SCRIPT_DIR/workloads"
RESULTS_DIR="$SCRIPT_DIR/results"

# --- Configuration ---
MODEL="qwen/qwen3-32b"
GPU="H100"
TP=1
LATENCY_MODEL="trained-physics"
NUM_INSTANCES=4
ROUTING="round-robin"
SNAPSHOT_REFRESH=50000  # 50ms — GAIE pod scrape interval
SEEDS=(42 123 456)

POLICIES=("gaie-legacy" "adaptive-admission")
WORKLOADS=("w1_overload_burst" "w2_multitenant_varied" "w3_oscillating")

# --- Parse filter args ---
POLICY_FILTER=""
WORKLOAD_FILTER=""
for arg in "$@"; do
    case "$arg" in
        baseline)   POLICY_FILTER="gaie-legacy" ;;
        adaptive)   POLICY_FILTER="adaptive-admission" ;;
        w1)         WORKLOAD_FILTER="w1_overload_burst" ;;
        w2)         WORKLOAD_FILTER="w2_multitenant_varied" ;;
        w3)         WORKLOAD_FILTER="w3_oscillating" ;;
        *)          echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# --- Build ---
echo "=== Building BLIS ==="
(cd "$REPO_ROOT" && go build -o blis main.go)

# --- Run matrix ---
run_count=0
fail_count=0

for policy in "${POLICIES[@]}"; do
    [[ -n "$POLICY_FILTER" && "$policy" != "$POLICY_FILTER" ]] && continue

    for workload in "${WORKLOADS[@]}"; do
        [[ -n "$WORKLOAD_FILTER" && "$workload" != "$WORKLOAD_FILTER" ]] && continue

        spec="$WORKLOAD_DIR/${workload}.yaml"
        if [[ ! -f "$spec" ]]; then
            echo "WARN: missing $spec, skipping"
            continue
        fi

        for seed in "${SEEDS[@]}"; do
            outdir="$RESULTS_DIR/$policy/$workload/seed${seed}"
            mkdir -p "$outdir"

            echo "--- Running: policy=$policy workload=$workload seed=$seed ---"

            if "$BLIS" run \
                --model "$MODEL" \
                --hardware "$GPU" \
                --tp "$TP" \
                --latency-model "$LATENCY_MODEL" \
                --num-instances "$NUM_INSTANCES" \
                --routing-policy "$ROUTING" \
                --admission-policy "$policy" \
                --snapshot-refresh-interval "$SNAPSHOT_REFRESH" \
                --workload-spec "$spec" \
                --seed "$seed" \
                > "$outdir/stdout.txt" 2> "$outdir/stderr.txt"; then
                echo "  OK -> $outdir/stdout.txt"
                ((run_count++))
            else
                echo "  FAILED (exit $?) -> $outdir/stderr.txt"
                ((fail_count++))
            fi
        done
    done
done

echo ""
echo "=== Done: $run_count succeeded, $fail_count failed ==="
echo "Results in: $RESULTS_DIR/"
