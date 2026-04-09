#!/usr/bin/env bash
# sim2real_golden_correct — Strategy Evolution with Correct 50ms Scrape Interval
#
# For each router: swap routing.go → build BLIS → run all workloads → restore.
# Baseline uses stock BLIS (no swap).
#
# Usage:
#   cd experiments/sim2real_golden_correct
#   bash compare.sh           # Run core workloads
#   bash compare.sh --all     # Run all 7 workloads

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
ROUTERS_DIR="$SCRIPT_DIR/routers"
WORKLOADS_DIR="$SCRIPT_DIR/workloads"

ORIGINAL_ROUTING="$REPO_ROOT/sim/routing.go"
BACKUP_ROUTING="$SCRIPT_DIR/.routing.go.backup"

MODEL="Qwen/Qwen3-32B"
HARDWARE="H100"
TP=1
LATENCY_MODEL="trained-physics"
NUM_INSTANCES=4
CACHE_SIGNAL_DELAY=2000000      # 2s — matches llm-d defaultSpeculativeTTL
SNAPSHOT_REFRESH=50000           # 50ms — GAIE RefreshMetricsInterval default
TIMEOUT_SECS=120

SEEDS="42 123 456"

# Router entries: label:router_file:policy_yaml
ROUTERS="
baseline-211:STOCK:policy_baseline_211.yaml
baseline-322:STOCK:policy_baseline_322.yaml
adaptive-correct:router_adaptive_expanded.go:policy_adaptive_expanded.yaml
"

CORE_WORKLOADS="
workload_fm3_burst
workload_fm5_short_output
workload_fm8_short_output_highrate
"

EXTENDED_WORKLOADS="
"

# Parse --all flag
RUN_ALL=false
if [ "${1:-}" = "--all" ]; then
    RUN_ALL=true
fi

if [ "$RUN_ALL" = "true" ]; then
    WORKLOADS="$CORE_WORKLOADS $EXTENDED_WORKLOADS"
    wl_mode="all 7 workloads"
else
    WORKLOADS="$CORE_WORKLOADS"
    wl_mode="5 core workloads (use --all for all 7)"
fi

# Count workloads and compute total runs
wl_count=0
for w in $WORKLOADS; do [ -n "$w" ] && wl_count=$((wl_count + 1)); done
router_count=0
for r in $ROUTERS; do [ -n "$r" ] && router_count=$((router_count + 1)); done
seed_count=0
for s in $SEEDS; do seed_count=$((seed_count + 1)); done
total_runs=$((router_count * wl_count * seed_count))

echo "============================================================"
echo " sim2real_golden_correct — Strategy Evolution (50ms refresh)"
echo "============================================================"
echo ""
echo "Model:       $MODEL"
echo "Hardware:    $HARDWARE (TP=$TP)"
echo "Latency:     $LATENCY_MODEL"
echo "Instances:   $NUM_INSTANCES"
echo "Cache delay: ${CACHE_SIGNAL_DELAY}us (2s)"
echo "Snapshot:    ${SNAPSHOT_REFRESH}us (50ms)"
echo "Timeout:     ${TIMEOUT_SECS}s per run"
echo "Seeds:       $SEEDS"
echo "Workloads:   $wl_mode"
echo "Total runs:  $total_runs ($router_count routers x $wl_count workloads x $seed_count seeds)"
echo ""

# --- Build baseline BLIS first ---
echo "--- Building BLIS (baseline / stock routing.go) ---"
cd "$REPO_ROOT"
if ! go build -o blis main.go; then
    echo "FATAL: Initial build failed"
    exit 1
fi
echo "Build complete."
echo ""

# --- Backup original routing.go ---
cp "$ORIGINAL_ROUTING" "$BACKUP_ROUTING"

# --- Safety: restore routing.go on exit ---
cleanup() {
    echo ""
    echo "--- Restoring original routing.go ---"
    cp "$BACKUP_ROUTING" "$ORIGINAL_ROUTING"
    rm -f "$BACKUP_ROUTING"
    echo "Restored."
}
trap cleanup EXIT

rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

run_count=0
fail_count=0
timeout_count=0

for router_entry in $ROUTERS; do
    [ -z "$router_entry" ] && continue
    router_label="${router_entry%%:*}"
    rest="${router_entry#*:}"
    router_file="${rest%%:*}"
    policy_yaml="${rest#*:}"
    policy_path="$ROUTERS_DIR/$policy_yaml"

    echo "============================================================"
    echo " Router: $router_label"
    echo "============================================================"

    # --- Swap routing.go if needed ---
    if [ "$router_file" = "STOCK" ]; then
        cp "$BACKUP_ROUTING" "$ORIGINAL_ROUTING"
        echo "  Using stock BLIS routing.go (no swap)"
    else
        router_path="$ROUTERS_DIR/$router_file"
        if [ ! -f "$router_path" ]; then
            echo "  ERROR: router file not found: $router_path"
            continue
        fi
        cp "$router_path" "$ORIGINAL_ROUTING"
        echo "  Swapped: $router_file -> sim/routing.go"
    fi

    # --- Rebuild BLIS with swapped routing.go ---
    echo "  Building BLIS..."
    cd "$REPO_ROOT"
    if ! go build -o blis main.go 2>"$RESULTS_DIR/${router_label}_build.log"; then
        echo "  BUILD FAILED — see ${router_label}_build.log"
        cp "$BACKUP_ROUTING" "$ORIGINAL_ROUTING"
        continue
    fi
    echo "  Build OK"
    echo ""

    # --- Run workloads ---
    for wl_name in $WORKLOADS; do
        [ -z "$wl_name" ] && continue
        wl_path="$WORKLOADS_DIR/${wl_name}.yaml"

        for seed in $SEEDS; do
            metrics_out="$RESULTS_DIR/${wl_name}_${router_label}_seed${seed}.json"
            printf "  %-35s %-20s seed=%-3s ... " "$wl_name" "$router_label" "$seed"

            cd "$REPO_ROOT"
            ./blis run \
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
                > "${metrics_out%.json}.stdout" 2>&1 &
            blis_pid=$!
            ( sleep $TIMEOUT_SECS && kill $blis_pid 2>/dev/null ) &
            watchdog_pid=$!

            wait $blis_pid 2>/dev/null
            exit_code=$?
            kill $watchdog_pid 2>/dev/null
            wait $watchdog_pid 2>/dev/null

            if [ "$exit_code" = "0" ]; then
                echo "done"
                run_count=$((run_count + 1))
            elif [ "$exit_code" = "143" ] || [ "$exit_code" = "137" ]; then
                echo "TIMEOUT (${TIMEOUT_SECS}s)"
                timeout_count=$((timeout_count + 1))
                fail_count=$((fail_count + 1))
            else
                echo "FAILED (exit=$exit_code)"
                fail_count=$((fail_count + 1))
            fi
        done
    done

    echo ""
    echo "  Progress: $run_count succeeded, $fail_count failed ($timeout_count timeouts)"
    echo ""
done

# --- Restore routing.go ---
cp "$BACKUP_ROUTING" "$ORIGINAL_ROUTING"

echo "============================================================"
echo " Running analysis..."
echo "============================================================"
echo ""
python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR"

echo ""
echo "============================================================"
echo " Benchmark complete: $run_count succeeded, $fail_count failed ($timeout_count timeouts)"
echo " Results in: $RESULTS_DIR/"
echo "============================================================"
