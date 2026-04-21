#!/bin/bash
# Phase 1: Run baseline (3:2:2) against all probe workloads
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS="$SCRIPT_DIR/results"
WORKLOADS="$SCRIPT_DIR/workloads"
BLIS="$REPO_ROOT/blis"

mkdir -p "$RESULTS"

COMMON="\
  --model Qwen/Qwen3-14B \
  --hardware H100 \
  --tp 1 \
  --latency-model trained-physics \
  --num-instances 4 \
  --block-size-in-tokens 64 \
  --gpu-memory-utilization 0.95 \
  --total-kv-blocks 4719 \
  --max-num-scheduled-tokens 2048 \
  --max-num-running-reqs 256 \
  --max-model-len 40960 \
  --snapshot-refresh-interval 50000 \
  --cache-signal-delay 50000 \
  --routing-policy weighted \
  --routing-scorers precise-prefix-cache:3,queue-depth:2,kv-utilization:2"

cd "$REPO_ROOT"

for workload in w1_prefix_heavy w2_cold_burst w3_kv_pressure w4_bursty_asymmetric; do
  echo "Running baseline: $workload"
  $BLIS run $COMMON --workload-spec "$WORKLOADS/${workload}.yaml" \
    > "$RESULTS/baseline_${workload}.txt" 2>&1
  echo "  Done: $workload"
done

echo ""
echo "All baseline runs complete."
echo "Results in: $RESULTS/"
