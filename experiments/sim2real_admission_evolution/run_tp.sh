#!/bin/bash
set -e

DIR="experiments/sim2real_admission_evolution"
RDIR="$DIR/results_tp"
mkdir -p "$RDIR"

run() {
  local model=$1 admission=$2 workload=$3 output=$4
  echo "Running: $output"
  ./blis run \
    --model "$model" \
    --hardware H100 \
    --tp 1 \
    --latency-model trained-physics \
    --num-instances 4 \
    --routing-policy round-robin \
    --snapshot-refresh-interval 50000 \
    --admission-policy "$admission" \
    --workload-spec "$workload" \
    > "$RDIR/$output" 2>&1
  echo "  Done: $output"
}

# 14B runs
run Qwen/Qwen3-14B gaie-legacy        "$DIR/workloads_tp/w1_14b.yaml"                14b_baseline_w1.txt
run Qwen/Qwen3-14B adaptive-admission  "$DIR/workloads_tp/w1_14b.yaml"                14b_iter11_w1.txt
run Qwen/Qwen3-14B gaie-legacy        "$DIR/workloads_tp/w2_14b_burst.yaml"           14b_baseline_w2.txt
run Qwen/Qwen3-14B adaptive-admission  "$DIR/workloads_tp/w2_14b_burst.yaml"           14b_iter11_w2.txt
run Qwen/Qwen3-14B gaie-legacy        "$DIR/workloads_tp/w3_14b_high_sheddable.yaml"   14b_baseline_w3.txt
run Qwen/Qwen3-14B adaptive-admission  "$DIR/workloads_tp/w3_14b_high_sheddable.yaml"   14b_iter11_w3.txt

# 32B runs
run Qwen/Qwen3-32B gaie-legacy        "$DIR/workloads_32b/w1_sustained_overload.yaml"  32b_baseline_w1.txt
run Qwen/Qwen3-32B adaptive-admission  "$DIR/workloads_32b/w1_sustained_overload.yaml"  32b_iter11_w1.txt
run Qwen/Qwen3-32B gaie-legacy        "$DIR/workloads_32b/w2_burst.yaml"               32b_baseline_w2.txt
run Qwen/Qwen3-32B adaptive-admission  "$DIR/workloads_32b/w2_burst.yaml"               32b_iter11_w2.txt
run Qwen/Qwen3-32B gaie-legacy        "$DIR/workloads_32b/w3_high_sheddable.yaml"       32b_baseline_w3.txt
run Qwen/Qwen3-32B adaptive-admission  "$DIR/workloads_32b/w3_high_sheddable.yaml"       32b_iter11_w3.txt

echo "All 12 runs complete."
