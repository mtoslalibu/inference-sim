#!/bin/bash
# Run baseline (gaie-legacy) and adaptive-admission on all workloads
set -euo pipefail

BLIS="$(cd "$(dirname "$0")/../.." && pwd)/blis"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS="${WORKDIR}/results"
WORKLOADS="${WORKDIR}/workloads"

MODEL="qwen/qwen3-14b"
COMMON_FLAGS="--num-instances 4 --routing-policy round-robin --snapshot-refresh-interval 50000"

ADMISSION_POLICY="${1:-both}"  # "gaie-legacy", "adaptive-admission", or "both"
ITERATION="${2:-baseline}"     # label for results subdirectory

mkdir -p "${RESULTS}"

run_one() {
    local policy="$1"
    local workload="$2"
    local wname
    wname=$(basename "$workload" .yaml)
    local outfile="${RESULTS}/${ITERATION}_${policy}_${wname}.txt"

    echo "=== Running ${policy} on ${wname} (iter=${ITERATION}) ==="
    ${BLIS} run --model ${MODEL} \
        --workload-spec "${workload}" \
        --admission-policy "${policy}" \
        ${COMMON_FLAGS} \
        > "${outfile}" 2>/dev/null
    echo "  -> ${outfile}"
}

policies=()
if [ "${ADMISSION_POLICY}" = "both" ]; then
    policies=("gaie-legacy" "adaptive-admission")
else
    policies=("${ADMISSION_POLICY}")
fi

for policy in "${policies[@]}"; do
    for wl in "${WORKLOADS}"/*.yaml; do
        run_one "${policy}" "${wl}"
    done
done

echo ""
echo "All runs complete. Results in ${RESULTS}/"
echo "Run: python3 ${WORKDIR}/analyze.py ${RESULTS}/${ITERATION}_*.txt"
