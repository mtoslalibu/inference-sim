#!/bin/bash
# Run GAIE-legacy (vanilla) vs GAIE-legacy (quintic-patched) comparison for Qwen3-14B
#
# This script:
#   1. Builds BLIS (vanilla) and runs gaie-legacy baseline for all workloads
#   2. Patches GAIELegacyAdmission in sim/admission.go with the binary quintic algorithm
#   3. Rebuilds BLIS and runs gaie-legacy (now quintic) for all workloads
#   4. Resets sim/admission.go via git checkout
#   5. Prints a comparison table
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RESULTS="$SCRIPT_DIR/results"
WORKLOADS="$SCRIPT_DIR/workloads"
BLIS="$REPO_ROOT/blis"
ADMISSION_FILE="$REPO_ROOT/sim/admission.go"

mkdir -p "$RESULTS"

# --- BLIS flags calibrated to real vLLM deployment ---
run_blis() {
  local admission=$1 workload=$2 output=$3
  echo "  Running: $output"
  "$BLIS" run \
    --model Qwen/Qwen3-14B \
    --hardware H100 \
    --tp 1 \
    --latency-model trained-physics \
    --num-instances 4 \
    --routing-policy round-robin \
    --snapshot-refresh-interval 50000 \
    --block-size-in-tokens 64 \
    --gpu-memory-utilization 0.95 \
    --total-kv-blocks 4719 \
    --max-num-scheduled-tokens 2048 \
    --max-num-running-reqs 256 \
    --max-model-len 40960 \
    --admission-policy "$admission" \
    --workload-spec "$workload" \
    > "$RESULTS/$output" 2>&1
  echo "  Done: $output"
}

# ============================================================
# STEP 1: Build vanilla BLIS, run gaie-legacy baseline
# ============================================================
echo "=== Step 1: Building BLIS (vanilla) and running gaie-legacy baseline ==="
cd "$REPO_ROOT"
go build -o blis main.go

WORKLOAD_LIST="w1_under w1_mid w2_under w2_mid chatbot_under chatbot_mid codecompletion_under codecompletion_mid blindspot_under blindspot_mid"

for w in $WORKLOAD_LIST; do
  run_blis gaie-legacy "$WORKLOADS/${w}.yaml" "baseline_${w}.txt" &
done
wait
echo "=== Baseline complete ==="

# ============================================================
# STEP 2: Patch GAIELegacyAdmission with binary quintic
# ============================================================
echo "=== Step 2: Patching GAIELegacyAdmission with binary quintic ==="

# Binary quintic algorithm (k=300):
#   protected (IsSheddable=false, priority >= 0): always admit
#   droppable (IsSheddable=true, priority < 0): p = min(sat^5 * 300, 1.0)

python3 << 'PATCH_EOF'
path = "sim/admission.go"
with open(path) as f:
    content = f.read()

# 1. Add counter fields to GAIELegacyAdmission struct
old_struct = '''type GAIELegacyAdmission struct {
	// Per-instance queue depth at which the QD component reaches 1.0.
	// Default: 5 — from GAIE DefaultQueueDepthThreshold
	// (saturationdetector/utilization/config.go:31).
	QDThreshold float64

	// Per-instance KV cache utilization at which the KV component reaches 1.0.
	// Default: 0.8 — from GAIE DefaultKVCacheUtilThreshold
	// (saturationdetector/utilization/config.go:33).
	KVThreshold float64

	PriorityMap *SLOPriorityMap // priority mapping for IsSheddable check
}'''

new_struct = '''type GAIELegacyAdmission struct {
	// Per-instance queue depth at which the QD component reaches 1.0.
	// Default: 5 — from GAIE DefaultQueueDepthThreshold
	// (saturationdetector/utilization/config.go:31).
	QDThreshold float64

	// Per-instance KV cache utilization at which the KV component reaches 1.0.
	// Default: 0.8 — from GAIE DefaultKVCacheUtilThreshold
	// (saturationdetector/utilization/config.go:33).
	KVThreshold float64

	PriorityMap *SLOPriorityMap // priority mapping for IsSheddable check

	totalAdmitted int
	totalRejected int
}'''

assert old_struct in content, "ERROR: Could not find GAIELegacyAdmission struct"
content = content.replace(old_struct, new_struct, 1)

# 2. Replace Admit method with binary quintic
old_admit = '''// Admit implements AdmissionPolicy. Non-sheddable requests always pass.
// Sheddable requests are rejected when pool-average saturation >= 1.0.
func (g *GAIELegacyAdmission) Admit(req *Request, state *RouterState) (bool, string) {
	if !g.PriorityMap.IsSheddable(req.SLOClass) {
		return true, ""
	}
	sat := g.saturation(state.Snapshots)
	if sat >= 1.0 {
		return false, fmt.Sprintf("gaie-saturated: class=%s saturation=%.2f", req.SLOClass, sat)
	}
	return true, ""
}'''

new_admit = '''// Admit implements AdmissionPolicy. Binary quintic (patched for experiment).
// Protected (priority >= 0): always admit. Droppable (priority < 0): p = min(sat^5 * 300, 1.0).
func (g *GAIELegacyAdmission) Admit(req *Request, state *RouterState) (bool, string) {
	if !g.PriorityMap.IsSheddable(req.SLOClass) {
		g.totalAdmitted++
		return true, ""
	}
	sat := g.saturation(state.Snapshots)
	sat5 := sat * sat * sat * sat * sat
	p := sat5 * 300.0
	if p > 1.0 {
		p = 1.0
	}
	if p > 0 {
		ordinal := float64(g.totalAdmitted+g.totalRejected) / 100.0
		randVal := ordinal - float64(int(ordinal))
		if randVal < p {
			g.totalRejected++
			return false, fmt.Sprintf("quintic: %s-shed sat=%.3f p=%.2f", req.SLOClass, sat, p)
		}
	}
	g.totalAdmitted++
	return true, ""
}'''

assert old_admit in content, "ERROR: Could not find GAIELegacyAdmission.Admit to patch"
content = content.replace(old_admit, new_admit, 1)

with open(path, 'w') as f:
    f.write(content)

print("Patched GAIELegacyAdmission successfully.")
PATCH_EOF

# ============================================================
# STEP 3: Rebuild BLIS with quintic-patched GAIE, run treatment
# ============================================================
echo "=== Step 3: Rebuilding BLIS (quintic) and running treatment ==="
go build -o blis main.go

WORKLOAD_LIST="w1_under w1_mid w2_under w2_mid chatbot_under chatbot_mid codecompletion_under codecompletion_mid blindspot_under blindspot_mid"

for w in $WORKLOAD_LIST; do
  run_blis gaie-legacy "$WORKLOADS/${w}.yaml" "quintic_${w}.txt" &
done
wait
echo "=== Quintic complete ==="

# ============================================================
# STEP 4: Restore original admission.go
# ============================================================
echo "=== Step 4: Restoring sim/admission.go to original ==="
git checkout "$ADMISSION_FILE"
go build -o blis main.go
echo "=== Code restored and rebuilt ==="

# ============================================================
# STEP 5: Compare results
# ============================================================
echo ""
echo "=== Results Comparison ==="
echo ""

python3 << 'COMPARE_EOF'
import re, os

RDIR = os.environ.get("RESULTS_DIR", "experiments/sim2real_admission__llmd_real/apr28-meeting-comprehensive/sim2real_bundle_paramfree/results")
workloads = ["w1_under", "w1_mid", "w2_under", "w2_mid", "chatbot_under", "chatbot_mid", "codecompletion_under", "codecompletion_mid", "blindspot_under", "blindspot_mid"]
tier_order = ["critical", "sheddable"]

def parse(path):
    with open(path) as f:
        text = f.read()
    result = {}
    for m in re.finditer(
        r'  (\w+):\n    TTFT: mean=([\d.]+) p99=([\d.]+) \(n=(\d+)\)\n    E2E:\s+mean=([\d.]+) p99=([\d.]+) \(n=(\d+)\)',
        text
    ):
        tier = m.group(1)
        result[tier] = {
            'ttft_mean': float(m.group(2)) / 1000.0,
            'ttft_p99':  float(m.group(3)) / 1000.0,
            'e2e_mean':  float(m.group(5)) / 1000.0,
            'e2e_p99':   float(m.group(6)) / 1000.0,
            'n':         int(m.group(7)),
        }
    for m in re.finditer(r'Shed \((\w+)\): (\d+)', text):
        tier = m.group(1)
        if tier in result:
            result[tier]['shed'] = int(m.group(2))
    rej = re.search(r'Rejected Requests \(Admission\): (\d+)', text)
    result['_total_rejected'] = int(rej.group(1)) if rej else 0
    return result

data = {}
for w in workloads:
    data[w] = {}
    for group, prefix in [('baseline', 'baseline'), ('quintic', 'quintic')]:
        path = os.path.join(RDIR, f"{prefix}_{w}.txt")
        if os.path.exists(path):
            data[w][group] = parse(path)

# --- Latency table ---
print("LATENCY (ms) — GAIE-Legacy (B) vs Binary Quintic k=300 (Q)")
print()
print(f"{'':>21}| {'GAIE BASELINE':^40} | {'QUINTIC':^40} | {'E2E DELTA':^15}")
print(f"{'Workload':<10} {'Tier':<10} | {'E2E-Mean':>9} {'E2E-P99':>9} {'TTFT-Mean':>10} {'TTFT-P99':>9} | {'E2E-Mean':>9} {'E2E-P99':>9} {'TTFT-Mean':>10} {'TTFT-P99':>9} | {'dMean':>7} {'dP99':>7}")
print('-' * 130)
for w in workloads:
    for tier in tier_order:
        b = data[w].get('baseline', {}).get(tier)
        q = data[w].get('quintic', {}).get(tier)
        if not b or not q:
            continue
        dm = ((q['e2e_mean'] - b['e2e_mean']) / b['e2e_mean'] * 100) if b['e2e_mean'] else 0
        dp = ((q['e2e_p99'] - b['e2e_p99']) / b['e2e_p99'] * 100) if b['e2e_p99'] else 0
        print(f"{w:<10} {tier:<10} | {b['e2e_mean']:>9.0f} {b['e2e_p99']:>9.0f} {b['ttft_mean']:>10.1f} {b['ttft_p99']:>9.1f} | {q['e2e_mean']:>9.0f} {q['e2e_p99']:>9.0f} {q['ttft_mean']:>10.1f} {q['ttft_p99']:>9.1f} | {dm:>+6.0f}% {dp:>+6.0f}%")
    print()

# --- Shed table ---
print()
print("SHED STATS — Completed (n) and Shed counts per tier")
print()
print(f"{'Workload':<10} {'Tier':<10} | {'n (B)':>7} {'Shed (B)':>9} | {'n (Q)':>7} {'Shed (Q)':>9} | {'dShed':>7}")
print('-' * 72)
for w in workloads:
    for tier in tier_order:
        b = data[w].get('baseline', {}).get(tier, {})
        q = data[w].get('quintic', {}).get(tier, {})
        bn = b.get('n', 0); qn = q.get('n', 0)
        bs = b.get('shed', 0); qs = q.get('shed', 0)
        print(f"{w:<10} {tier:<10} | {bn:>7} {bs:>9} | {qn:>7} {qs:>9} | {qs-bs:>+7}")
    br = data[w].get('baseline', {}).get('_total_rejected', 0)
    qr = data[w].get('quintic', {}).get('_total_rejected', 0)
    print(f"{'':>10} {'TOTAL':>10} | {'':>7} {br:>9} | {'':>7} {qr:>9} | {qr-br:>+7}")
    print()

COMPARE_EOF

echo ""
echo "All done. Results in: $RESULTS"
