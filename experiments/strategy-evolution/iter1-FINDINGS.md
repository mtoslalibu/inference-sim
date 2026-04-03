# Iteration 1 FINDINGS: Marginal-Gain Scoring

## H-main Results

| Metric | W1 vs 3:2:2 | W1 vs GLIA | W2 vs 3:2:2 | W2 vs GLIA |
|--------|-------------|-----------|-------------|-----------|
| E2E Mean | -0.0% | -26.6% | +0.0% | -1.9% |
| E2E P99 | +0.6% | -44.6% | -0.3% | +0.8% |
| TTFT Mean | +1.1% | -72.9% | +0.0% | -72.2% |
| TTFT P99 | -0.4% | -88.0% | +5.8% | -83.0% |

**Prediction vs Outcome:**
- W1 vs GLIA: **Confirmed** — ≥20% E2E P99 improvement (got 44.6%)
- W1 vs 3:2:2: **Confirmed** — within 5% (got +0.6%)
- W2 vs both: **Refuted** — no improvement, essentially identical to 3:2:2

**H-main partially confirmed.** Beats GLIA strongly on W1 but is a carbon copy of 3:2:2 on both workloads.

## Root Cause Analysis

**Critical insight: With 2 instances, min-max normalization makes all weighted scoring equivalent.**

With N=2 instances, min-max normalization reduces every signal to binary: the better instance gets 1.0, the worse gets 0.0 (or both get 0.5/1.0 when tied). Any linear combination of binary signals produces the same ordering regardless of weights.

This means:
1. `precise-prefix-cache:3, queue-depth:2` ≡ `precise-prefix-cache:1, queue-depth:1` ≡ marginal-gain α=1.0 ≡ any weighted combination
2. The only time weights matter is when signals **disagree** — and even then, with binary signals, the higher-weighted signal always wins
3. The kv-utilization component in 3:2:2 is the third binary signal. It agrees with queue-depth most of the time (correlated), making it redundant, not harmful.

**Conclusion: The weighted scoring framework itself is the bottleneck, not the weights.** Any improvement must come from a fundamentally different decision mechanism.

## Principles Extracted

**P1: Min-max normalization makes weights irrelevant at N=2 instances.**
All linear combinations of min-max normalized binary signals produce the same routing decisions. The only exception is when signals disagree and weights determine the winner.

**P2: To beat weighted scoring, the algorithm must use magnitude-aware decisions.**
Instead of "which instance is better" (binary), the algorithm must answer "is the cache advantage worth the load cost" (magnitude comparison in comparable units).

## Iteration Decision

**Iterate.** Marginal-gain scoring is not a new mechanism — it degenerates to the same behavior as weighted scoring. Need a fundamentally different approach.

**Next direction:** Physics-informed routing — convert cache benefit and load cost to common time units, then make a threshold decision.
