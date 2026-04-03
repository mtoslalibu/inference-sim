# Iterations 4-5 FINDINGS: Attempting to Beat 3:2:2 by 30%+

## Hypothesis
Static weights are the weakness of 3:2:2. Adaptive algorithms that use request-level information (size, prefix group) or non-linear routing mechanisms (hash affinity, priority scheduling, token-weighted load) should beat 3:2:2 by 30%+ on targeted workloads.

## Approaches Tested

### Iter 4a: Hash-Affinity Routing
**Mechanism:** Hash(PrefixGroup) → deterministic 2-instance affinity set. Load-balance within affinity set. Overflow to all instances when affinity set overloaded.

**Results (W6, 4 instances):** +5.3% E2E mean, +68% TTFT P99 vs 3:2:2.
**Results (W7, 8 instances):** +10.7% E2E mean, +165% TTFT P99 vs 3:2:2.

**Finding:** Hash affinity is WORSE than 3:2:2. The 2-instance constraint creates load imbalance that dwarfs cache locality benefits. Confirms P3: cache routing is counterproductive even when deterministic.

### Iter 4b: Weight Ratio Sweep
**Mechanism:** Exhaustive sweep of weight ratios across 3 workloads at 4 instances.

**Results:** ALL configurations (ppc322, ppc511, qd1, ppc1qd1, qd1kv5, lb1, ppc2qd5) produce identical E2E within ±0.3%. Only exception: pure kv:1 (catastrophic, 2-3x worse).

**Finding (P7):** InFlightRequests dominance means all non-degenerate weight ratios converge. The user's premise that "4:1:1 beats for some, 1:1:10 for others" does not hold in the blackbox DES.

### Iter 4c: Instance Count Sweep (8 instances)
**Results:** 3:2:2 and qd:1 still identical at 8 instances. GLIA much worse (66% E2E gap). Hash-affinity still worse than both.

### Iter 5a: Size-Aware Priority Scheduling
**Mechanism:** Route() sets `RoutingDecision.Priority = 1/(1+inputTokens)`. With `priority-fcfs` scheduler, small requests jump queue.

**Finding (P8):** Priority field is OVERWRITTEN by PriorityPolicy every scheduling step. With default `constant` policy, all priorities become 0.0. Routing-set priority has zero effect.

### Iter 5b: SJF Scheduler Comparison
**Results (W9):** SJF gives -2.3% E2E mean but -43% TTFT mean vs FCFS. This improvement comes from scheduling, not routing — both algorithms benefit equally from SJF.

### Iter 5c: Cache Signal Delay Sweep
**Critical Finding:** Swept cache-signal-delay from 0s to 5s on W8.

| Cache Delay | 3:2:2 E2E | ACLB E2E | qd:1 E2E |
|-------------|-----------|----------|----------|
| 0.0s | 4583 | 4581 | 1113 |
| 0.1s | 3042 | 3041 | 1113 |
| 0.5s | 1114 | 1113 | 1113 |
| 1.0s | 1114 | 1113 | 1113 |
| 2.0s | 1114 | 1113 | 1113 |
| 5.0s | 1114 | 1113 | 1113 |

**Finding (P9):** The precise-prefix-cache signal is TOXIC when fresh (0-0.1s delay) — causes 3-4x degradation for ANY cache-aware algorithm (both weighted and filter-based). At ≥0.5s it's stale enough to be harmless noise. There is NO regime where the cache signal improves routing — it's either harmful or useless.

**Root cause:** With large KV cache (1M blocks), all instances cache all prefixes. Fresh cache signal correctly identifies cached instances → all requests route there → catastrophic load imbalance. Stale signal provides no differentiation → harmless noise.

## New Principles

### P7: Routing opportunity ceiling is ~1% E2E in the blackbox DES
Three factors combine to make all non-degenerate routing algorithms equivalent:
1. InFlightRequests is perfectly synchronous (updated per routing decision)
2. KV cache is oversized (all instances cache all prefixes → no cache differentiation)
3. Blackbox model has no preemption/KV-pressure effects (latency is pure computation)

**Implication:** The 30% improvement target vs 3:2:2 is outside the routing opportunity ceiling. It requires changes to scheduling, admission, or the simulator's signal propagation model.

### P8: Priority-through-routing is defeated by PriorityPolicy
The `req.Priority` field set by routing is overwritten every step by `sim.priorityPolicy.Compute()`. With the default `constant` policy, all priorities become 0.0.

### P9: Prefix cache signal is never beneficial for routing
At any cache-signal-delay value:
- Fresh (0-0.1s): Causes catastrophic load imbalance (3-4x E2E degradation)
- Moderate (0.5-2s): Stale enough to be noise → zero routing benefit
- Very stale (5s+): Pure noise → zero routing benefit

The only regime where cache information helps is when combined with GLIA's inadmissibility penalty (which is itself harmful). The Adaptive-Cache-LB's cache triage is equivalent to a no-op at 2s delay.

## Conclusion

The Adaptive-Cache-LB algorithm (iter3) represents the practical ceiling for routing-level improvements in the blackbox DES:
- **Beats GLIA** by 23-41% E2E (from GLIA's inadmissibility penalty, not from our cache awareness)
- **Matches 3:2:2** within ±1% (both converge to InFlightRequests-dominated load balance)
- **Cannot beat 3:2:2 by 30%+** due to P7 (routing opportunity ceiling)

The improvement over GLIA comes from NOT having GLIA's -10.0 penalty cliff, not from superior cache awareness.
