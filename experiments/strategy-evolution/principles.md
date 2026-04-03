# Principles Catalog — Routing Algorithm Evolution

## P1: Min-max normalization makes weights irrelevant at N=2

With 2 instances, min-max normalization reduces every signal to binary: better gets 1.0, worse gets 0.0. Any linear combination of binary signals produces the same argmax regardless of weights.

**Evidence:** Iteration 1 — marginal-gain scoring with α=1.0 produced byte-identical results to 3:2:2 weighted scoring on all workloads.

**Implication:** To differentiate algorithms at N=2, must use non-normalized scoring or non-linear decision mechanisms.

## P2: Any monotone load scoring produces the same argmax

Whether linear, exponential, or min-max: if the scoring function is monotonically decreasing in load, the instance with lowest load always wins. This holds at any N.

**Evidence:** Iteration 2 — exp(-β*load) with β=0.1 produced identical results to queue-depth min-max scoring.

**Implication:** To beat least-loaded routing, the algorithm must use information beyond aggregate load count.

## P3: Cache-aware routing is counterproductive when used as a weight

Routing to the instance with better cache hits creates load imbalance. The increased queuing delay from load imbalance outweighs the prefill time saved from cache hits. Even with oracle cache (0s delay), pure cache routing is worse than pure load balance.

**Evidence:** At 4 instances with diverse prefix groups, ppc:5,qd:1 produced worse E2E P99 (+8%) than qd:1 alone.

**Implication:** Cache signal should be used as a FILTER (triage), not a WEIGHT. Only consider cache when deciding between instances with similar load.

## P4: InFlightRequests dominates routing quality

The synchronous InFlightRequests signal (updated per routing decision) makes all load-based routing algorithms equivalent within <0.5%. The stale QueueDepth/BatchSize from periodic snapshots add negligible information.

**Evidence:** At 5s vs 30s staleness, E2E mean differs by <0.1%. Round-robin vs optimal differs by <0.5%.

**Implication:** The "routing opportunity ceiling" is very low. Massive improvements (>20%) can only come from exploiting GLIA's fundamental architectural gap (no prefix cache awareness), not from better signal processing.

## P5: kv-utilization is toxic alone but harmless in combination

Pure kv-utilization routing is catastrophic (90% worse at N=4 with extreme variance). But at 29% weight in the 3:2:2 combo, it has zero measurable impact because InFlightRequests in the queue-depth component dominates.

**Evidence:** 3:2:2 vs qd:1 differs by <0.1% across all workloads and configurations tested.

**Implication:** The prior principle RP-6 ("kv-utilization counterproductive") is correct for solo use but the practical impact in weighted combinations is negligible.

## P6: Two-stage routing (filter + balance) prevents cache-induced imbalance

Using cache as a triage filter (reduce candidate set) then load-balancing within the filtered set avoids the P3 problem. Cache only narrows the options; load governs the final choice.

**Evidence:** Iteration 3 — Adaptive-Cache-LB beats GLIA by 23-41% on E2E while matching 3:2:2 exactly.

**Implication:** The winning architecture for cache-aware routing is: filter by cache, then balance by load. Not: weighted combination of cache and load scores.

## P7: Routing opportunity ceiling is ~1% E2E in the blackbox DES

Three factors combine to make all non-degenerate routing algorithms equivalent within ~1% E2E:
1. InFlightRequests is perfectly synchronous (updated per routing decision, never stale)
2. KV cache is oversized (1M blocks default; all instances cache all prefixes → no cache differentiation)
3. Blackbox model has no preemption/KV-pressure effects (latency is pure function of token count and batch size)

**Evidence:** Iteration 4 — exhaustive weight ratio sweep across 3 workloads × 7 weight ratios. All non-degenerate configurations produce identical E2E within ±0.3%. Hash-affinity with 2-instance constraint was 5-11% WORSE. Token-weighted tiebreaking had zero measurable impact.

**Implication:** Beating 3:2:2 by 30%+ requires changes beyond routing: scheduling, admission control, batch formation, or a simulator with imperfect InFlightRequests (e.g., network-delayed updates).

## P8: Priority-through-routing is defeated by PriorityPolicy

The `req.Priority` field set by `RoutingDecision.Priority` is overwritten every scheduling step by `sim.priorityPolicy.Compute()`. With the default `constant` policy, all priorities become 0.0 regardless of what the routing layer sets.

**Evidence:** Iteration 5 — Priority-Cache-LB set Priority = 1/(1+inputTokens) and tested with `--scheduler priority-fcfs`. Results were byte-identical to FCFS because the constant PriorityPolicy zeros out all routing-set priorities.

**Implication:** To influence scheduling from routing, the simulator would need a PriorityPolicy that preserves or incorporates routing-set priorities.

## P9: Prefix cache signal provides no routing benefit at any staleness level

At any cache-signal-delay value, the `precise-prefix-cache` scorer either hurts or has no effect:
- Fresh (0–0.1s): Causes 3–4x E2E degradation by creating catastrophic load imbalance (ALL cache-aware requests pile onto the cached instance)
- Moderate to stale (≥0.5s): Signal becomes stale noise; indistinguishable from random tiebreaking

Root cause: with oversized KV cache (1M blocks), ALL instances cache ALL prefixes. Fresh cache signal accurately identifies this → no differentiation. When some instances temporarily lack a prefix (post-eviction), the fresh signal creates pile-on; the stale signal misses the window.

**Evidence:** Iteration 5 — swept cache-signal-delay [0s, 0.1s, 0.5s, 1s, 2s, 5s] for 3:2:2, Adaptive-Cache-LB, and qd:1. Both cache-aware algorithms degraded identically at low delay. At ≥0.5s, all three converged to within 0.2%.

**Implication:** The only beneficial use of cache information is to avoid GLIA's failure mode (no cache awareness + inadmissibility penalty). The Adaptive-Cache-LB's advantage over GLIA comes from NOT penalizing cache-rich instances, not from superior cache-aware routing.
