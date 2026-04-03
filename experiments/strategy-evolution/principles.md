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
