# Strategy Evolution Ledger

## Final Results (4 instances, 3 seeds avg)

### W1: Prefix-Burst (80% prefix-heavy with 2 prefix groups, 20% cold, gamma cv=2.0, rate=100, 6000 reqs)

| Strategy | E2E Mean (ms) | E2E P99 (ms) | TTFT Mean (ms) | TTFT P99 (ms) | TPS |
|----------|--------------|--------------|----------------|---------------|-----|
| **Adaptive-CLB** | **1241.6** | **2326.4** | **41.8** | **108.1** | 10436 |
| BLIS 3:2:2 | 1242.5 | 2342.0 | 41.5 | 109.7 | 10439 |
| GLIA HRA | 1624.6 | 3935.2 | 67.0 | 369.3 | 10416 |

### W2: GLIA-Stress (80% prefix-heavy with long prefixes, gamma cv=3.0, rate=80, 5000 reqs)

| Strategy | E2E Mean (ms) | E2E P99 (ms) | TTFT Mean (ms) | TTFT P99 (ms) | TPS |
|----------|--------------|--------------|----------------|---------------|-----|
| **Adaptive-CLB** | **996.5** | **2353.6** | **74.6** | **266.2** | 6061 |
| BLIS 3:2:2 | 998.7 | 2349.3 | 74.1 | 266.7 | 6063 |
| GLIA HRA | 1286.8 | 3383.2 | 132.8 | 636.0 | 6022 |

### Improvement Summary

| Workload | vs GLIA E2E Mean | vs GLIA E2E P99 | vs GLIA TTFT P99 | vs 3:2:2 E2E Mean | vs 3:2:2 E2E P99 |
|----------|-----------------|-----------------|------------------|-------------------|------------------|
| W1 Prefix-Burst | **-23.6%** | **-40.9%** | **-70.7%** | -0.1% | -0.7% |
| W2 GLIA-Stress | **-22.6%** | **-30.4%** | **-58.1%** | -0.2% | +0.2% |

## Iteration Log

| Iter | Strategy | Mechanism | Key Finding | Prediction Accuracy | Status |
|------|----------|-----------|-------------|-------------------|--------|
| 0 | BLIS 3:2:2 | Static weighted scorers (ppc:3,qd:2,kv:2) | Baseline | — | Measured |
| 0 | GLIA HRA | KV headroom projection, no cache | Baseline | — | Measured |
| 1 | Marginal-Gain | normCache - α*normLoad | **P1: Min-max normalization makes weights irrelevant at N=2.** Algorithm degenerated to identical behavior as 3:2:2. | H-main partially confirmed (beats GLIA), H-improvement refuted (≡3:2:2) | Bundle verified |
| 2 | Exp-Decay | exp(-β*load) scoring | **P2: Any monotone scoring of load produces same argmax.** InFlightRequests synchronous signal makes all load-based algorithms equivalent (<0.1% difference). | H-main refuted | Fast-fail |
| 2b | Signal analysis | Tested signal freshness, instance count, workload variance | **P3: Cache-aware routing counterproductive** — creates load imbalance outweighing cache benefit. **P4: InFlightRequests dominates** — routing opportunity ceiling is <0.5%. **P5: kv-utilization alone is catastrophic** (90% worse at N=4) but at 29% weight harmless. | — | Principle extraction |
| 3 | Adaptive-Cache-LB | Two-stage: cache triage → load balance | **Beats GLIA by 23-41% E2E, matches 3:2:2.** Cache filter prevents load imbalance (P3) while maintaining cache awareness. | H-main confirmed: beats GLIA on W1 and W2, matches 3:2:2 | **Winner** |
| 4a | Hash-Affinity | Hash(PrefixGroup) → 2-instance affinity set | **5-11% WORSE than 3:2:2.** Load imbalance from affinity constraint outweighs cache locality. P3 confirmed for deterministic routing. | H-main refuted | Rejected |
| 4b | Weight Ratio Sweep | 7 weight combos × 3 workloads | **P7: All non-degenerate weight ratios produce identical E2E (±0.3%).** InFlightRequests synchronous signal makes all load-based routing equivalent. | — | Principle extraction |
| 4c | 8-Instance Scale | Same configs at N=8 | Same convergence at larger scale. GLIA gap widens (66% E2E). Hash-affinity still worse. | — | Confirming |
| 5a | Priority-Cache-LB | Size-aware Priority + cache triage | **P8: Priority field overwritten by PriorityPolicy.** Routing-set priority has zero effect. | H-main refuted | Dead end |
| 5b | SJF Scheduling | System-level SJF (--scheduler sjf) | -2.3% E2E, -43% TTFT mean. Scheduling improvement, not routing — applies equally to all algorithms. | — | Out of scope |
| 5c | Cache Delay Sweep | Swept cache-signal-delay 0s–5s | **P9: Cache signal is toxic when fresh (3-4x degradation) and useless when stale (±0.2%).** No regime where cache signal improves routing. | — | Principle extraction |
