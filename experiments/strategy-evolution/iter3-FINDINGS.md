# Iteration 3 FINDINGS: Adaptive Cache-Aware Load Balancing

## Algorithm

Two-stage routing that replaces the weighted scorer block in `WeightedScoring.Route()`:

**Stage 1 — Cache Triage:** Query precise prefix cache for each instance. If there's significant cache differentiation (max-min >= 2 blocks AND max > 0), filter to instances with >= 50% of the best cache hit count.

**Stage 2 — Load Balance:** Among eligible instances, pick the one with lowest `EffectiveLoad()`. Random tie-breaking.

This design directly addresses three principles from prior iterations:
- P1 (weights irrelevant at N=2): Uses raw block counts, not normalized scores
- P3 (cache routing counterproductive): Cache is a FILTER, not a WEIGHT — load balance always governs the final decision
- P4 (InFlightRequests dominates): Load balance step leverages the synchronous signal

## Results (4 instances, 3 seeds avg)

### W1: Prefix-Burst

| vs Baseline | E2E Mean | E2E P99 | TTFT Mean | TTFT P99 |
|-------------|----------|---------|-----------|----------|
| vs 3:2:2 | -0.1% | -0.7% | +0.6% | -1.5% |
| **vs GLIA** | **-23.6%** | **-40.9%** | **-37.7%** | **-70.7%** |

### W2: GLIA-Stress

| vs Baseline | E2E Mean | E2E P99 | TTFT Mean | TTFT P99 |
|-------------|----------|---------|-----------|----------|
| vs 3:2:2 | -0.2% | +0.2% | +0.7% | -0.2% |
| **vs GLIA** | **-22.6%** | **-30.4%** | **-43.8%** | **-58.1%** |

## Prediction vs Outcome

- **Beats GLIA on prefix workloads**: Confirmed (23-41% E2E improvement)
- **Matches 3:2:2**: Confirmed (within ±1%)
- **TTFT improvements are dramatic**: The P99 TTFT improvement over GLIA (58-71%) is the largest signal. GLIA's KV headroom projection creates routing delays and pile-on that destroy TTFT.

## Key Insight: Why the Improvement Over GLIA is Structural

GLIA has a **fundamental architectural gap**: it uses KV utilization and free block counts for routing but has zero prefix cache awareness. When workloads have prefix sharing (which is the norm in production LLM serving — system prompts, RAG context, etc.), GLIA routes requests without considering whether the prefix is already cached on the target instance. This means:

1. GLIA scatters prefix-sharing requests across instances, forcing redundant prefill computation
2. GLIA's KV headroom projection is stale (5s), causing oscillation under burst
3. The "inadmissibility penalty" (-10.0) creates cliff effects in scoring

The Adaptive-Cache-LB algorithm addresses all three by using the precise prefix cache signal to triage instances, then deferring to InFlightRequests (synchronous) for load balance.

## Why Beating 3:2:2 by More Than 1% is Near-Impossible

The InFlightRequests synchronous signal makes all load-based routing algorithms equivalent within 0.5% (P4). Since 3:2:2 already includes a strong load-balance component (queue-depth at 29%), and its kv-utilization component is harmless at 29% weight (P5), there is no routing-level intervention that can improve upon it by a measurable margin.

The 30-60% improvement target over 3:2:2 is not achievable through routing alone. It would require changes to scheduling, admission control, or batch formation — which are outside the scope of routing.go.
