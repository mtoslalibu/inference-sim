# Problem Statement: Adaptive Routing Algorithm Discovery

## Goal

Discover a routing algorithm for BLIS that **automatically decides which signals to use, how to weight them, and what routing strategy to follow** — without hand-tuned weights, overly complex logic, unrealistic thresholds, or ad hoc rules.

The algorithm replaces the static weighted scorer block in `sim/routing.go` (`WeightedScoring.Route()`).

## Baselines to Beat

### Baseline A: BLIS Static Weighted (3:2:2)
- **Config:** `precise-prefix-cache:3, queue-depth:2, kv-utilization:2`
- **Mechanism:** Linear weighted combination of min-max normalized scorers, argmax selection
- **Weakness:** Fixed weights cannot adapt to changing load conditions. KV-utilization signal is known to be counterproductive under pressure (RP-6 from prior Strategy Evolution). Static weights over-commit to prefix cache scoring even when all instances have equal cache state.

### Baseline B: GLIA HRA (Head-Room Allocator)
- **Config:** `baseline_glia.go` drop-in replacement
- **Mechanism:** Projects per-instance KV headroom after hypothetical request placement. Scores based on projected utilization + queue load, with inadmissibility penalty.
- **Weakness:** No prefix cache awareness at all. Uses only KV utilization + queue depth signals. Relies on stale KV metrics (~5s). Hard-coded parameters (decodeToPromptRatio=0.6, safetyFraction=0.03).

## Hardware & Model Setup

| Parameter | Value |
|-----------|-------|
| Model | `qwen/qwen2.5-7b-instruct` |
| GPU | H100 |
| TP | 1 |
| Latency model | blackbox |
| Instances | 2 |
| Snapshot refresh interval | 5,000,000 us (5s) — stale metrics |
| Cache signal delay | 2,000,000 us (2s) — precise prefix cache signals |

## Constraints on the Evolved Algorithm

1. **Cannot use prefix-affinity scorer** (router-side LRU cache). Can use all other signals: `precise-prefix-cache`, `no-hit-lru`, `queue-depth`, `kv-utilization`, `load-balance`, and any raw snapshot fields.
2. Must fit within `WeightedScoring.Route()` — replace the scoring block, not the entire routing framework.
3. Must be deterministic given the same RNG seed.
4. No external state beyond what's available in `RouterState` and `Request`.
5. Realistic: no oracle access to future arrivals, no unrealistic thresholds.

## Available Signals

| Signal | Source | Freshness | Notes |
|--------|--------|-----------|-------|
| `QueueDepth` | snapshot | ~5s stale | Periodic refresh |
| `BatchSize` | snapshot | ~5s stale | Periodic refresh |
| `InFlightRequests` | snapshot | Synchronous | Updated per routing decision |
| `KVUtilization` | snapshot | ~5s stale | 0.0-1.0, known counterproductive under pressure |
| `FreeKVBlocks` | snapshot | ~5s stale | Absolute free blocks |
| `CacheHitRate` | snapshot | ~5s stale | Cumulative hit rate |
| `EffectiveLoad()` | derived | Mixed | QueueDepth + BatchSize + InFlightRequests |
| `precise-prefix-cache` | cacheQueryFn | ~2s stale | Actual KV cache block hits, min-max normalized |
| `no-hit-lru` | cacheQueryFn | ~2s stale | Cold request → LRU distribution |
| `req.InputTokens` | request | Fresh | Token sequence for cache lookup |
| `req.MaxOutputLen` | request | Fresh | Client budget (not oracle OutputTokens, INV-9) |

## Workload Design

Two workloads, each designed to stress a specific baseline weakness:

### Workload 1: "Prefix-Heavy Burst" (breaks GLIA)
- **Rationale:** GLIA has zero prefix cache awareness. With heavy prefix sharing, a cache-aware router should concentrate prefix-sharing requests on the instance that already has the prefix cached, saving prefill compute. GLIA will scatter them.
- **Design:** High prefix overlap (large prefix-tokens), moderate rate, bursty arrival. ~60s+ sim time.

### Workload 2: "Load Imbalance Stress" (breaks 3:2:2)
- **Rationale:** Static 3:2:2 weights over-commit to prefix cache scoring. Under asymmetric load or when prefix cache is uniformly distributed, the kv-utilization component (known counterproductive per RP-6) can cause pile-on. High rate creates stale-metric-driven routing errors.
- **Design:** High rate with variable request sizes, minimal prefix sharing. ~60s+ sim time. The load creates queue buildup where stale queue-depth signals mislead the static weights.

## Quantitative Success Criteria

| Metric | Target |
|--------|--------|
| E2E mean latency | 30-60% improvement over the worse baseline on each workload |
| E2E P99 latency | 30-60% improvement over the worse baseline on each workload |
| Cross-baseline | At least as good as the better baseline on each workload |
| Robustness | Results hold across 3 seeds (42, 123, 7) |
| Sim duration | Each run >=60s sim time to capture multiple stale signal cycles |

## Prior Knowledge Inventory

From the Strategy Evolution principles catalog (prior work):
- **RP-1:** Orthogonal signals > pre-combined signals
- **RP-6:** KV-utilization scorer is counterproductive under pressure
- **RP-10:** PA:QD safety rule (prefix-affinity weight should not dominate queue-depth too aggressively)
- **S6:** Scheduling is zero-sum at saturation (routing is the lever, not scheduling)
- Winning prior strategy: `pa:4,qd:3` — prefix and load balance are the two essential dimensions

## Seeds

All experiments run with seeds: 42, 123, 7
