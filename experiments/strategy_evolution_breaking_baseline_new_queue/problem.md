# Strategy Evolution: Breaking Baseline with Load-Aware Scorer

## Context

PR #960 changes the `queue-depth` scorer from reading `EffectiveLoad()` (QueueDepth + BatchSize + InFlightRequests) to reading `QueueDepth` only, achieving GIE parity with llm-d's real `queue-scorer`. This removes the synchronous `InFlightRequests` feedback that previously prevented pile-on under stale snapshots.

Prior experiment (`strategy_evolution_breaking_baseline/`) demonstrated a three-regime adaptive router beating the static 2:1:1 baseline by 26-54% on failure-mode workloads. That router used `queue-depth` (with `EffectiveLoad()`). Under PR #960, both the baseline and adaptive-v1 lose synchronous feedback.

## Problem Statement

With QueueDepth-only scoring under 5s snapshot refresh, **all instances report identical queue depths within a refresh window**. Min-max normalization produces equal scores, eliminating load differentiation. Requests pile onto one instance per window, causing:

- KV cache exhaustion and preemption cascades (FM-1: TTFT P99 from 1.3s to 37.8s)
- Near-livelock under tight KV budgets (issue #963)
- Complete loss of queue-depth signal value

## Hypothesis

The llm-d `load-aware` scorer (#958) provides threshold-capped linear scoring that survives stale snapshots:

- **Formula**: empty queue = 0.5, otherwise `0.5 * (1 - min(QueueDepth, 128) / 128)`
- **Score range**: [0.0, 0.5]
- **Key property**: Once `QueueDepth >= 128`, score = 0 (hard overload protection). This absolute threshold means an overloaded instance scores 0 regardless of stale snapshot equalization.

An `adaptive-v2` router replacing `queue-depth` with `load-aware` in the three-regime architecture will:
1. Beat the 2:1:1 baseline (which also uses stale `queue-depth`) by >= 25% E2E on at least 3 FMs
2. Match or beat adaptive-v1 (which now also suffers from stale `queue-depth`)
3. Never regress more than -3% on any FM

## Scorers Used (All GIE/llm-d Parity)

| Scorer | llm-d Equivalent | Signal | Freshness |
|--------|------------------|--------|-----------|
| `precise-prefix-cache` | `prefix-scorer` | KV cache block hits | Periodic (cache-signal-delay) |
| `queue-depth` | `queue-scorer` (PR #960) | QueueDepth only | Periodic (snapshot-refresh) |
| `kv-utilization` | `kv-cache-utilization-scorer` | KVUtilization | Periodic (snapshot-refresh) |
| `load-aware` | `load-aware-scorer` (#958) | QueueDepth + threshold cap | Periodic (snapshot-refresh) |

**No BLIS-only scorers** (e.g., `load-balance` which reads `EffectiveLoad()`) are used.

## Policies Compared

| Label | Config | Notes |
|-------|--------|-------|
| `baseline-211` | ppc:2, qd:1, kvu:1 (weighted) | llm-d production default, now with QueueDepth-only |
| `adaptive-v1` | `routing: adaptive` | Prior three-regime, uses qd internally (also affected) |
| `adaptive-v2` | `routing: adaptive-v2` | New three-regime using load-aware instead of qd |

### Adaptive-v2 Regime Table

| Regime | Condition | Weights | Rationale |
|--------|-----------|---------|-----------|
| Cache-affinity | spread > 0.1 | ppc:4, la:1 | Strong prefix routing; la provides mild load signal |
| Memory-aware | spread <= 0.1, avgKVUtil > 0.7 | la:1, kvu:1 | No prefix to exploit; avoid KV exhaustion |
| Load-balance | spread <= 0.1, avgKVUtil <= 0.7 | la:1 | Pure load distribution with threshold protection |

## Workloads (7 Failure Modes)

Reused from prior experiment — these stress-test exactly the dimensions where stale snapshots hurt:

- **FM-1**: Prefix Pile-On (1 prefix group, 4 instances)
- **FM-2a**: Groups > Instances (6 prefix groups, 4 instances)
- **FM-2b**: Groups < Instances (2 prefix groups, 4 instances)
- **FM-3**: Burst Absorption (10x burst at t=5s)
- **FM-4**: Multi-Regime Phased (prefix-heavy then cold traffic)
- **FM-5**: Short Output Classification (low output, high prefix reuse)
- **FM-6**: Cold Traffic Under KV Pressure (zero prefix sharing)

## Experiment Parameters

- Model: Qwen/Qwen3-32B, Hardware: H100, TP=1
- Instances: 4
- Cache signal delay: 2s (2,000,000 us)
- Snapshot refresh: 5s (5,000,000 us)
- Latency model: trained-physics
- Seeds: 42, 123, 456 (3 per config)
- Total runs: 3 policies x 7 workloads x 3 seeds = 63

## Success Criteria

1. adaptive-v2 beats baseline-211 by >= 25% E2E or >= 30% TTFT on at least 3 FMs
2. adaptive-v2 never worse than -3% on any FM
3. adaptive-v2 beats adaptive-v1 on FMs where stale queue-depth hurts most
4. Self-contained bundle reproduces results

## Iteration Plan

- **Iter 1**: Run all 63 experiments, collect results table
- **Iter 2**: If adaptive-v2 regresses on any FM, tune thresholds or load-aware weight
- **Iter 3**: Finalize FINDINGS.md, create bundle
