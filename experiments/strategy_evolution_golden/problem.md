# Strategy Evolution: Golden Adaptive Router

## Context

Prior experiments established:
1. **v2 adaptive router** beats baseline 2:1:1 on 6/7 FMs with no regressions, using regime detection (drop kvu when cache matters) + load-aware threshold cap
2. **v3 (inflight-requests)** showed synchronous InFlightRequests gives massive TTFT wins (82-97% over v2 on FM-1) but regresses on FM-5 due to min-max normalization over-spreading short requests

Two new GIE/llm-d parity scorers are now available:
- **active-requests** (#957): llm-d's `(maxCount - count) / maxCount` on InFlightRequests. Gentler than min-max — idle instances always score 1.0 regardless of max.
- **running-requests** (#956): GIE's min-max on BatchSize (running/in-batch request count). Captures execution-phase load that QueueDepth misses.

## Hypothesis

A "golden" adaptive router that combines v2's regime detection with all available GIE-parity load signals will:
1. Preserve v2's wins from regime detection (ppc/kvu conflict elimination)
2. Add synchronous load feedback via active-requests (like v3) without v3's FM-5 regression (llm-d formula vs min-max)
3. Use running-requests as additional load dimension for finer-grained spreading

## Scorers Used (All GIE/llm-d Parity)

| Scorer | llm-d/GIE Equivalent | Signal | Freshness |
|--------|---------------------|--------|-----------|
| `precise-prefix-cache` | prefix-scorer | KV cache block hits | Periodic (cache-signal-delay) |
| `load-aware` | load-aware-scorer (#958) | QueueDepth + threshold cap | Periodic (snapshot-refresh) |
| `active-requests` | active-request-scorer (#957) | InFlightRequests | **Synchronous** |
| `running-requests` | running-requests-size-scorer (#956) | BatchSize | Periodic (snapshot-refresh) |
| `kv-utilization` | kv-cache-utilization-scorer | KVUtilization | Periodic (snapshot-refresh) |

## Policies Compared

| Label | Config | Notes |
|-------|--------|-------|
| baseline-211 | ppc:2, qd:1, kvu:1 (weighted) | llm-d production default (QueueDepth-only) |
| adaptive-v2 | `routing: adaptive-v2` | Prior winner: regime detection + load-aware |
| adaptive-golden | `routing: adaptive-golden` | New: regime detection + 5 scorers |

### Golden Regime Table

| Regime | Condition | Weights | Rationale |
|--------|-----------|---------|-----------|
| Cache-affinity | spread > 0.1 | ppc:4, ar:1, la:1 | Strong prefix routing with synchronous + periodic load guards |
| Memory-aware | spread <= 0.1, avgKVUtil > 0.7 | ar:2, la:1, rr:1, kvu:1 | Multi-signal load balancing with KV protection |
| Load-balance | spread <= 0.1, avgKVUtil <= 0.7 | ar:2, la:1, rr:1 | All load signals, synchronous dominant |

Key design choices:
- **ar weight 2x** in load regimes: synchronous signal is most valuable under stale snapshots
- **rr included** in load regimes: captures batch execution load that queue depth misses
- **kvu only in memory-aware**: avoid fighting ppc (lesson from v2)
- **la always present**: threshold cap provides hard overload protection floor

## Experiment Parameters

Same as prior experiments:
- Model: Qwen/Qwen3-32B, Hardware: H100, TP=1, Instances: 4
- Cache signal delay: 2s, Snapshot refresh: 5s
- Latency model: trained-physics
- Seeds: 42, 123, 456
- Total: 3 policies x 7 workloads x 3 seeds = 63 runs

## Success Criteria

1. Golden matches or beats v2 on all 7 FMs
2. Golden beats baseline by >= 25% E2E or >= 30% TTFT on at least 3 FMs
3. Golden never regresses more than -3% vs v2 on any FM
4. Golden captures v3's TTFT wins on FM-1/FM-2a without FM-5 regression
