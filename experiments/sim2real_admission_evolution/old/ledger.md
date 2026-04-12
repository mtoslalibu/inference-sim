# Admission Control Evolution Ledger

## Configuration
- Model: Qwen3-32B on H100 TP=1, trained-physics latency model
- Cluster: 4 instances, round-robin routing, 50ms snapshot refresh
- Workloads: w1 (overload burst, agg_rate=500), w2 (multitenant, agg_rate=500), w3 (oscillating, agg_rate=3000)
- Seeds: 42, 123, 456 (3 seeds per experiment)

## Baseline (GAIE Legacy)
| Workload | Completed | Rejected | E2E P99 ms | TTFT P99 ms | Critical E2E P99 ms | Standard E2E P99 ms | Jain |
|----------|-----------|----------|------------|-------------|---------------------|---------------------|------|
| w1 | 2683 | 3510 | 52,438 | 50,269 | 52,010 | 52,430 | — |
| w2 | 9955 | 3216 | 6,753 | 133 | 3,494 | 6,889 | 0.78 |
| w3 | 11,795 | 539 | 3,471 | 126 | 3,441 | 3,452 | — |

Baseline weakness: binary cliff (reject all sheddable when sat>=1.0), never sheds critical/standard even during extreme overload → 52s P99 on w1.

## Evolution Summary

| Iter | Key Mechanism | w1 E2E P99 Δ% | w2 E2E P99 Δ% | w3 E2E P99 Δ% | w1 Crit P99 Δ% | w2 Crit P99 Δ% | Winner? |
|------|---------------|----------------|----------------|----------------|-----------------|-----------------|---------|
| 1 | QD-proportional shedding, all tiers | **-63.0%** | +12.0% | +1.1% | **-59.0%** | +126.6% | Best w1, regresses w2 |
| 2 | EMA-smoothed QD, higher thresholds | -50.6% | +36.5% | +1.5% | -45.7% | +182.7% | Still regresses w2 |
| 3 | Never shed critical, aggressive sheddable | -1.7% | **-2.2%** | **-3.6%** | +4.2% | **-2.3%** | Best w2/w3, no w1 gain |
| 4 | Dual-gate (sat AND qd), EMA=0.15 | -22.8% | -2.1% | -3.6% | -16.6% | -2.2% | Good balance |
| **5** | **Fast EMA + instantaneous max, dual-gate** | **-20.0%** | **-2.2%** | **-3.8%** | **-15.1%** | **-2.1%** | **Best balanced** |

## Iteration 5 (Winner) — Detailed Results

### w1_overload_burst
| Metric | Baseline | Adaptive | Delta |
|--------|----------|----------|-------|
| E2E Mean (ms) | 8,678 | 7,343 | **-15.4%** |
| E2E P99 (ms) | 52,438 | 41,957 | **-20.0%** |
| TTFT Mean (ms) | 5,358 | 3,899 | **-27.2%** |
| TTFT P99 (ms) | 50,269 | 38,801 | **-22.8%** |
| Standard E2E P99 | 52,430 | 40,485 | **-22.8%** |
| Critical E2E P99 | 52,010 | 44,163 | **-15.1%** |

### w2_multitenant_varied
| Metric | Baseline | Adaptive | Delta |
|--------|----------|----------|-------|
| E2E Mean (ms) | 3,782 | 3,651 | **-3.5%** |
| E2E P99 (ms) | 6,753 | 6,605 | **-2.2%** |
| TTFT P99 (ms) | 133 | 96 | **-27.7%** |
| Critical E2E P99 | 3,494 | 3,422 | **-2.1%** |
| Standard E2E P99 | 6,889 | 6,761 | **-1.9%** |
| Jain Fairness | 0.78 | 0.80 | **+2.6%** |
| Standard shed | 0 | 0 | No regression |

### w3_oscillating
| Metric | Baseline | Adaptive | Delta |
|--------|----------|----------|-------|
| E2E Mean (ms) | 2,142 | 2,073 | **-3.2%** |
| E2E P99 (ms) | 3,471 | 3,340 | **-3.8%** |
| TTFT P99 (ms) | 126 | 117 | **-7.2%** |
| Critical E2E P99 | 3,441 | 3,332 | **-3.2%** |
| Standard E2E P99 | 3,452 | 3,342 | **-3.2%** |

## Key Principles Discovered

1. **Dual-gate for tier protection**: Shedding critical/standard requires BOTH saturation AND queue depth to exceed thresholds. This prevents over-shedding in moderate load (w2 has high saturation from KV but low QD).

2. **Sheddable-first aggressive shedding**: Single-signal (saturation-only) shedding for sheddable tier starting at sat=0.5 frees capacity for critical/standard.

3. **Max(instantaneous, EMA) for burst responsiveness**: EMA alone is too slow for sharp bursts. Taking the maximum of instantaneous and EMA signals catches both burst onset and sustained overload.

4. **Graduated probability, not cliff**: P(reject) = f(excess) instead of binary yes/no avoids the baseline's "all or nothing" behavior.

5. **Tenant fairness via shed probability bonus**: Over-represented tenants (>1.5x average requests) face higher shed probability, not hard caps.

## Transfer Path to llm-d

The winning algorithm maps to a GAIE `scheduling.Filter` plugin:
- **Signals needed**: per-endpoint queue depth, KV cache utilization (already available in GAIE metrics)
- **State needed**: EMA accumulator (per-filter instance), tenant request counters
- **Decision**: Return all endpoints = admit, empty = reject
- **Implementation**: ~100 lines of Go in a new `AdaptiveFilter` struct
