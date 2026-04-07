# Problem: Where Does Static 2:1:1 Routing Break?

## Motivation

Prior strategy evolution experiments (strategy-evolution, strategy-evolution-2_1_1) discovered a
**structural ceiling of ~1-3% E2E improvement** over the static 2:1:1 baseline
(`precise-prefix-cache:2, queue-depth:1, kv-utilization:1`). However, those experiments may have
been testing workloads where 2:1:1 *already works well*:

- **Modest prefix diversity** — workloads had 1-2 prefix groups, not enough to stress
  the cache-affinity vs load-balance tradeoff.
- **No dominant prefix group** — traffic was spread across cohorts, so the ppc scorer
  never strongly favored one instance over another.
- **Steady-state workloads** — no regime shifts that would expose static weight rigidity.
- **Symmetric instances** — homogeneous hardware with even traffic produces identical signals.

This experiment asks: **under what conditions does 2:1:1 structurally fail, and by how much?**
We are not yet proposing an adaptive replacement — we first need to map the failure surface.

## Baseline Under Study

**Static 2:1:1 weighted scoring**: `precise-prefix-cache:2, queue-depth:1, kv-utilization:1`

This is the llm-d production configuration. The `precise-prefix-cache` scorer queries actual
instance KV cache state (with 2s signal delay). Scores are min-max normalized, linearly
combined, argmax selected.

## Hypothesized Failure Modes

### FM-1: Prefix Pile-On (Dominant Prefix Group)

**Thesis:** When one prefix group dominates traffic (>=70%), the `ppc:2` weight actively routes
most requests to whichever instance cached that prefix. The `qd:1` counterweight cannot
overcome the 2x prefix score, especially with stale queue-depth (5s refresh). This creates
self-reinforcing pile-on: the loaded instance stays "hot" in cache, attracting even more traffic.

**Why 2:1:1 fails here:** The prefix cache scorer returns 1.0 for the instance with the cached
prefix and 0.0 for others (min-max normalized). With weight 2, this contributes 2.0 to one
instance vs 0.0 to others. Queue-depth at weight 1 can only counterbalance if the loaded
instance has a normalized queue depth of 1.0 — but with stale signals, it doesn't reflect
the pile-on in time.

**Control:** Pure load-balance (`queue-depth:1` only) should not pile on.

### FM-2: Prefix Groups vs Instance Count Mismatch

**Thesis:** When the number of active prefix groups exceeds the instance count (e.g., 5 groups
across 4 instances, or 7 groups across 4), no clean partition exists. The cache scorer creates
unstable affinity — requests from the same prefix group oscillate between instances as cache
state shifts, causing unnecessary prefill recomputation.

With fewer groups than instances (e.g., 2 groups across 4 instances), 2:1:1 concentrates
traffic on 2 instances while the other 2 idle — the cache scorer has no reason to send traffic
to instances without cached prefixes.

**Why 2:1:1 fails here:** The scorer picks whichever instance has the most blocks cached for a
given prefix, but when groups outnumber instances, the "best" instance changes as cache fills.
When instances outnumber groups, excess capacity is wasted. Static weights cannot adapt the
cache-vs-load tradeoff to the group/instance ratio.

**Control:** Round-robin or pure load-balance distributes evenly regardless of prefix topology.

### FM-3: Burst Absorption Under Stale Signals

**Thesis:** During a sudden traffic spike, stale signals (queue-depth at 5s, cache at 2s) are
wrong. The prefix scorer keeps pointing at an already-overloaded instance because it cached the
right prefix 2 seconds ago. By the time the queue-depth signal catches up, the damage is done —
one instance has a massive queue while others idle.

**Why 2:1:1 fails here:** InFlightRequests is synchronous and helps via the `queue-depth`
scorer, but at weight 1 it's overpowered by `ppc:2` during bursts when all requests share a
prefix. The pile-on happens in the gap between burst arrival and queue signal refresh.

**Control:** Pure load-balance reacts instantly via InFlightRequests without stale-signal lag.

### FM-4: Multi-Regime / Phased Workloads

**Thesis:** A workload that shifts between distinct phases (prefix-heavy -> cold -> mixed ->
prefix-heavy with different group) needs different weight profiles at each phase. Static 2:1:1
pays the "prefix tax" during cold phases (the scorer queries cache and routes based on stale
hits from the previous phase) and may under-weight cache during prefix-heavy phases.

**Why 2:1:1 fails here:** The weights are a compromise that's suboptimal for every regime.
During cold phases, `ppc:2` is dead weight (all cache scores are 0 or equal). During
prefix-heavy phases with a dominant group, `ppc:2` causes pile-on (FM-1). The static policy
can't shift between "chase cache" and "spread load" as conditions change.

**Control:** Phase-aware oracle (simulated by running the best policy per phase) establishes
the improvement ceiling. Pure load-balance is regime-agnostic.

## Model & Hardware Setup

| Parameter | Value |
|-----------|-------|
| Model | `qwen/qwen3-14b` |
| GPU | H100 |
| TP | 1 |
| Latency model | trained-physics |
| Snapshot refresh interval | 5,000,000 us (5s) |
| Cache signal delay | 2,000,000 us (2s) |
| Block size | 16 tokens |
| Total KV blocks | 17,600 (from defaults.yaml — much smaller than qwen2.5-7b's 67,659) |

**Why trained-physics + qwen3-14b:** Previous experiments used blackbox with qwen2.5-7b where
`alpha[1]=0.0` — queue depth had zero TTFT impact and the 67,659-block cache absorbed all
prefixes. Trained-physics computes TTFT from actual prefill FLOPs (β₁ₐ=0.152 correction on
roofline compute), so cache hits that skip prefill tokens directly reduce TTFT. Qwen3-14b is
a larger model (longer prefill, more cache benefit per hit) with only 17,600 KV blocks — not
enough for all instances to cache all prefix groups simultaneously. This breaks through two
factors that created the prior ~1% ceiling: (1) TTFT now responds to routing decisions, and
(2) cache differentiation persists across instances.

## Comparison Policies

For each failure mode workload, compare:

| Label | Config | Rationale |
|-------|--------|-----------|
| **baseline-211** | `ppc:2, qd:1, kvu:1` | The policy under study |
| **lb-only** | `queue-depth:1` | Pure load-balance, no cache awareness |
| **no-kvu** | `ppc:2, qd:1` | Tests RP-6 (remove kvu noise) |
| **ppc-heavy** | `ppc:4, qd:1` | More aggressive cache chasing |
| **qd-heavy** | `ppc:1, qd:4` | Load-balance dominates cache |

## Experimental Controls

1. **Instance count varies per FM:** FM-1 (4, 8), FM-2 (4 with 2/5/7 prefix groups),
   FM-3 (4), FM-4 (4).
2. **Signal staleness:** 5s snapshot refresh, 2s cache signal delay (production realistic).
3. **Seeds:** 42, 123, 456 (3 seeds for cross-seed consistency).
4. **Each BLIS run is fully independent** — no shared state between runs.
5. **Default KV cache size** — no `--total-kv-blocks` override.

## Success Criteria

This is a **diagnostic experiment**, not an optimization experiment. Success means:

1. **Identify at least 1 workload** where 2:1:1 loses >=15% E2E (mean or P99) vs the best
   comparison policy. Bonus: >=25% gap.
2. **Root-cause each failure** with file:line evidence linking the mechanism to the scorer
   framework code.
3. **Map the failure surface**: which parameters (instance count, prefix distribution, burst
   intensity, regime shift timing) flip 2:1:1 from "fine" to "broken."
4. **Produce actionable findings** for designing an adaptive policy in a follow-up experiment.

## Prior Knowledge

From strategy-evolution experiments:
- **P1:** Min-max normalization makes weights irrelevant at N=2 (binary signals)
- **P3:** Cache-aware routing counterproductive when used as weight (causes pile-on)
- **P4:** InFlightRequests dominates; stale signals add <0.5%
- **P7:** Routing opportunity ceiling ~1% with oversized cache
- **P9:** Cache signal toxic when fresh (pile-on), useless when stale (no differentiation)
- **RP-6:** kv-utilization counterproductive under low utilization

**Key gaps in prior work:**
1. Findings P7 and P9 were established with oversized KV cache (67,659 blocks) where all
   instances cached all prefixes. Qwen3-14b has only 17,600 blocks — persistent cache
   differentiation between instances becomes possible.
2. Blackbox alpha[1]=0 meant TTFT was insensitive to routing. Trained-physics computes
   TTFT from actual prefill FLOPs, so cache hits directly reduce TTFT.
3. Prior workloads had modest prefix diversity. This experiment tests dominant groups,
   group/instance mismatches, and phased regime shifts.
