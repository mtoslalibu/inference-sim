# Problem Statement: Admission Control Evolution

## Goal

Discover a new admission control algorithm via BLIS simulations that beats the default llm-d admission controller by a large margin in SLO attainment and E2E latency. The winner gets transferred to llm-d as a GAIE Filter plugin.

---

## Baseline: GAIE Legacy Admission

The default admission in production llm-d. One if-statement:

```
if priority < 0 AND saturation >= 1.0 → reject
else → admit
```

Where:
- `saturation = avg(max(queueDepth/5, kvUtil/0.8))` per instance
- 4 SLO tiers: critical (4), standard (3), sheddable (-1), batch (-2)
- critical/standard always admitted, sheddable/batch rejected when saturated

Implementation in BLIS: `sim/admission.go` → `GAIELegacyAdmission` with `SLOTierPriorityGAIE`.

### Why It's Beatable

It's a single binary gate — saturated or not. Problems:

1. **Cliff shedding**: At saturation 1.0, ALL sheddable+batch get rejected instantly. At 0.99, ALL get admitted. No gradual degradation. The 0.99→1.0 gap is where the biggest E2E wins live.
2. **No tier ordering**: `IsSheddable()` is just `priority < 0`. Sheddable (-1) and batch (-2) are shed equally — no "shed batch first, then sheddable" logic.
3. **No tenant fairness**: One heavy tenant saturates the cluster → everyone's sheddable requests get dropped.
4. **No load trajectory**: Doesn't know if load is rising or falling. Same reaction at spike peak and during recovery.
5. **Ignores request size**: Doesn't look at input token length. A 1024-token batch request costs 4x the KV cache of a 256-token one, but both are treated identically.
6. **Fixed threshold**: Average saturation 0.95 might hide one instance at 1.5 and three at 0.77 — `avg` masks hot spots.

---

## Setup

```bash
blis run \
  --model qwen/qwen3-32b \
  --gpu H100 --tp 1 \
  --latency-model trained-physics \
  --num-instances 4 \
  --admission-policy gaie-legacy \
  --snapshot-refresh-interval 50000 \
  --routing-policy round-robin \
  --seed 42
```

Round-robin routing isolates the admission variable — we're not testing routing here. Snapshot refresh at 50ms matches GAIE's actual pod scrape interval (see `signals-and-evolution-point.md` Section 6).

Seeds: 42, 123, 456.

**Capacity reference**: Qwen3-32B on H100 TP=1 handles ~2-4 req/s per instance (trained-physics, BF16, mean input ~256 tokens). 4 instances ≈ 8-16 req/s total capacity depending on token lengths and batching.

---

## Workloads

Each workload: 90+ seconds, all 4 SLO tiers.

**Tier mix** (critical for showing improvement — sheddable+batch must be ≥40% of traffic):

| Tier | Workload 1 | Workload 2 | Workload 3 |
|------|-----------|-----------|-----------|
| critical | 20% | 15% | 20% |
| standard | 35% | 30% | 30% |
| sheddable | 25% | 30% | 30% |
| batch | 20% | 25% | 20% |

### Workload 1: Overload Burst

Steady traffic → 5x burst → recovery. Tests graceful degradation.
Uniform request shapes across tiers (isolates the admission variable).

| Phase | Time | Rate | What happens |
|-------|------|------|-------------|
| Warm-up | 0-15s | 3.5 req/s | System comfortable (~25% util) |
| Ramp | 15-25s | 7 req/s | Approaching saturation |
| Burst | 25-55s | 17.5 req/s (5x) | Overloaded, baseline rejects all sheddable+batch equally |
| Recovery | 55-90s | 3.5 req/s | Draining, baseline stops rejecting |

Use gamma arrivals with high CV (4-8) during burst phase to create micro-bursts that exploit 50ms snapshot staleness. Poisson is too smooth — real traffic is bursty.

**Win condition**: Shed batch first, then sheddable. Protect critical/standard throughout. Higher goodput and lower E2E for admitted requests.

### Workload 2: Multi-Tenant with Varied Shapes

4 tenants, one sends 3x more. Sustained ~1.1 saturation. Tests fairness + token-aware shedding.
Different request shapes per SLO tier (realistic: batch jobs tend to be larger).

| Tier | Typical Input Tokens | Rationale |
|------|---------------------|-----------|
| critical | Short (128-256) | Latency-sensitive, small prompts |
| standard | Medium (256-512) | Normal user traffic |
| sheddable | Medium (256-512) | Best-effort, same as standard |
| batch | Long (1024-2048) | Bulk processing, large contexts |

Tenant distribution: tenant-A 45% (heavy), tenant-B 20%, tenant-C 20%, tenant-D 15%.

| Phase | Time | Rate | What happens |
|-------|------|------|-------------|
| Ramp-up | 0-20s | 6 req/s | All tenants start, light load |
| Sustained | 20-70s | 11 req/s | Heavy tenant at 3x, cluster at ~1.1 saturation |
| Cool-down | 70-90s | 4 req/s | Heavy tenant drops |

**Win condition**: Heavy tenant's sheddable requests shed more aggressively than light tenants'. Higher Jain fairness index. Smarter algo uses input length to shed large batch requests first (baseline ignores token length entirely).

### Workload 3: Oscillating Load

Repeated oscillation between overload and recovery. Tests hysteresis / load trajectory awareness.
The baseline flaps between admit-all and reject-all every cycle — no smoothing.

| Phase | Pattern | Duration |
|-------|---------|----------|
| Warm-up | 4 req/s | 0-10s |
| Oscillation | [5s at 16 req/s, 5s at 2 req/s] × 7 cycles | 10-80s |
| Cool-down | 3 req/s | 80-90s |

Uniform request shapes. Average load across oscillation ≈ 9 req/s (near capacity).

**Win condition**: Evolved algo uses EMA/window to smooth oscillations; baseline flaps between full-admit and full-reject on every cycle. Higher sustained goodput for sheddable+batch, lower TTFT variance.

---

## Evolution Point

The algorithm lives in the `EVOLVE-BLOCK` of `AdaptiveAdmission.Admit()` in `sim/admission.go`.
The struct has mutable state fields for stateful algorithms (you can add new fields as needed):

| Field | Purpose |
|-------|---------|
| `tenantTokens` | Per-tenant token budget tracker |
| `tenantRequests` | Per-tenant request counter |
| `classCounters` | Per-SLO-class admission counter |
| `windowStart` / `windowCount` | Sliding window for rate estimation (compute `arrivalRate = windowCount / (clock - windowStart)`) |
| `totalAdmitted` / `totalRejected` | Running totals |

### Available Signals (transferable to llm-d)

Per-instance (from `state.Snapshots`):

| BLIS | llm-d equivalent | Notes |
|------|-----------------|-------|
| `snap.QueueDepth` | `m.WaitingQueueSize` | Exact match |
| `snap.InFlightRequests` | `m.WaitingQueueSize + m.RunningRequestsSize` | BLIS InFlight = dispatched but not completed (queued + running). NOT the same as `RunningRequestsSize` alone. |
| `snap.KVUtilization` | `m.KVCacheUsagePercent / 100` | Scale difference only |
| `snap.FreeKVBlocks` | derivable from `CacheNumBlocks` + `KVCacheUsagePercent` | BLIS has it directly |

Cluster-wide derived signals (pre-computed in scaffolding):

| Signal | How It's Computed |
|--------|-------------------|
| `numInstances` | `len(state.Snapshots)` |
| `totalInFlight` | Sum of `InFlightRequests` across all instances |
| `totalQueueDepth` | Sum of `QueueDepth` across all instances |
| `maxKVUtil` | Max `KVUtilization` across all instances |
| `avgKVUtil` | Average `KVUtilization` across all instances |
| `minFreeKV` | Minimum `FreeKVBlocks` across all instances |

Per-request:

| BLIS | llm-d equivalent |
|------|-----------------|
| `req.SLOClass` | `request.Headers["x-slo-class"]` |
| `req.TenantID` | `request.Headers["x-tenant-id"]` |
| `len(req.InputTokens)` | parseable from request body |

Plus: `state.Clock` (sim time) → `time.Now()`, `numInstances` → `len(endpoints)`.

**Do NOT use**: `BatchSize`, `CacheHitRate` — these don't exist in llm-d.

### Transfer to llm-d

The winning algorithm becomes a GAIE `scheduling.Filter` plugin:

```go
// In llm-d: implement scheduling.Filter
func (p *Plugin) Filter(ctx, state, request, endpoints) []Endpoint {
    // Read signals from endpoints
    for _, e := range endpoints {
        m := e.GetMetrics()
        // m.RunningRequestsSize, m.WaitingQueueSize, m.KVCacheUsagePercent
    }
    sloClass := request.Headers["x-slo-class"]
    tenantID := request.Headers["x-tenant-id"]

    // YOUR ADMISSION LOGIC HERE
    // return endpoints    → admit
    // return []Endpoint{} → reject
}
```

Register with one line: `plugin.Register("my-admission", Factory)`.

---

## Success Criteria

**Definition**: `goodput = completed_requests_in_tier / total_arrived_in_tier` (admission rate that actually completes, not just admission rate).

### Primary Targets

| Metric | Target | Why achievable |
|--------|--------|---------------|
| Goodput for sheddable+batch | **>30% over baseline** | Baseline cliff-sheds everything equally; graduated shedding alone clears this |
| E2E P99 for critical (burst phase) | **>20% reduction** vs baseline | Early shedding reduces queue depth → lower wait for admitted requests |
| TTFT P99 for standard (burst phase) | **>15% reduction** vs baseline | Same mechanism — shorter queues from proactive shedding |
| Jain fairness index (Workload 2) | **>0.15 improvement** | Tenant-aware shedding vs blind per-cluster average |

### Guard Rails (no regression)

| Metric | Target |
|--------|--------|
| TTFT P99 for critical | Within 5% of baseline |
| TTFT P99 for standard | Within 10% of baseline |
| E2E P99 for critical | Within 5% of baseline |
| Throughput (tokens/sec) | No decrease vs baseline |

### Per-Tier Admission Targets (during overload)

| Tier | Target Admission Rate | Baseline Behavior |
|------|----------------------|-------------------|
| critical | 100% | 100% (always admitted) |
| standard | ≥95% | 100% (always admitted) |
| sheddable | ≥40% | 0% when saturated |
| batch | ≥15% | 0% when saturated |

These encode the ordering: critical > standard > sheddable > batch. The baseline has no ordering — it rejects sheddable and batch identically.

### Must Generalize

Works on all three workloads. Not overfit to one scenario.

---

## Evaluation

Measure per-phase, not just aggregate. The burst/sustained phases are where the evolved algo wins — warm-up and recovery dilute the signal.

| Workload | Key evaluation phase |
|----------|---------------------|
| Workload 1 | Burst phase (25-55s): goodput, E2E, TTFT |
| Workload 2 | Sustained phase (20-70s): fairness, per-tenant admission, token-aware shedding |
| Workload 3 | Oscillation phase (10-80s): goodput stability, TTFT variance |

---

## Constraints

1. Only use transferable signals (see table above)
2. Must not read `Request.OutputTokens` (INV-9)
3. Same seed = same results (deterministic)
4. All runs must satisfy INV-1 (request conservation)

---

## Pre-Experiment Wiring Needed

Before the first evolution iteration can run, these items must be completed:

1. **Integrate `AdaptiveAdmission`** from `experiments/sim2real_admission_results_analysis/best_program.go` into `sim/admission.go` as a new policy alongside existing ones
2. **Register `"adaptive-admission"`** in `validAdmissionPolicies` in `sim/bundle.go`
3. **Add `case "adaptive-admission"`** to `NewAdmissionPolicy` factory and `cluster.go` switch
4. **Fix per-tier rejection recording**: `cluster_event.go` currently only records `shedByTier` for `*sim.TierShedAdmission` (hardcoded type assertion). Must generalize so that `gaie-legacy` and `adaptive-admission` rejections also populate per-tier counts
5. **Create workload YAML specs** for all three workloads with concrete rates, tier mixes, and tenant distributions
6. **Commit** working-tree changes to `sim/admission.go` and `sim/bundle.go` (GAIELegacy addition)
