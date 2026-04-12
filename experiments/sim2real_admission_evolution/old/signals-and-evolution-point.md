# Signals & Evolution Point for Admission Control

Where do we evolve admission logic in BLIS, what signals can we use, and do they match llm-d?

---

## 1. Where to Evolve in BLIS

The evolution point is the `EVOLVE-BLOCK` inside `AdaptiveAdmission.Admit()` in `sim/admission.go` (see `best_program.go` for the current best).

```go
func (a *AdaptiveAdmission) Admit(req *Request, state *RouterState) (bool, string) {
    // --- Derived signals (fixed, computed before the block) ---
    numInstances, totalInFlight, totalQueueDepth, maxKVUtil, avgKVUtil,
    minFreeKV, inputLen, sloClass, tenantID, clock  // all pre-computed

    // EVOLVE-BLOCK-START
    //   YOUR LOGIC HERE — use any signal above + stateful fields on `a`
    //   return true, ""        → admit
    //   return false, "reason" → reject
    // EVOLVE-BLOCK-END
}
```

Everything **above** the EVOLVE-BLOCK is fixed scaffolding (signal extraction). Everything **inside** the block is what gets evolved/replaced. The struct also has mutable state fields (`tenantTokens`, `tenantRequests`, `classCounters`, `windowStart`, etc.) for stateful algorithms.

---

## 2. Signals Available in BLIS

### Per-Instance Signals (from `RoutingSnapshot`)

These come from `state.Snapshots` — one per instance:

| Signal | Type | What It Is |
|--------|------|------------|
| `QueueDepth` | int | Requests waiting in instance queue |
| `BatchSize` | int | Requests currently in the running batch |
| `KVUtilization` | float64 | KV cache usage ratio (0.0 - 1.0) |
| `FreeKVBlocks` | int64 | Free KV cache blocks on the instance |
| `InFlightRequests` | int | Requests dispatched but not yet completed |
| `CacheHitRate` | float64 | Prefix cache hit rate |
| `Model` | string | Which model this instance serves |
| `GPUType` | string | GPU hardware type (e.g. "H100") |
| `TPDegree` | int | Tensor parallelism degree |

### Cluster-Wide Derived Signals (pre-computed in scaffolding)

| Signal | How It's Computed |
|--------|-------------------|
| `numInstances` | `len(state.Snapshots)` |
| `totalInFlight` | Sum of `InFlightRequests` across all instances |
| `totalQueueDepth` | Sum of `QueueDepth` across all instances |
| `maxKVUtil` | Max `KVUtilization` across all instances |
| `avgKVUtil` | Average `KVUtilization` across all instances |
| `minFreeKV` | Minimum `FreeKVBlocks` across all instances |

### Per-Request Signals

| Signal | What It Is |
|--------|------------|
| `inputLen` | Number of input tokens (`len(req.InputTokens)`) |
| `sloClass` | SLO tier: "critical", "standard", "sheddable", "batch", "background" |
| `tenantID` | Tenant identifier |
| `clock` | Current simulation time (microseconds) |

### Stateful Fields (on the `AdaptiveAdmission` struct)

| Field | Purpose |
|-------|---------|
| `tenantTokens` | Per-tenant token budget tracker |
| `tenantRequests` | Per-tenant request counter |
| `classCounters` | Per-SLO-class admission counter |
| `windowStart` / `windowCount` | Sliding window for rate estimation |
| `totalAdmitted` / `totalRejected` | Running totals |

---

## 3. Signals Available in llm-d (Filter Plugin)

Inside a `scheduling.Filter`, you get `[]scheduling.Endpoint`. Each endpoint has `GetMetrics()`:

| llm-d Signal | Type | What It Is |
|-------------|------|------------|
| `RunningRequestsSize` | int | Requests currently running on pod |
| `WaitingQueueSize` | int | Requests waiting in pod queue |
| `KVCacheUsagePercent` | float64 | KV cache usage (0-100 scale, not 0-1) |
| `KvCacheMaxTokenCapacity` | int | Max tokens the KV cache can hold |
| `CacheBlockSize` | int | Tokens per KV cache block |
| `CacheNumBlocks` | int | Total GPU blocks for KV cache |
| `UpdateTime` | time.Time | When metrics were last refreshed |
| `ActiveModels` | map | Models loaded on GPU |
| `MaxActiveModels` | int | Max models that can be loaded |

From the request: `request.Headers["x-slo-class"]`, `request.Headers["x-tenant-id"]`.

---

## 4. Signal Mapping: Do They Match?

| BLIS Signal | llm-d Signal | Match? |
|-------------|-------------|--------|
| `snap.QueueDepth` | `m.WaitingQueueSize` | **Yes** — same concept |
| `snap.InFlightRequests` | `m.WaitingQueueSize + m.RunningRequestsSize` | **Close** — BLIS InFlight = dispatched-to-completion (queued + running). NOT the same as `RunningRequestsSize` alone, which is only currently-executing requests. In llm-d, combine both to get the BLIS equivalent. |
| `snap.KVUtilization` (0-1) | `m.KVCacheUsagePercent` (0-100) | **Yes** — divide by 100 |
| `snap.FreeKVBlocks` | `m.CacheNumBlocks` - derived from `KVCacheUsagePercent` | **Close** — BLIS has it directly, llm-d needs math |
| `snap.BatchSize` | *Not available* | **Gap** — llm-d doesn't expose current batch size |
| `snap.CacheHitRate` | *Not available as metric* | **Gap** — llm-d uses scorer-based prefix matching instead |
| `req.SLOClass` | `request.Headers["x-slo-class"]` | **Yes** |
| `req.TenantID` | `request.Headers["x-tenant-id"]` | **Yes** |
| `len(req.InputTokens)` | *Not directly in Filter* | **Gap** — need to parse from request body or use PrepareData plugin |
| `state.Clock` | `time.Now().UnixMicro()` | **Yes** — different source, same concept |
| `len(state.Snapshots)` | `len(endpoints)` | **Yes** |

**Bottom line**: Queue depth and KV util match exactly. InFlightRequests maps to the *sum* of WaitingQueueSize + RunningRequestsSize (not RunningRequestsSize alone — that's only running, not queued). BatchSize and CacheHitRate don't transfer directly. The signals used in practice (queue depth, KV util, SLO class, tenant ID) are all available on both sides.

---

## 5. Can You Use the Saturation Detector Inside an Admission Plugin?

### In BLIS: Yes, but it's not wired that way today

The saturation detectors (`UtilizationDetector`, `ConcurrencyDetector`) take a `*RouterState` and return a float64. The `AdaptiveAdmission.Admit()` method also receives `*RouterState`. So you **can** call the saturation detector inside the EVOLVE-BLOCK:

```go
// Inside EVOLVE-BLOCK — you could do this:
sat := computeSaturation(state)  // same formula as UtilizationDetector
if sat >= 1.0 && sloClass == "sheddable" {
    return false, "saturated"
}
```

You don't even need to import the detector — just recompute the same formula inline:

```go
// Inline utilization saturation (same as GAIE default)
satSum := 0.0
for _, snap := range state.Snapshots {
    qRatio := float64(snap.QueueDepth) / 5.0         // queueDepthThreshold
    kvRatio := snap.KVUtilization / 0.8               // kvCacheUtilThreshold
    satSum += math.Max(qRatio, kvRatio)
}
saturation := satSum / float64(numInstances)
```

All the data is already there in `state.Snapshots`. The saturation detector is just a formula over those same signals.

### In llm-d: Yes, same idea

Inside the Filter plugin, you have `[]scheduling.Endpoint`, each with `GetMetrics()`. You can compute the exact same saturation formula:

```go
// Inside Filter() — same formula as GAIE's utilization detector
satSum := 0.0
for _, e := range endpoints {
    m := e.GetMetrics()
    qRatio := float64(m.WaitingQueueSize) / 5.0
    kvRatio := (m.KVCacheUsagePercent / 100.0) / 0.8
    satSum += math.Max(qRatio, kvRatio)
}
saturation := satSum / float64(len(endpoints))
```

You don't need to call the actual `SaturationDetector` plugin — you just replicate the formula. The metrics are the same ones the detector reads.

### So the answer is: yes on both sides

You can compute saturation inside the admission/filter logic on both BLIS and llm-d, using the same formula and the same underlying signals. The saturation detector isn't a separate service — it's just a function over endpoint metrics that are already available to you.

---

## 6. Metrics Freshness: Scrape Interval vs Staleness Guard vs EPP Self-Publishing

GAIE has **three** separate timing parameters that are easy to confuse (defaults from `pkg/epp/server/options.go`):

| Parameter | Default | CLI Flag | What It Does |
|-----------|---------|----------|--------------|
| `RefreshMetricsInterval` | **50ms** | `--refresh-metrics-interval` | How often the data layer collector scrapes each pod's `/metrics` |
| `MetricsStalenessThreshold` | **2s** | `--metrics-staleness-threshold` | If last successful scrape > this, treat pod as saturated |
| `RefreshPrometheusMetricsInterval` | **5s** | `--refresh-prometheus-metrics-interval` | How often EPP publishes its own aggregated Prometheus gauges |

### How these relate — the data flow

```
Pod vLLM /metrics  --[50ms scrape]--> in-memory Metrics struct ---> scorers, filters, saturation detector
                     (RefreshMetricsInterval)       |
                                               [5s aggregate]
                                   (RefreshPrometheusMetricsInterval)
                                                    |
                                                    v
                                           EPP's own /metrics  ---> Grafana/dashboards (monitoring EPP)
```

- **50ms** (`RefreshMetricsInterval`): The data layer collector (`datalayer.NewRuntime(pollingInterval)`) scrapes each pod's vLLM `/metrics` endpoint on this timer. The result updates the in-memory `Metrics` struct (`WaitingQueueSize`, `KVCacheUsagePercent`, `UpdateTime`). **This is what scorers, filters, and the saturation detector read.** This is the signal freshness that matters for routing and admission decisions.

- **2s** (`MetricsStalenessThreshold`): Safety guard. If `time.Since(metrics.UpdateTime) > 2s`, the utilization detector treats that pod as score 1.0 (saturated). This catches pod crashes, network blips, or stalled metrics endpoints. It does NOT affect normal operation — with 50ms scrapes, metrics are always <50ms old.

- **5s** (`RefreshPrometheusMetricsInterval`): EPP's own observability. Every 5s, the `StartMetricsLogger` reads the already-collected pod metrics and publishes **aggregated cluster-level gauges** (avg KV cache, avg queue size, ready pod count) to EPP's own Prometheus endpoint. This is for Grafana dashboards and alerting about EPP health. **Scorers and admission never read this.** It has zero effect on routing or admission decisions.

### Correction: sim2real_golden used the wrong interval

The `experiments/sim2real_golden/compare.sh` used `SNAPSHOT_REFRESH=5000000` (5s) with the comment "production-like staleness." This was likely matched to `RefreshPrometheusMetricsInterval` (5s) — but that's the **EPP self-publishing** interval, not the pod scrape interval. The actual signal freshness that scorers see is governed by `RefreshMetricsInterval` = **50ms**.

**The 5s setting was not fine for routing either.** It made all routers artificially worse — `queue-depth` and `load-aware` scorers were blind for 5s at a time, causing pile-on that wouldn't happen in real production with 50ms scraping. The baseline was especially handicapped since its only load-balancing signal (`queue-depth:1`) was stale. The adaptive router results are directionally valid but the magnitudes may change with correct scraping. See `experiments/sim2real_golden/extra/scrape-interval-correction.md` for the full correction and rerun plan.

For **admission experiments**, the impact is even more severe. With 5s staleness and a 2s staleness threshold, every pod would appear stale on most decisions if we modeled the staleness guard — the saturation detector would see 100% saturation almost constantly, rejecting all sheddable requests.

### How to model this in BLIS

BLIS models scrape frequency via `--snapshot-refresh-interval` (microseconds). For GAIE parity:

```bash
# Match GAIE's 50ms pod scrape interval (correct for admission parity)
blis run --model qwen/qwen3-32b --snapshot-refresh-interval 50000
```

When set, `QueueDepth`, `BatchSize`, and `KVUtilization` in `RoutingSnapshot` are only refreshed every 50ms of simulated time. Between refreshes, the admission policy (and router) see stale values — same effect as GAIE seeing slightly-old Prometheus scrapes.

**What BLIS does NOT model**: the "if metrics are >2s old, treat as saturated" fallback. In simulation there are no real scrape failures — the `CachedSnapshotProvider` always returns a value (possibly stale, but never missing). To model pod-crash / metrics-dark scenarios, we'd need to add scrape failure injection, which doesn't exist today. For normal admission control experiments, the periodic staleness via `--snapshot-refresh-interval` is sufficient.

### Recommended settings for sim2real experiments

| Scenario | `--snapshot-refresh-interval` | Notes |
|----------|-------------------------------|-------|
| Oracle (no staleness) | `0` (default) | Live signals, no delay |
| GAIE production parity | `50000` (50ms) | Matches `RefreshMetricsInterval` default |
| Pessimistic / slow scraping | `500000` (500ms) | Models overloaded metrics endpoint |
| Extreme staleness | `2000000` (2s) | At the `MetricsStalenessThreshold` — every decision stale |
| sim2real_golden setting | `5000000` (5s) | Matched wrong parameter; OK for routing stress, too stale for admission |

---

## 7. Summary

| Question | Answer |
|----------|--------|
| Where to evolve in BLIS? | `EVOLVE-BLOCK` in `AdaptiveAdmission.Admit()`, `sim/admission.go` |
| What signals? | In-flight, queue depth, KV util, free KV blocks, SLO class, tenant ID, input length, clock, plus stateful counters |
| Do signals match llm-d? | Core signals (in-flight, queue depth, KV util) match exactly. BatchSize and CacheHitRate are BLIS-only. |
| Can I use saturation in admission? | Yes on both sides — just compute the formula inline from the same endpoint metrics |
| Anything to avoid? | Don't rely on `BatchSize` or `CacheHitRate` in evolved logic if you want clean transfer to llm-d |
