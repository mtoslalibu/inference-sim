# Default (Legacy) Admission Controller — Deep Dive

How does the default admission controller work in llm-d/GAIE, how does BLIS model it, and what's missing?

---

## 1. What It Does (Plain English)

The default admission controller in GAIE is dead simple:

> When a request arrives, check two things: (1) is the system saturated? (2) is the request sheddable? If **both** are true, reject it with HTTP 429. Otherwise, let it through.

That's it. Non-sheddable requests **always** pass through, even if the system is on fire. The controller never queues, never delays, never does anything clever — it's a binary gate.

---

## 2. Where It Lives in Code (GAIE)

### The Controller

**File:** `pkg/epp/requestcontrol/admission.go`

```go
// The struct — just holds a saturation detector and endpoint locator
type LegacyAdmissionController struct {
    saturationDetector flowcontrol.SaturationDetector
    endpointCandidates contracts.EndpointCandidates
}

// Admit() — the entire logic
func (lac *LegacyAdmissionController) Admit(ctx, reqCtx, priority) error {
    return rejectIfSheddableAndSaturated(ctx, lac.saturationDetector, ...)
}
```

The actual decision happens in one function:

```go
func rejectIfSheddableAndSaturated(..., priority int) error {
    if IsSheddable(priority) {
        if sd.Saturation(ctx, endpoints) >= 1.0 {
            return Error{Code: ResourceExhausted, Msg: "system saturated, sheddable request dropped"}
        }
    }
    return nil  // admit
}
```

Two checks, one if-statement. That's the whole controller.

### What Makes a Request "Sheddable"?

**File:** `pkg/epp/util/request/sheddable.go`

```go
func IsSheddable(priority int) bool {
    return priority < 0
}
```

Priority comes from the InferenceObjective CRD. If a request's priority is negative, it's sheddable. Zero and above are protected — they always get in.

### How Saturation Is Computed

**File:** `pkg/epp/framework/plugins/flowcontrol/saturationdetector/utilization/detector.go`

The **default** detector is utilization-based (not concurrency). It uses two signals per pod:

```go
func (d *Detector) Saturation(ctx, candidates []Endpoint) float64 {
    var totalScore float64
    for _, e := range candidates {
        metrics := e.GetMetrics()

        // Safety: if metrics are stale (>200ms old), treat pod as saturated
        if metrics == nil || time.Since(metrics.UpdateTime) > d.config.MetricsStalenessThreshold {
            totalScore += 1.0
            continue
        }

        qRatio  := float64(metrics.WaitingQueueSize) / float64(d.config.QueueDepthThreshold)
        kvRatio := metrics.KVCacheUsagePercent / d.config.KVCacheUtilThreshold

        totalScore += max(qRatio, kvRatio)  // whichever is worse
    }
    return totalScore / float64(len(candidates))
}
```

In plain English:
- For each pod: take the **worse** of (queue fullness, KV cache fullness)
- Average that across all pods
- If the average >= 1.0, system is "saturated"

Defaults: `queueDepthThreshold = 5`, `kvCacheUtilThreshold = 0.8` (80%).

### How Metrics Arrive

Metrics come from **Prometheus scraping**. A collector goroutine does HTTP GET to each pod's `/metrics` endpoint on a timer. The `WaitingQueueSize` and `KVCacheUsagePercent` fields are extracted from vLLM's Prometheus output. There's a staleness guard — if metrics are older than 200ms, the pod is treated as saturated (safe default).

### Where It's Wired as Default

**File:** `cmd/epp/runner/runner.go`

```go
} else {
    setupLog.Info("Experimental Flow Control layer is disabled, using legacy admission control")
    admissionController = requestcontrol.NewLegacyAdmissionController(saturationDetector, endpointCandidates)
}
```

Flow control is off by default → legacy controller is what runs in production.

### What the Client Sees on Rejection

- **HTTP 429** (Too Many Requests)
- Body: `"inference error: ResourceExhausted - system saturated, sheddable request dropped"`

---

## 3. How BLIS Models This

### Default: AlwaysAdmit (Does NOT Match)

BLIS defaults to `AlwaysAdmit` — every request gets in, no saturation check, no shedding. This is intentional for backward compatibility but **does not model production behavior**.

**File:** `sim/admission.go`

```go
type AlwaysAdmit struct{}

func (a *AlwaysAdmit) Admit(_ *Request, _ *RouterState) (bool, string) {
    return true, ""
}
```

### Closest Match: TierShedAdmission

The BLIS equivalent of GAIE's legacy controller is `TierShedAdmission`:

**File:** `sim/admission.go`

```go
type TierShedAdmission struct {
    OverloadThreshold int  // max per-instance effective load before shedding
    MinAdmitPriority  int  // minimum priority admitted under overload
}

func (t *TierShedAdmission) Admit(req *Request, state *RouterState) (bool, string) {
    // Batch/Background bypass (deferred queue handles them separately)
    if class == "batch" || class == "background" {
        return true, ""
    }

    // Find max effective load across all instances
    maxLoad := 0
    for _, snap := range state.Snapshots {
        if l := snap.EffectiveLoad(); l > maxLoad {
            maxLoad = l
        }
    }

    // Under threshold → admit all
    if maxLoad <= t.OverloadThreshold {
        return true, ""
    }

    // Over threshold → reject if priority too low
    priority := SLOTierPriority(class)
    if priority < t.MinAdmitPriority {
        return false, "tier-shed: ..."
    }
    return true, ""
}
```

Same idea: check load, reject low-priority stuff when overloaded, always admit high-priority.

### Saturation Detectors (Exact Match)

BLIS has the same two detectors as GAIE, with the same formulas:

**File:** `sim/saturation.go`

**UtilizationDetector:**
```go
func (u *UtilizationDetector) Saturation(state *RouterState) float64 {
    sum := 0.0
    for _, snap := range state.Snapshots {
        qScore  := float64(snap.QueueDepth) / u.queueDepthThreshold
        kvScore := snap.KVUtilization / u.kvCacheUtilThreshold
        sum += math.Max(qScore, kvScore)
    }
    return sum / float64(len(state.Snapshots))
}
```

Same formula as GAIE: `avg(max(qd/thresh, kv/thresh))` per instance.

**ConcurrencyDetector:**
```go
func (c *ConcurrencyDetector) Saturation(state *RouterState) float64 {
    totalInFlight := 0
    for _, snap := range state.Snapshots {
        totalInFlight += snap.InFlightRequests
    }
    return float64(totalInFlight) / float64(len(state.Snapshots) * c.maxConcurrency)
}
```

Same formula as GAIE: `totalInflight / (instances * maxConcurrency)`.

### Priority Mapping

Different numbering, same concept:

| GAIE | BLIS | Meaning |
|------|------|---------|
| priority < 0 → sheddable | priority 0 (background), 1 (batch), 2 (sheddable) | Can be dropped |
| priority >= 0 → protected | priority 3 (standard), 4 (critical) | Always admitted |

---

## 4. Parity Scorecard

| Aspect | GAIE Legacy | BLIS (TierShedAdmission) | Parity? |
|--------|-------------|--------------------------|---------|
| Reject sheddable when saturated | Yes | Yes | **Yes** |
| Always admit non-sheddable | Yes | Yes | **Yes** |
| Utilization-based saturation formula | `avg(max(qd/t, kv/t))` | Same formula | **Yes** |
| Concurrency-based saturation formula | `total / (n * max)` | Same formula | **Yes** |
| Stateless (no memory between decisions) | Yes | Yes | **Yes** |
| Saturation signal: which detector is default | Utilization | User chooses (flow control flag) | **Close** |
| Saturation threshold | `>= 1.0` | `> OverloadThreshold` (effective load, not 0-1 scale) | **Different signal** |
| Metrics staleness fallback | >200ms → treat as saturated | No staleness guard | **Gap** |
| HTTP error code on rejection | 429 with message | N/A (simulation, no HTTP) | N/A |

---

## 5. What's Missing in BLIS

### Gap A: Saturation Signal Mismatch

This is the most important difference. GAIE's legacy controller checks saturation as a **0-to-1 normalized score** (average of per-pod ratios). BLIS's `TierShedAdmission` checks **EffectiveLoad** (raw count: `QueueDepth + BatchSize + InFlightRequests`) against a threshold integer.

These are different signals:
- GAIE: "is the cluster on average running at capacity?" (normalized, ratio-based)
- BLIS TierShedAdmission: "does any single instance have more than N things in progress?" (absolute, max-based)

BLIS **does** have the same normalized saturation detectors (`UtilizationDetector`, `ConcurrencyDetector`) — but they're only used by the **gateway queue flow control** path, not by `TierShedAdmission`. The two mechanisms are decoupled.

**To get true GAIE legacy parity**, you'd want TierShedAdmission to use the saturation detectors instead of raw EffectiveLoad. Or: combine `--admission-policy always-admit` with `--flow-control` and the utilization detector — then the gateway queue's saturation gate does the job.

### Gap B: Metrics Staleness Handling

GAIE treats stale metrics (>200ms old) as "pod is saturated" — a safety fallback. BLIS's saturation detectors always use whatever snapshot values are available, governed by `--snapshot-refresh-interval` but with no "if stale, assume worst" logic. In simulation this is less critical (no real network delays), but if we're modeling staleness effects it matters.

### Gap C: TierShedAdmission Uses Max, Not Average

GAIE's saturation is the **average** across pods. BLIS's TierShedAdmission triggers on the **max** effective load across instances. Under uneven load, max triggers earlier than average — BLIS is more aggressive about shedding. This is a subtle but real behavioral difference.

### Gap D: No Combined "Saturated AND Sheddable" Single Check

In GAIE, it's one atomic check: `if sheddable AND saturated → reject`. In BLIS, the admission policy and saturation detector are separate systems. To get the GAIE legacy behavior exactly, you either need to:
1. Wire the saturation detector into TierShedAdmission (doesn't exist today), or
2. Use flow control mode with always-admit + saturation gate (close but not identical — adds queuing)

Neither is a perfect 1:1 reproduction of the GAIE legacy controller's simplicity.

---

## 6. How to Get Closest to GAIE Legacy in BLIS Today

**Option 1: TierShedAdmission (closest behavior, different signal)**
```bash
blis run --model qwen/qwen3-32b \
  --admission-policy tier-shed \
  --tier-shed-threshold 5 \
  --tier-shed-min-priority 3
```
Rejects sheddable/batch/background when any instance has EffectiveLoad > 5. Close in spirit, different signal.

**Option 2: Flow control with utilization detector (same signal, adds queuing)**
```bash
blis run --model qwen/qwen3-32b \
  --flow-control \
  --saturation-detector utilization \
  --queue-depth-threshold 5 \
  --kv-cache-util-threshold 0.8
```
Uses the exact GAIE saturation formula, but adds a gateway queue (GAIE legacy doesn't queue — it just rejects). Requests are held, not shed.

**Neither option is a perfect match.** The gap is small but worth knowing about.

---

## 7. Transfer Path: BLIS to llm-d (Proven)

We already have a working sim2real transfer for admission control. The `by60-admission-policy` plugin in `experiments/sim2real_admission_results_analysis/` proves the pattern:

### How It Works

The admission logic lives in llm-d as a **GAIE Filter plugin** — not by modifying GAIE's admission controller. A Filter that returns all endpoints = admit; returns empty list = reject. This is clean: one new file in llm-d, one line in `register.go`, zero GAIE changes.

### The Translation Recipe

| BLIS (simulation) | llm-d (production) |
|--------------------|---------------------|
| `AdmissionPolicy` interface | `scheduling.Filter` interface |
| `Admit(req, state) (bool, string)` | `Filter(ctx, state, req, endpoints) []Endpoint` |
| `return true, ""` (admit) | `return endpoints` (admit) |
| `return false, "reason"` (reject) | `return []scheduling.Endpoint{}` (reject) |
| `req.SLOClass` | `request.Headers["x-slo-class"]` |
| `req.TenantID` | `request.Headers["x-tenant-id"]` |
| `state.Clock` (simulation clock) | `time.Now().UnixMicro()` |
| `snap.InFlightRequests` | `endpoint.GetMetrics().RunningRequestsSize` |
| `snap.QueueDepth` | `endpoint.GetMetrics().WaitingQueueSize` |
| `snap.KVUtilization` | `endpoint.GetMetrics().KVCacheUsagePercent / 100.0` |
| `len(state.Snapshots)` (instances) | `len(endpoints)` (endpoints) |

### What's Already Been Transferred

The `AdaptiveAdmission` from `best_program.go` (BLIS) became `admission_blis_weighted_scoring.go` (llm-d Filter plugin). Line-for-line identical logic:
- Always admit critical/standard
- Shed batch at 50% load ratio, sheddable at 75%
- Tenant fairness: heavy tenants (>1.5x average) get tighter thresholds
- 10-second sliding window for capacity estimation

### What This Means for New Work

Any new admission policy we develop and validate in BLIS can follow the same path:
1. Implement as `AdmissionPolicy` in BLIS, validate in simulation
2. Translate to a `scheduling.Filter` in llm-d using the mapping above
3. Register with `plugin.Register("plugin-name", Factory)` — one diff
4. Configure in the llm-d scheduling profile YAML

The filter plugin approach bypasses the legacy admission controller entirely. Our admission logic runs inside the scheduling pipeline, which gives us access to the same endpoint metrics the scorers use. No need to touch GAIE's admission controller or flow controller code.
