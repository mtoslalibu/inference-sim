# Sim2Real Bundle: Qwen3-14B Preemptive Shed Admitter

## Summary

Simulation-evolved admission control algorithm that dramatically improves SLO attainment for critical and standard requests under overload. Discovered via BLIS simulation, transferable to llm-d / GAIE as an `AdmissionPlugin`.

**Model**: Qwen/Qwen3-14B on 4x H100-SXM-80GB, TP=1, trained-physics latency model

## Where Is the Discovered Algorithm

**File**: `algorithm/admission.go`
**Struct**: `AdaptiveAdmission`
**Method**: `Admit(req *Request, state *RouterState) (bool, string)` — line 241
**Core logic**: lines 280-353, between `// EVOLVE-BLOCK-START` and `// EVOLVE-BLOCK-END`

Everything else in that file (TokenBucket, GAIELegacyAdmission, TierShed, etc.) is other admission policies — ignore them. The discovered algorithm is **only** the EVOLVE-BLOCK inside `AdaptiveAdmission.Admit()`.

### What the Algorithm Does (Plain English)

1. Compute **cluster saturation** using the standard GAIE formula: `avg(max(QD/5.0, KV/0.8))` across all instances
2. **Critical and standard** requests: always admit, never reject
3. **Sheddable** requests: if saturation >= 0.01, probabilistically reject with probability ramping linearly from 0 to 1 over [0.01, 0.10]
4. **Batch** requests: if saturation >= 0.005, probabilistically reject with probability ramping linearly from 0 to 1 over [0.005, 0.05]

The key insight: the GAIE legacy baseline waits until saturation=1.0 to start shedding — by then queues are deep and latency is ruined. This algorithm starts at saturation=0.01 (effectively any non-zero load). The **timing** of shedding matters more than the total amount shed.

## Results: Iter11 vs GAIE Legacy Baseline

### W1: Sustained Overload (rate=110, 1.5x capacity)

Tier mix: critical 20%, standard 30%, sheddable 45%, batch 5%

| SLO Target | Critical Baseline | Critical Iter11 | **Gain** | Standard Baseline | Standard Iter11 | **Gain** |
|-----------|------------------|----------------|---------|------------------|----------------|---------|
| E2E<5s   | 3.9%  | 34.5% | **+30.6pp** | 4.2%  | 34.9% | **+30.6pp** |
| E2E<8s   | 28.5% | 96.5% | **+67.9pp** | 29.9% | 95.9% | **+66.1pp** |
| E2E<10s  | 59.9% | 99.9% | **+40.0pp** | 60.7% | 100%  | **+39.3pp** |
| E2E<12s  | 86.6% | 100%  | +13.4pp     | 86.7% | 100%  | +13.3pp     |
| Rejects  | 1,248 | 3,055 |             |        |       |             |

### W2: Burst (rate=219, 1x→2x capacity)

30s at 1x capacity then 30s at 2x. Tier mix same as W1.

| SLO Target | Critical Baseline | Critical Iter11 | **Gain** | Standard Baseline | Standard Iter11 | **Gain** |
|-----------|------------------|----------------|---------|------------------|----------------|---------|
| E2E<5s   | 8.3%  | 36.4% | **+28.1pp** | 8.5%  | 36.0% | **+27.5pp** |
| E2E<8s   | 43.4% | 83.3% | **+39.8pp** | 43.6% | 78.8% | **+35.2pp** |
| E2E<10s  | 69.9% | 96.7% | **+26.8pp** | 68.2% | 95.6% | **+27.4pp** |
| E2E<12s  | 87.3% | 99.7% | +12.4pp     | 87.0% | 99.6% | +12.6pp     |
| Rejects  | 1,589 | 2,939 |             |        |       |             |

### W3: High Sheddable (rate=210, 2.9x capacity, 65% sheddable)

Tier mix: critical 10%, standard 20%, sheddable 65%, batch 5%

| SLO Target | Critical Baseline | Critical Iter11 | **Gain** | Standard Baseline | Standard Iter11 | **Gain** |
|-----------|------------------|----------------|---------|------------------|----------------|---------|
| E2E<5s   | 3.1%  | 20.8% | **+17.7pp** | 2.8%  | 22.3% | **+19.4pp** |
| E2E<8s   | 25.1% | 86.6% | **+61.6pp** | 26.3% | 87.1% | **+60.7pp** |
| E2E<10s  | 56.1% | 99.4% | **+43.3pp** | 58.0% | 99.1% | **+41.2pp** |
| E2E<12s  | 85.2% | 100%  | +14.8pp     | 86.1% | 100%  | +13.9pp     |
| Rejects  | 6,825 | 8,246 |             |        |       |             |

## Simulation Config

- Latency model: `trained-physics` (production-realistic, no roofline)
- Hardware: 4x H100-SXM-80GB, TP=1
- Cluster capacity: ~73 req/s
- Routing: round-robin
- Snapshot staleness: 50ms (`--snapshot-refresh-interval 50000`)
- Horizon: 60s, Poisson arrivals, Gaussian tokens (mean 1024 input / 256 output)

## Bundle Contents

```
sim2real_bundle_14b/
├── README.md                          # this file
├── algorithm/
│   └── admission.go                   # original BLIS source — look for EVOLVE-BLOCK (lines 280-353) in AdaptiveAdmission.Admit()
├── workloads/
│   ├── w1_14b.yaml                    # sustained 1.5x overload
│   ├── w2_14b_burst.yaml              # burst 1x→2x
│   └── w3_14b_high_sheddable.yaml     # 2.9x with 65% sheddable
└── results/
    ├── 14b_baseline_w1.txt            # GAIE legacy baseline
    ├── 14b_baseline_w2.txt
    ├── 14b_baseline_w3.txt
    ├── 14b_iter11_w1.txt              # preemptive shed algorithm
    ├── 14b_iter11_w2.txt
    └── 14b_iter11_w3.txt
```

## How to Transfer to llm-d / GAIE

The original BLIS algorithm in `algorithm/admission.go` uses BLIS-specific types (`*Request`, `*RouterState`, `InstanceSnapshot`). To port it to GAIE/llm-d, you need to implement the `requestcontrol.AdmissionPlugin` interface and map the signals. Here's exactly how.

### The GAIE Interface You Need to Implement

The interface is in `pkg/epp/framework/interface/requestcontrol/plugins.go:78`:

```go
type AdmissionPlugin interface {
    plugin.Plugin
    AdmitRequest(ctx context.Context, request *types.LLMRequest, pods []types.Endpoint) error
}
```

- Return `nil` to **admit** the request
- Return an `error` to **reject** the request (the error message is the denial reason)
- `plugin.Plugin` just requires a `TypedName() plugin.TypedName` method

### Signal Mapping: BLIS → GAIE

The BLIS algorithm reads these signals. Here's what they map to in GAIE:

| What the algorithm reads | BLIS code | GAIE equivalent | Notes |
|--------------------------|-----------|-----------------|-------|
| Number of instances | `len(state.Snapshots)` | `len(pods)` | Direct |
| Per-instance queue depth | `snap.QueueDepth` | `pod.GetMetrics().WaitingQueueSize` | Both are `int` |
| Per-instance KV util (0-1) | `snap.KVUtilization` | `pod.GetMetrics().KVCacheUsagePercent / 100.0` | **GAIE uses percent (0-100), BLIS uses fraction (0-1)** |
| Request SLO class | `req.SLOClass` (string: "critical", "standard", "sheddable", "batch") | `request.Objectives.Priority` (int: >=0 is protected, -1 is sheddable, <=-2 is batch) | **Different representation — see mapping below** |

### Priority Mapping

BLIS uses string SLO classes. GAIE uses integer priority on `request.Objectives.Priority`:

| BLIS `sloClass` | GAIE `Priority` | Behavior |
|------------------|-----------------|----------|
| `"critical"` | `>= 0` (e.g. 4) | Always admit |
| `"standard"` | `>= 0` (e.g. 3) | Always admit |
| `"sheddable"` | `-1` | Shed with ramp [0.01, 0.10] |
| `"batch"` | `<= -2` | Shed with ramp [0.005, 0.05] |

So in GAIE terms: `if priority >= 0: admit. if priority == -1: sheddable ramp. if priority <= -2: batch ramp.`

### Saturation Formula (Identical)

The BLIS saturation formula is already identical to GAIE's `utilization/detector.go`:

```go
// BLIS (algorithm/admission.go lines 288-300):
for _, snap := range state.Snapshots {
    qRatio := float64(snap.QueueDepth) / 5.0
    kvRatio := snap.KVUtilization / 0.8
    saturation += max(qRatio, kvRatio)
}
saturation /= float64(numInstances)

// GAIE equivalent:
for _, pod := range pods {
    m := pod.GetMetrics()
    qRatio := float64(m.WaitingQueueSize) / 5.0
    kvRatio := (m.KVCacheUsagePercent / 100.0) / 0.8    // <-- only difference: divide by 100
    saturation += max(qRatio, kvRatio)
}
saturation /= float64(len(pods))
```

The **only difference** is `KVCacheUsagePercent / 100.0` (GAIE uses percent, BLIS uses fraction).

### Pseudo-Random Shedding

The BLIS algorithm uses a deterministic pseudo-random value based on a request counter:

```go
requestOrdinal := float64(a.totalAdmitted+a.totalRejected) / 100.0
randVal := requestOrdinal - float64(int(requestOrdinal))  // fractional part
if randVal < p { REJECT }
```

In production you can replace this with `rand.Float64() < p` for true randomness, or keep the counter-based approach for reproducible testing.

### Plugin Registration

In `llm-d-inference-scheduler/pkg/plugins/register.go`, register your factory:

```go
plugin.Register("preemptive-shed-admitter", yourFactory)
```

The factory signature is `func(name string, parameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error)`.

### YAML Configuration

Wire the plugin in your `EndpointPickerConfig`:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: EndpointPickerConfig
spec:
  requestControl:
    admissionPlugins:
      - type: preemptive-shed-admitter
        name: my-admitter
        parameters:
          queueDepthThreshold: 5
          kvCacheUtilThreshold: 0.8
          sheddableShedStart: 0.01
          sheddableShedFull: 0.10
          batchShedStart: 0.005
          batchShedFull: 0.05
```

### Setting Request Priority

In your `InferenceModel` CRs, set `spec.objectives.priority`:
- `priority >= 0`: protected (critical, standard) — never shed
- `priority = -1`: sheddable
- `priority <= -2`: batch

### Key GAIE Source Files

- `AdmissionPlugin` interface: `pkg/epp/framework/interface/requestcontrol/plugins.go:78`
- Plugin base interface: `pkg/epp/framework/interface/plugin/plugins.go:21`
- Plugin registry + Factory signature: `pkg/epp/framework/interface/plugin/registry.go`
- Saturation formula reference: `pkg/epp/framework/plugins/flowcontrol/saturationdetector/utilization/detector.go`
- Endpoint metrics (WaitingQueueSize, KVCacheUsagePercent): `pkg/epp/framework/interface/datalayer/metrics.go`
- Request priority: `pkg/epp/framework/interface/scheduling/types.go:40` (`RequestObjectives.Priority`)
- Existing plugin example: `llm-d-inference-scheduler/pkg/plugins/register.go`
