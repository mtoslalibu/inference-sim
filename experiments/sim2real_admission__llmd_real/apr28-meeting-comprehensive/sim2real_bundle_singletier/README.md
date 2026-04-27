# Sim2Real Bundle: Binary Quintic Admission (Qwen3-14B, Parameter-Free)

Simulation-evolved admission control using a single quintic probability curve for all droppable traffic. Binary tier model: protected (priority >= 0, always admit) vs droppable (priority < 0, probabilistic shedding). Transferable to llm-d/GAIE as an `AdmissionPlugin`.

## File Structure

```
sim2real_bundle_singletier/
  algorithm/admission.go   — Quintic admission plugin (Go, GAIE-compatible)
  workloads/               — 10 workload YAMLs (5 shapes x 2 load levels)
  simulation/              — BLIS run script, results, and comparison tables
  config.md                — Real deployment config (vLLM flags, GAIE thresholds)
  README.md                — This file
```

## Algorithm: Quintic Admission (Treatment)

**Source**: [`algorithm/admission.go`](algorithm/admission.go)

The algorithm computes GAIE pool-average saturation, then rejects droppable requests with probability proportional to saturation raised to the 5th power:

1. Compute **cluster saturation**: `avg(max(QD/5.0, KV/0.8))` across all instances
2. **Protected** (priority >= 0): always admit
3. **Droppable** (priority < 0): reject with probability `min(sat^5 * 300, 1.0)`

### Why quintic (5th power)?

The power law creates a natural shape that adapts to load without any thresholds:

| Saturation | sat^5 * 300 | Meaning |
|------------|-------------|---------|
| 0.12 (under-capacity) | 0.007 | ~1% shed — virtually zero waste |
| 0.25 (moderate) | 0.29 | Starting to protect |
| 0.34 (overloaded) | 1.0 | All droppable fully shed |

The transition from "nearly nothing" to "full shedding" happens naturally between sat 0.2-0.34. No explicit threshold needed.

### Nil metrics handling (startup edge case)

When a pod first starts and hasn't reported metrics yet (~200-500ms), treat it as fully loaded — matching GAIE's conservative default. In `computeSaturation`, if metrics are nil, add `1.0` to the total instead of skipping the pod. Without this, nil-metric pods pull the saturation average down, causing under-shedding during startup.

## Config

BLIS simulation calibrated to real vLLM on 4x H100-SXM-80GB (~73 req/s). See [`config.md`](config.md) for details.

## How to Transfer to llm-d/GAIE

### GAIE Interface

Implement `requestcontrol.AdmissionPlugin` (`pkg/epp/framework/interface/requestcontrol/plugins.go:78`). Return `nil` to admit, `error` to reject.

### Signal Mapping

| Signal | GAIE accessor |
|--------|--------------|
| Queue depth | `pod.GetMetrics().WaitingQueueSize` |
| KV utilization (0-1) | `pod.GetMetrics().KVCacheUsagePercent` (already 0-1 despite the field name) |
| Priority | `request.Objectives.Priority` |

### Priority Tiers (InferenceModel CRs)

| Category | Tiers | Priority | Quintic shedding |
|----------|-------|----------|-----------------|
| **Protected** | critical, standard | >= 0 | Never |
| **Droppable** | sheddable, batch, background | < 0 | `min(sat^5 * 300, 1.0)` |

### Transfer: Quintic Admission (Treatment)

**Pseudocode:**

```
function AdmitRequest(request, pods):
    priority = request.Objectives.Priority

    if priority >= 0:       return ADMIT   // protected (critical, standard)
    if len(pods) == 0:      return ADMIT   // safe default

    // Cluster saturation (GAIE formula)
    saturation = 0
    for each pod in pods:
        m = pod.GetMetrics()
        if m == nil:
            saturation += 1.0          // no metrics yet → treat as fully loaded
            continue
        qRatio  = m.WaitingQueueSize / 5.0
        kvRatio = m.KVCacheUsagePercent / 0.8
        saturation += max(qRatio, kvRatio)
    saturation /= len(pods)

    // Quintic probability — single curve for all droppable traffic
    sat5 = saturation^5
    p = min(sat5 * 300, 1.0)

    if rand() < p:            return REJECT
    return ADMIT
```

**YAML Config:**

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: quintic-shed-admitter
  name: quintic-shed
  parameters:
    queueDepthThreshold: 5
    kvCacheUtilThreshold: 0.8
    power: 5
    k: 300
- type: random-picker
saturationDetector:
  queueDepthThreshold: 999999999
  kvCacheUtilThreshold: 0.999
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: random-picker
```

### Disabling GAIE Legacy Admission (Required for Quintic)

GAIE has a **two-layer admission pipeline**. Both layers run on every request:

1. **Layer 1 (built-in, always active)**: `LegacyAdmissionController` — rejects sheddable requests (priority < 0) when saturation >= 1.0. Runs at `director.go:168` *before* any custom plugins. Cannot be disabled via config.
2. **Layer 2 (plugin-based)**: Custom `AdmissionPlugin` instances registered in YAML — run at `director.go:189` *after* Layer 1 passes.

**Problem**: We want our quintic plugin to be the **sole** admission decision-maker. If Layer 1 is also active, both run — our plugin's decisions are masked by the legacy controller at saturation >= 1.0.

**Solution**: Set the legacy saturation detector thresholds astronomically high via inline `saturationDetector` config so it never triggers:

```yaml
saturationDetector:
  queueDepthThreshold: 999999999   # effectively infinite — legacy never triggers on QD
  kvCacheUtilThreshold: 0.999      # near-max (1.0 is boundary of GAIE validation)
```

This makes `Saturation() ≈ 0.0` always, so `LegacyAdmissionController` at `admission.go:64-84` never enters the rejection path.

**Variant 1 (Default llm-d)** does NOT need this — it uses the built-in legacy admission as-is, which is exactly what we're comparing against.

### Load Generator (blis observe)

Use **BLIS v0.7.10** image for `blis observe`. Important flags:

```bash
blis observe \
  --max-concurrency 10000 \
  --warmup-requests 200 \
  --timeout 900 \
  ...
```

- **`--max-concurrency 10000`**: BLIS simulation is true open-loop — all requests arrive at their scheduled times with no concurrency limit. The default (`--max-concurrency 256`) is too low for overload experiments. High value ensures true open-loop arrival.
- **`--warmup-requests 200`**: Discard initial requests to avoid cold-start artifacts (KV cache empty, scheduler not warmed up).
- **`--timeout 900`**: Per-request HTTP timeout (15 min). Prevents premature timeouts for long-context requests under overload.

### Deployment Plan

Two variants to deploy and compare:

1. **Default llm-d** — built-in GAIE legacy admission (no custom plugin, no saturation override)
2. **Quintic** (`admission.go` → `quintic-shed-admitter` plugin) — quintic probabilistic shedding, with legacy admission disabled via noop saturation detector

Both use `random-picker` with no scorers — purely random endpoint selection. The admission algorithm is the only variable.

### Key GAIE Source Files

- `AdmissionPlugin` interface: `pkg/epp/framework/interface/requestcontrol/plugins.go:78`
- Saturation formula: `pkg/epp/framework/plugins/flowcontrol/saturationdetector/utilization/detector.go`
- Endpoint metrics: `pkg/epp/framework/interface/datalayer/metrics.go`
- Request priority: `pkg/epp/framework/interface/scheduling/types.go:40`
- Plugin registry: `llm-d-inference-scheduler/pkg/plugins/register.go`

## Workloads

Five workload shapes testing different traffic mixes and request sizes, each at two load levels (under/mid):

| Family | Critical % | Sheddable % | Input Tokens | Output Tokens | Rate (under/mid) | Purpose |
|--------|-----------|-------------|--------------|---------------|-------------------|---------|
| **W1** | 50% | 50% | 1024 | 256 | 35 / 90 | Balanced split, real-validated rates |
| **W2** | 30% | 70% | 1024 | 256 | 35 / 90 | Droppable-dominant, real-validated rates |
| **Chatbot** | 80% | 20% | 4096 | 1024 | 5 / 10 | Long-context, critical-dominant |
| **Code Completion** | 30% | 70% | 2048 | 128 | 40 / 95 | Short output, high throughput |
| **Blindspot** | 10% | 90% | 4096 | 1024 | 5 / 10 | Long-context, droppable-dominant |

Binary tiers: critical (priority >= 0, protected) and sheddable (priority < 0, droppable). All workloads: Poisson arrivals, Gaussian tokens, seed=42, streaming=true.

W1/W2 rates match real vLLM experiments. New workload rates calibrated down ~30% based on sim-vs-real latency gap analysis (sim underestimates E2E by ~1.35x under-load, ~1.6x mid-load).

Simulation results and comparison tables: see [`simulation/notes.md`](simulation/notes.md).

