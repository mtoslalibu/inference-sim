# Feedback: Translating Adaptive Router to Real llm-d

This document tells you exactly what to change in `adaptive_v2.go`, `treatment_config.yaml`,
and `register.go` to faithfully translate the simulation algorithm to production llm-d.

---

## Problem with Current Translation

The current `adaptive_v2.go` uses 3 scorers (ppc, load-aware, kv-util) and collapses
the simulation's 5 scorers into 3. This is wrong. All 5 scorers exist in the real system.
The collapse changes the algorithm behavior and loses the independent signal weighting
that produces the simulation wins.

---

## The 5 Scorers: Sim-to-Real Mapping

| # | Sim scorer name | Real scorer type | Where registered | What it reads |
|---|----------------|------------------|-----------------|---------------|
| 0 | `precise-prefix-cache` | `precise-prefix-cache-scorer` | llm-d `register.go` | KV cache hash maps via ZMQ events |
| 1 | `load-aware` | `load-aware-scorer` | llm-d `register.go` | `WaitingQueueSize` (queue depth) |
| 2 | `active-requests` | `active-request-scorer` | llm-d `register.go` | In-flight requests tracked by EPP itself (synchronous, zero staleness) |
| 3 | `running-requests` | `running-requests-size-scorer` | GAIE runner (auto-registered) | `RunningRequestsSize` from endpoint metrics |
| 4 | `kv-utilization` | `kv-cache-utilization-scorer` | GAIE runner (auto-registered) | `KVCacheUsagePercent` (0.0-1.0 range) |

### Parity: load-aware IS the same in sim and real

Both use the same formula: `0.5 * (1 - queueDepth/128)`, range [0, 0.5].

**BLIS sim** (`sim/routing_scorers.go:272-286`):
```go
const loadAwareQueueThreshold = 128  // Matches llm-d's QueueThresholdDefault

func scoreLoadAware(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
    for _, snap := range snapshots {
        if snap.QueueDepth == 0 {
            scores[snap.ID] = 0.5
        } else {
            clamped := snap.QueueDepth
            if clamped > loadAwareQueueThreshold { clamped = loadAwareQueueThreshold }
            scores[snap.ID] = 0.5 * (1.0 - float64(clamped)/float64(loadAwareQueueThreshold))
        }
    }
}
```

**Real llm-d** (`pkg/plugins/scorer/load_aware.go:83-99`):
```go
func (s *LoadAware) Score(..., endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
    for _, endpoint := range endpoints {
        waitingRequests := float64(endpoint.GetMetrics().WaitingQueueSize)
        if waitingRequests == 0 {
            scoredEndpoints[endpoint] = 0.5
        } else {
            if waitingRequests > s.queueThreshold { waitingRequests = s.queueThreshold }
            scoredEndpoints[endpoint] = 0.5 * (1.0 - (waitingRequests / s.queueThreshold))
        }
    }
}
```

Identical logic. Both read queue depth only, both output [0, 0.5], both default to threshold 128.

NOTE: BLIS also has a SEPARATE scorer called `load-balance` (`sim/routing_scorers.go:194-200`)
that uses `1/(1+EffectiveLoad)`. That is NOT used by the adaptive algorithm. Don't confuse them.

---

## The Algorithm: 3 Regimes with 5 Scorers

On every request, detect the regime and apply regime-specific weights:

```
Scorer index order: [0]=ppc, [1]=load-aware, [2]=active-requests, [3]=running-requests, [4]=kv-util

Regime detection:
  cacheSpread = max(ppc_score) - min(ppc_score) across all endpoints
  avgKVUtil   = average of endpoint KVCacheUsagePercent values

if cacheSpread > 0.1:
    REGIME: cache-affinity
    weights = [4, 1, 1, 0, 0]   (ppc:4, la:1, ar:1, rr:0, kvu:0)

elif avgKVUtil > 0.7:
    REGIME: memory-aware
    weights = [0, 1, 2, 1, 1]   (ppc:0, la:1, ar:2, rr:1, kvu:1)

else:
    REGIME: load-balance
    weights = [0, 1, 2, 1, 0]   (ppc:0, la:1, ar:2, rr:1, kvu:0)

Normalize weights to sum to 1, compute weighted sum per endpoint, pick max.
```

---

## Changes Needed in adaptive_v2.go

### 1. Change from 3 sub-scorers to 5

Current (WRONG):
```go
type AdaptiveV2Parameters struct {
    PPCScorerName string `json:"ppcScorerName"`
    LAScorerName  string `json:"laScorerName"`
    KVUScorerName string `json:"kvuScorerName"`
    ...
}

type AdaptiveV2Scorer struct {
    ppcScorer scheduling.Scorer
    laScorer  scheduling.Scorer
    kvuScorer scheduling.Scorer
    ...
}
```

Change to (CORRECT):
```go
type AdaptiveV2Parameters struct {
    PPCScorerName string `json:"ppcScorerName"`  // precise-prefix-cache-scorer
    LAScorerName  string `json:"laScorerName"`   // load-aware-scorer
    ARScorerName  string `json:"arScorerName"`   // active-request-scorer
    RRScorerName  string `json:"rrScorerName"`   // running-requests-size-scorer
    KVUScorerName string `json:"kvuScorerName"`  // kv-cache-utilization-scorer
    CacheSpreadThreshold float64 `json:"cacheSpreadThreshold"`
    KVPressureThreshold  float64 `json:"kvPressureThreshold"`
}

type AdaptiveV2Scorer struct {
    typedName plugin.TypedName

    ppcScorer scheduling.Scorer // index 0: precise-prefix-cache-scorer
    laScorer  scheduling.Scorer // index 1: load-aware-scorer
    arScorer  scheduling.Scorer // index 2: active-request-scorer
    rrScorer  scheduling.Scorer // index 3: running-requests-size-scorer
    kvuScorer scheduling.Scorer // index 4: kv-cache-utilization-scorer

    cacheSpreadThreshold float64
    kvPressureThreshold  float64
}
```

### 2. Update the factory to resolve all 5 scorers

Add lookups for `arScorerName` and `rrScorerName` using the same pattern:
```go
arScorer, err := plugin.PluginByType[scheduling.Scorer](handle, params.ARScorerName)
rrScorer, err := plugin.PluginByType[scheduling.Scorer](handle, params.RRScorerName)
```

### 3. Update the Score() function regime weights

Change from 3-element weight arrays to 5-element:

```go
// Run all 5 sub-scorers
allScores := []map[scheduling.Endpoint]float64{
    s.ppcScorer.Score(ctx, cycleState, request, endpoints),  // index 0
    s.laScorer.Score(ctx, cycleState, request, endpoints),   // index 1
    s.arScorer.Score(ctx, cycleState, request, endpoints),   // index 2
    s.rrScorer.Score(ctx, cycleState, request, endpoints),   // index 3
    s.kvuScorer.Score(ctx, cycleState, request, endpoints),  // index 4
}

// ... regime detection stays the same (cacheSpread from allScores[0]) ...

// Updated weights (5 elements, not 3):
switch {
case cacheSpread > s.cacheSpreadThreshold:
    rawWeights = [5]float64{4.0, 1.0, 1.0, 0.0, 0.0}  // cache-affinity
    regime = "cache-affinity"
case avgKVUtil > s.kvPressureThreshold:
    rawWeights = [5]float64{0.0, 1.0, 2.0, 1.0, 1.0}  // memory-aware
    regime = "memory-aware"
default:
    rawWeights = [5]float64{0.0, 1.0, 2.0, 1.0, 0.0}  // load-balance
    regime = "load-balance"
}
```

### 4. Cache spread computation stays the same

cacheSpread is computed from allScores[0] (ppc scores). No change needed.

### 5. KVCacheUsagePercent range

The real `KVCacheUsagePercent` is 0.0-1.0 (confirmed from GAIE source).
The threshold of 0.7 means "70% KV cache full". This is correct as-is.

---

## Changes Needed in treatment_config.yaml

### Full corrected config:

```yaml
kind: EndpointPickerConfig
metadata:
  name: adaptive-v2-routing
plugins:
  - type: data-parallel-profile-handler
    name: dp-handler
    parameters:
      primaryPort: 8000

  # Tokenizer — required by precise-prefix-cache-scorer
  - type: tokenizer
    parameters:
      modelName: "${MODEL_NAME}"              # e.g., Qwen/Qwen3-32B
      udsTokenizerConfig:
        socketFile: /tmp/tokenizer/tokenizer-uds.socket

  # Scorer 0: precise-prefix-cache (index 0 in regime weights)
  # Reference config: https://github.com/llm-d/llm-d/blob/main/guides/precise-prefix-cache-aware/gaie-kv-events/values.yaml
  - type: precise-prefix-cache-scorer
    name: ppc
    parameters:
      tokenProcessorConfig:
        blockSize: 64                         # must match vLLM block size
      indexerConfig:
        speculativeIndexing: true
        tokenizersPoolConfig:
          modelName: "${MODEL_NAME}"
          uds:
            socketFile: /tmp/tokenizer/tokenizer-uds.socket
      kvEventsConfig:
        topicFilter: "kv@"
        concurrency: 4
        discoverPods: false
        zmqEndpoint: "tcp://*:5557"

  # Scorer 1: load-aware (index 1 in regime weights)
  # Reads WaitingQueueSize. Default threshold=128.
  - type: load-aware-scorer
    name: la

  # Scorer 2: active-request (index 2 in regime weights)
  # Tracks in-flight requests via PreRequest/ResponseComplete hooks.
  # Fully synchronous — zero staleness. This is the key signal.
  # Default requestTimeout=2m, idleThreshold=0, maxBusyScore=1.0.
  - type: active-request-scorer
    name: ar

  # Scorer 3: running-requests-size (index 3 in regime weights)
  # Reads RunningRequestsSize from endpoint metrics (periodic).
  # No parameters needed.
  # NOTE: This scorer is registered by GAIE runner, not llm-d.
  #       It is auto-available — no registration needed in register.go.
  - type: running-requests-size-scorer
    name: rr

  # Scorer 4: kv-cache-utilization (index 4 in regime weights)
  # Reads KVCacheUsagePercent (0.0-1.0). No parameters needed.
  # NOTE: This scorer is registered by GAIE runner, not llm-d.
  #       It is auto-available — no registration needed in register.go.
  - type: kv-cache-utilization-scorer
    name: kvu

  # Filter: decode-filter
  - type: decode-filter
    name: decode

  # Adaptive regime-detection scorer (wraps the 5 component scorers)
  - type: adaptive-v2-scorer
    name: adaptive-v2
    parameters:
      ppcScorerName: ppc
      laScorerName: la
      arScorerName: ar
      rrScorerName: rr
      kvuScorerName: kvu
      cacheSpreadThreshold: 0.1
      kvPressureThreshold: 0.7

schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: decode
      - pluginRef: adaptive-v2
        weight: 1
```

### Key config notes:

1. **precise-prefix-cache-scorer**: Follow the llm-d guide exactly.
   The `tokenProcessorConfig.blockSize` MUST match your vLLM block size (typically 64).
   The `kvEventsConfig.zmqEndpoint` must match your ZMQ setup.
   The tokenizer UDS socket must be configured with the sidecar.

2. **load-aware-scorer**: No parameters needed. Default `threshold: 128` is fine.
   Only reads `WaitingQueueSize` — this is just queue depth, not full load.

3. **active-request-scorer**: No parameters needed for defaults.
   Optional: `requestTimeout` (default "2m"), `idleThreshold` (default 0),
   `maxBusyScore` (default 1.0). Defaults are fine.

4. **running-requests-size-scorer**: No parameters. Registered by GAIE runner.

5. **kv-cache-utilization-scorer**: No parameters. Registered by GAIE runner.

6. **Plugin ordering matters**: All 5 component scorers MUST appear BEFORE
   `adaptive-v2-scorer` in the plugins list. The adaptive scorer resolves
   them by name at initialization time via `plugin.PluginByType`.

---

## Changes Needed in register.go

Current register.go already has the adaptive-v2 registration:
```go
plugin.Register(scorer.AdaptiveV2Type, scorer.AdaptiveV2Factory)
```

No additional registrations needed for `running-requests-size-scorer` or
`kv-cache-utilization-scorer` — they are auto-registered by the GAIE runner
in `cmd/epp/runner/runner.go` (lines 457-459). The llm-d `RegisterAllPlugins()`
runs before the GAIE runner's registration, so component scorers from both
sources are available when `adaptive-v2-scorer` factory runs.

`active-request-scorer` is already registered in llm-d's `register.go`:
```go
plugin.Register(scorer.ActiveRequestType, scorer.ActiveRequestFactory)
```

No changes needed in register.go.

---

## Deployment Checklist

1. Update `adaptive_v2.go` to reference 5 sub-scorers (not 3)
2. Update `treatment_config.yaml` to instantiate all 5 component scorers
3. Ensure vLLM is configured with ZMQ KV cache events (`--enable-kv-cache-events`)
4. Ensure tokenizer UDS sidecar is deployed (see llm-d guide)
5. Set `blockSize` in PPC config to match vLLM block size
6. Set `MODEL_NAME` env var or replace `${MODEL_NAME}` in config
7. No changes needed in `register.go`

---

## Signal Freshness Summary

| Scorer | Signal type | Staleness |
|--------|------------|-----------|
| precise-prefix-cache-scorer | ZMQ events + speculative | ~2s (speculativeTTL) |
| load-aware-scorer | Periodic metrics scrape | ~50ms (GAIE RefreshMetricsInterval) |
| active-request-scorer | EPP-tracked PreRequest/ResponseComplete hooks | **0ms (synchronous)** |
| running-requests-size-scorer | Periodic metrics scrape | ~50ms |
| kv-cache-utilization-scorer | Periodic metrics scrape | ~50ms |

The `active-request-scorer` is the only fully synchronous signal. It was given
weight 2 (highest among load signals) in the memory-aware and load-balance regimes
specifically because its zero-staleness property prevents pile-on during bursts.
This is why collapsing it into load-aware was wrong — it eliminates the
differential weighting of the synchronous signal.

---

## Scorer Parity: Sim vs Real (Code Proof)

### Scorer 0: precise-prefix-cache — SAME (min-max normalization)

Both compute min-max normalization of prefix cache hit counts across instances/endpoints.
Output range: [0, 1].

### Scorer 1: load-aware — SAME (queue depth with threshold)

Both use `0.5 * (1 - queueDepth/128)`. Output range: [0, 0.5].
See code proof in the parity section above.

Note: output range [0, 0.5] means load-aware naturally contributes half as much
as other [0, 1] scorers at equal weight. This is by design in both sim and real.

### Scorer 2: active-requests — DIFFERENT normalization, same intent

**BLIS sim** (`sim/routing_scorers.go:202-255`):
```go
// max-only normalization: (maxCount - count) / maxCount
// Zero in-flight = 1.0, all equal non-zero = 0.0
func scoreActiveRequests(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
    // ... finds maxIFR across all instances ...
    for _, snap := range snapshots {
        ifr := snap.InFlightRequests
        if ifr == 0 { scores[snap.ID] = 1.0 }
        else { scores[snap.ID] = float64(maxIFR-ifr) / float64(maxIFR) }
    }
}
```

**Real llm-d** (`pkg/plugins/scorer/active_request.go:191-230`):
```go
func (s *ActiveRequest) Score(...) map[scheduling.Endpoint]float64 {
    // ... finds maxCount from EPP-tracked in-flight cache ...
    for _, endpoint := range endpoints {
        count := scoredEndpoints[endpointName]
        if count <= s.idleThreshold { scoredEndpointsMap[endpoint] = 1.0 }
        else { scoredEndpointsMap[endpoint] = float64(maxCount-count) / float64(maxCount) * s.maxBusyScore }
    }
}
```

Same normalization pattern (max-only). Real version adds `idleThreshold` and `maxBusyScore`
knobs (default to 0 and 1.0 = identical to sim behavior).

Key difference: sim reads `InFlightRequests` from snapshot (periodic). Real version
tracks requests itself via PreRequest/ResponseComplete hooks (fully synchronous, better).

### Scorer 3: running-requests — DIFFERENT normalization, same intent

**BLIS sim** (`sim/routing_scorers.go` — min-max of BatchSize):
```go
// min-max normalization: (max - value) / (max - min)
// All equal = 1.0
func scoreRunningRequests(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
    // min-max of snap.BatchSize
}
```

**Real GAIE** (`scorer/runningrequests/runningrequest.go:78-108`):
```go
// min-max normalization: (max - value) / (max - min)
// All equal = 1.0
func (s *RunningRequestsSizeScorer) Score(...) map[scheduling.Endpoint]float64 {
    // min-max of endpoint.GetMetrics().RunningRequestsSize
}
```

Same min-max normalization. Sim reads `BatchSize`, real reads `RunningRequestsSize`.
These measure the same thing (requests currently being processed on the GPU).

### Scorer 4: kv-utilization — SAME

**BLIS sim** (`sim/routing_scorers.go`): `1 - snap.KVUtilization`
**Real GAIE** (`scorer/kvcacheutilization/kvcache_utilization.go:79`): `1 - endpoint.GetMetrics().KVCacheUsagePercent`

Both 0.0-1.0 range. Identical.

---

## Scorer Configuration Knobs (Code Proof)

### precise-prefix-cache-scorer — HAS config (complex)

Source: `llm-d-inference-scheduler/pkg/plugins/scorer/precise_prefix_cache.go:59-81`

```go
type PrecisePrefixCachePluginConfig struct {
    TokenProcessorConfig *kvblock.TokenProcessorConfig `json:"tokenProcessorConfig"`
    IndexerConfig        *kvcache.Config               `json:"indexerConfig"`
    KVEventsConfig       *kvevents.Config              `json:"kvEventsConfig"`
    SpeculativeIndexing  bool                          `json:"speculativeIndexing"`
    SpeculativeTTL       string                        `json:"speculativeTTL"`
}
```

Required parameters (no useful defaults):
- `tokenProcessorConfig.blockSize` — MUST match vLLM block size (typically 64)
- `indexerConfig.tokenizersPoolConfig` — tokenizer model and UDS socket
- `kvEventsConfig.zmqEndpoint` — ZMQ endpoint for KV cache events

Optional:
- `speculativeIndexing` (default false, recommend true)
- `speculativeTTL` (default "2s" when speculative enabled)
- `kvEventsConfig.topicFilter` (default "", use "kv@" for filtering)
- `kvEventsConfig.concurrency` (default 1, recommend 4)
- `kvEventsConfig.discoverPods` (default true, set false if using explicit config)
- `indexerConfig.speculativeIndexing` (also settable at indexer level)

Reference config from llm-d guide (https://github.com/llm-d/llm-d/blob/main/guides/precise-prefix-cache-aware/gaie-kv-events/values.yaml):
```yaml
- type: precise-prefix-cache-scorer
  name: ppc
  parameters:
    tokenProcessorConfig:
      blockSize: 64
    indexerConfig:
      speculativeIndexing: true
      tokenizersPoolConfig:
        modelName: Qwen/Qwen3-32B
        uds:
          socketFile: /tmp/tokenizer/tokenizer-uds.socket
    kvEventsConfig:
      topicFilter: "kv@"
      concurrency: 4
      discoverPods: false
      zmqEndpoint: "tcp://*:5557"
```

Also needs tokenizer sidecar plugin BEFORE it in the plugins list:
```yaml
- type: tokenizer
  parameters:
    modelName: "${MODEL_NAME}"
    udsTokenizerConfig:
      socketFile: /tmp/tokenizer/tokenizer-uds.socket
```

### load-aware-scorer — HAS config (simple)

Source: `llm-d-inference-scheduler/pkg/plugins/scorer/load_aware.go:22-24`

```go
type loadAwareParameters struct {
    Threshold int `json:"threshold"`   // default: 128 (QueueThresholdDefault)
}
```

One parameter:
- `threshold` — queue depth at which score becomes 0. Default 128.

YAML (defaults are fine, no parameters needed):
```yaml
- type: load-aware-scorer
  name: la
  # parameters:          # optional
  #   threshold: 128     # default, only change if you know your queue behavior
```

### active-request-scorer — HAS config (simple)

Source: `llm-d-inference-scheduler/pkg/plugins/scorer/active_request.go:29-51`

```go
type ActiveRequestParameters struct {
    RequestTimeout string  `json:"requestTimeout"`  // default: "2m"
    IdleThreshold  int     `json:"idleThreshold"`   // default: 0
    MaxBusyScore   float64 `json:"maxBusyScore"`    // default: 1.0
}
```

Three parameters:
- `requestTimeout` — how long before an in-flight request is considered stale. Default "2m".
- `idleThreshold` — pods with <= this many requests score 1.0. Default 0 (only zero = idle).
- `maxBusyScore` — max score for busy pods. Default 1.0. Lower values create a gap between idle and busy.

YAML (defaults are fine, no parameters needed):
```yaml
- type: active-request-scorer
  name: ar
  # parameters:              # optional
  #   requestTimeout: "2m"   # default
  #   idleThreshold: 0       # default
  #   maxBusyScore: 1.0      # default
```

### running-requests-size-scorer — NO config

Source: `gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/scorer/runningrequests/runningrequest.go:37`

```go
func RunningRequestsSizeScorerFactory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
    return NewRunningRequestsSizeScorer().WithName(name), nil
}
```

Factory ignores `json.RawMessage` parameter entirely. No config knobs.

YAML:
```yaml
- type: running-requests-size-scorer
  name: rr
```

### kv-cache-utilization-scorer — NO config

Source: `gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/scorer/kvcacheutilization/kvcache_utilization.go:36`

```go
func KvCacheUtilizationScorerFactory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
    return NewKVCacheUtilizationScorer().WithName(name), nil
}
```

Factory ignores `json.RawMessage` parameter entirely. No config knobs.

YAML:
```yaml
- type: kv-cache-utilization-scorer
  name: kvu
```

### Summary of config needs

| Scorer | Has config? | Needs explicit config? |
|--------|------------|----------------------|
| precise-prefix-cache-scorer | Yes (complex) | YES — blockSize, ZMQ, tokenizer required |
| load-aware-scorer | Yes (1 param) | No — default threshold=128 is fine |
| active-request-scorer | Yes (3 params) | No — defaults match sim behavior |
| running-requests-size-scorer | No | No |
| kv-cache-utilization-scorer | No | No |

---

## Verification

After deploying, check the EPP logs at verbosity 4+ for lines like:
```
"Adaptive-v2 regime selected" regime="cache-affinity" cacheSpread=0.35 avgKVUtil=0.2
```

This confirms the regime detection is running and selecting regimes correctly.
