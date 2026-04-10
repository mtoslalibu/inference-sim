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

### Important: load-aware is NOT the same as in simulation

The sim's `load-aware` scorer computes `1/(1+QueueDepth+BatchSize+InFlightRequests)`.
The real llm-d `load-aware-scorer` ONLY reads `WaitingQueueSize` (queue depth).
It does NOT include BatchSize or InFlightRequests.

This is fine because the real system has separate scorers for each signal:
- `load-aware-scorer` covers queue depth (WaitingQueueSize)
- `active-request-scorer` covers in-flight requests (tracked by EPP)
- `running-requests-size-scorer` covers running/batch requests (RunningRequestsSize)

The sim combined some of these into `load-aware` because its EffectiveLoad included all three.
In production, keeping them separate is actually MORE faithful to the algorithm's intent.

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

## Scoring Function Differences from Simulation

These differences exist but should not affect regime detection or relative ordering:

| Scorer | Sim normalization | Real normalization |
|--------|------------------|-------------------|
| ppc | min-max across instances | min-max across endpoints |
| load-aware | `1/(1+EffectiveLoad)` where EffectiveLoad=QD+BS+IFR | `0.5*(1-WaitingQueueSize/threshold)`, range [0, 0.5] |
| active-requests | `1/(1+InFlightRequests)` | min-max of EPP-tracked in-flight counts |
| running-requests | `1/(1+BatchSize)` | min-max of RunningRequestsSize |
| kv-utilization | `1 - KVUtilization` | `1 - KVCacheUsagePercent` |

The real `load-aware-scorer` outputs in range [0, 0.5] while others output [0, 1.0].
This effectively halves its contribution relative to other scorers. If this causes
issues in practice, you can either:
- Double its weight in the regime arrays (e.g., la:2 instead of la:1 in cache-affinity)
- Or accept the difference since load-aware has lower weight than active-requests anyway

---

## Verification

After deploying, check the EPP logs at verbosity 4+ for lines like:
```
"Adaptive-v2 regime selected" regime="cache-affinity" cacheSpread=0.35 avgKVUtil=0.2
```

This confirms the regime detection is running and selecting regimes correctly.
