# Transferring Adaptive-v2 to Production (GAIE/llm-d)

## Part 1: How Scoring Works Today

### The Big Picture

When a request arrives, the scheduler picks which GPU instance should handle it.
It does this by scoring every available instance and picking the highest score.

The scoring pipeline has three stages:

```
Request arrives
    |
    v
[Filters]  -- remove instances that can't serve this request
    |
    v
[Scorers]  -- score remaining instances (0 to 1 each), multiply by weight, sum up
    |
    v
[Picker]   -- pick the instance with the highest total score
```

### Where the Scoring Happens

The weighted score accumulation lives in one function in GAIE (the framework llm-d imports):

**File:** `gateway-api-inference-extension/pkg/epp/scheduling/scheduler_profile.go`
**Function:** `runScorerPlugins` (lines 151-174)

```go
func (p *SchedulerProfile) runScorerPlugins(..., endpoints []Endpoint) map[Endpoint]float64 {
    weightedScorePerEndpoint := make(map[Endpoint]float64, len(endpoints))

    for _, scorer := range p.scorers {
        scores := scorer.Score(ctx, cycleState, request, endpoints)
        for endpoint, score := range scores {
            weightedScorePerEndpoint[endpoint] += enforceScoreRange(score) * scorer.Weight()
        }
    }
    return weightedScorePerEndpoint
}
```

In plain English: for each scorer, get a score per endpoint, multiply by that
scorer's static weight, add it to the running total. That's it.

### The 2:1:1 Baseline

The current production default uses three scorers with fixed weights:

| Scorer | Weight | What it measures |
|--------|--------|-----------------|
| precise-prefix-cache | 2 | How much of the request's prefix is cached on this instance (0-1) |
| queue-depth | 1 | How full the waiting queue is (0-1, lower queue = higher score) |
| kv-utilization | 1 | How full the GPU memory is (0-1, lower usage = higher score) |

These weights never change. Cache always gets 2x priority. This is the problem:
when many requests share a prefix, they all pile onto the one instance that
cached it, because the cache score (weight 2) overpowers the queue score (weight 1).

### How llm-d Wires It Up

llm-d does NOT contain scoring logic itself. It only provides scorer plugins.
The flow is:

1. `cmd/epp/main.go` calls `plugins.RegisterAllPlugins()` -- registers scorer
   factories (load-aware, active-request, precise-prefix-cache, etc.)
2. GAIE's runner reads the YAML config, creates scorer instances via factories,
   wraps each in `WeightedScorer` with the configured weight
3. At request time, GAIE's `runScorerPlugins` runs the static weighted sum

The YAML config controls which scorers run and their weights:

```yaml
# Current production config (2:1:1 baseline)
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: single-profile-handler
  - type: decode-filter
  - type: precise-prefix-cache-scorer
    parameters:
      tokenProcessorConfig:
        blockSize: 64
        hashSeed: "42"
      indexerConfig:
        kvBlockIndexConfig:
          enableMetrics: true
  - type: kv-cache-utilization-scorer
  - type: queue-scorer
  - type: max-score-picker
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: decode-filter
      - pluginRef: precise-prefix-cache-scorer
        weight: 2.0
      - pluginRef: kv-cache-utilization-scorer
        weight: 1.0
      - pluginRef: queue-scorer
        weight: 1.0
      - pluginRef: max-score-picker
```

### What Adaptive-v2 Changes

Instead of fixed weights, adaptive-v2 checks two signals before each request
and picks one of three weight profiles ("regimes"):

**Signal 1 -- Cache spread:** Look at the prefix-cache scores across all
instances. If one instance scores much higher than others (spread > 0.1),
it means cache hits are concentrated -- worth routing there.

**Signal 2 -- Memory pressure:** Average KV cache utilization across instances.
If above 70%, GPU memory is tight -- stop chasing cache hits and balance load.

| Condition | Regime | Weights (cache : load : kv) |
|-----------|--------|-----------------------------|
| Cache spread > 0.1 | Cache-affinity | 4 : 1 : 0 |
| Avg KV util > 0.7 | Memory-aware | 0 : 1 : 1 |
| Otherwise | Load-balance | 0 : 1 : 0 |

The key difference from baseline: adaptive-v2 uses `load-aware` instead of
`queue-depth`. Both are available in llm-d. The formulas are identical between
BLIS and llm-d (0.5 * (1 - queue/128), range [0, 0.5]).

---

## Part 2: How to Transfer Adaptive-v2 to GAIE

### Approach: Modify `runScorerPlugins`

We modify one function in GAIE. Instead of multiplying each scorer's output
by its static weight, we first collect all scores, detect the regime, then
apply regime-specific weights. Same pattern we used in BLIS.

### Step 1: Fork GAIE locally

```bash
# In your go.mod, replace the GAIE import with a local copy
cd your-llm-d-repo
go mod edit -replace sigs.k8s.io/gateway-api-inference-extension=../gateway-api-inference-extension
```

### Step 2: Modify `runScorerPlugins`

**File:** `pkg/epp/scheduling/scheduler_profile.go`

Replace the `runScorerPlugins` function (lines 151-174) with:

```go
func (p *SchedulerProfile) runScorerPlugins(ctx context.Context, request *fwksched.LLMRequest,
    cycleState *fwksched.CycleState, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {

    logger := log.FromContext(ctx)
    logger.V(logutil.DEBUG).Info("Before running scorer plugins", "endpoints", endpoints)

    weightedScorePerEndpoint := make(map[fwksched.Endpoint]float64, len(endpoints))
    for _, endpoint := range endpoints {
        weightedScorePerEndpoint[endpoint] = 0
    }

    // Step 1: Collect all scorer outputs first (don't accumulate yet).
    allScores := make([]map[fwksched.Endpoint]float64, len(p.scorers))
    for i, scorer := range p.scorers {
        logger.V(logutil.VERBOSE).Info("Running scorer plugin", "plugin", scorer.TypedName())
        before := time.Now()
        allScores[i] = scorer.Score(ctx, cycleState, request, endpoints)
        metrics.RecordPluginProcessingLatency(scorerExtensionPoint, scorer.TypedName().Type,
            scorer.TypedName().Name, time.Since(before))
    }

    // Step 2: Detect regime from cache scores + endpoint metrics.
    regimeWeights := p.detectRegime(endpoints, allScores)

    // Step 3: Accumulate with regime-specific weights.
    for i := range p.scorers {
        for endpoint, score := range allScores[i] {
            weightedScorePerEndpoint[endpoint] += enforceScoreRange(score) * regimeWeights[i]
        }
    }

    logger.V(logutil.VERBOSE).Info("Completed running scorer plugins successfully")
    return weightedScorePerEndpoint
}

// detectRegime checks cache spread and KV pressure to select per-request weights.
// Returns one weight per scorer (same length as p.scorers).
// Falls back to static weights if regime detection is not applicable.
//
// Scorer ordering (must match YAML config):
//   0 = precise-prefix-cache
//   1 = load-aware
//   2 = kv-utilization
func (p *SchedulerProfile) detectRegime(endpoints []fwksched.Endpoint,
    allScores []map[fwksched.Endpoint]float64) []float64 {

    // If we don't have exactly 3 scorers, fall back to static weights.
    if len(p.scorers) != 3 {
        weights := make([]float64, len(p.scorers))
        for i, s := range p.scorers {
            weights[i] = s.Weight()
        }
        return weights
    }

    // Cache spread: max - min of cache scores (scorer index 0).
    cacheSpread := 0.0
    if len(allScores) > 0 && len(allScores[0]) > 0 {
        minCache, maxCache := 1.0, 0.0
        for _, score := range allScores[0] {
            if score < minCache {
                minCache = score
            }
            if score > maxCache {
                maxCache = score
            }
        }
        cacheSpread = maxCache - minCache
    }

    // Average KV utilization from endpoint metrics.
    avgKVUtil := 0.0
    count := 0
    for _, ep := range endpoints {
        m := ep.GetMetrics()
        if m != nil {
            avgKVUtil += m.KVCacheUsagePercent / 100.0
            count++
        }
    }
    if count > 0 {
        avgKVUtil /= float64(count)
    }

    // Select regime weights: [cache, load-aware, kv-utilization]
    var weights [3]float64
    switch {
    case cacheSpread > 0.1:
        weights = [3]float64{4.0, 1.0, 0.0}   // cache-affinity
    case avgKVUtil > 0.7:
        weights = [3]float64{0.0, 1.0, 1.0}   // memory-aware
    default:
        weights = [3]float64{0.0, 1.0, 0.0}   // load-balance
    }

    // Normalize to sum to 1.
    sum := 0.0
    for _, w := range weights {
        sum += w
    }
    result := make([]float64, 3)
    for i, w := range weights {
        result[i] = w / sum
    }
    return result
}
```

### Step 3: Update the YAML Config

Replace `queue-scorer` with `load-aware-scorer`. The YAML weights don't matter
at runtime (regime detection overrides them), but they're needed for
instantiation:

```yaml
# Adaptive-v2 config
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: single-profile-handler
  - type: decode-filter
  - type: precise-prefix-cache-scorer
    parameters:
      tokenProcessorConfig:
        blockSize: 64
        hashSeed: "42"
      indexerConfig:
        kvBlockIndexConfig:
          enableMetrics: true
  - type: load-aware-scorer
  - type: kv-cache-utilization-scorer
  - type: max-score-picker
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: decode-filter
      - pluginRef: precise-prefix-cache-scorer
        weight: 1.0
      - pluginRef: load-aware-scorer
        weight: 1.0
      - pluginRef: kv-cache-utilization-scorer
        weight: 1.0
      - pluginRef: max-score-picker
```

**Important:** Scorer order in the YAML determines the index in `detectRegime`:
- Index 0 = precise-prefix-cache-scorer (used for cacheSpread)
- Index 1 = load-aware-scorer
- Index 2 = kv-cache-utilization-scorer

### Step 4: Build and Deploy

```bash
cd your-llm-d-repo
go build ./...
# Deploy as usual -- the only change is the GAIE fork + new YAML
```

### Step 5: Run Baseline Comparison

Deploy two configurations side by side:
1. **Baseline 2:1:1:** Original YAML with `queue-scorer` and weights 2:1:1
2. **Adaptive-v2:** New YAML with `load-aware-scorer` and the modified GAIE

Use the same workload patterns from BLIS simulation:
- FM-2a: Multiple prefix groups across instances (multi-tenant)
- FM-3: Burst traffic with shared prefixes
- FM-6: Cold traffic with high KV pressure

---

## Part 3: What Changes vs What Stays the Same

| Component | Changes? | Details |
|-----------|----------|---------|
| Scorers | No | Same scorers, same code, same formulas |
| Scorer YAML weights | Ignored | Regime detection overrides them at runtime |
| `runScorerPlugins` | Yes | Collects scores first, then applies regime weights |
| `detectRegime` | New | ~40 lines, reads cache scores + KV metrics |
| Filters | No | Unchanged |
| Picker | No | Still picks max score |
| llm-d code | No | Only GAIE fork is modified |

**Total code change:** ~60 lines in one file (`scheduler_profile.go`).

---

## Part 4: Signal Mapping (BLIS to Production)

| BLIS Signal | Production Signal | Source |
|-------------|-------------------|--------|
| `snap.KVUtilization` | `endpoint.GetMetrics().KVCacheUsagePercent / 100.0` | Endpoint metrics |
| `snap.QueueDepth` | `endpoint.GetMetrics().WaitingQueueSize` | Endpoint metrics |
| Cache scores (scorer index 0) | `allScores[0][endpoint]` | PrecisePrefixCacheScorer output |
| `cacheSpread` | `max(cacheScores) - min(cacheScores)` | Computed from scorer output |
| `avgKVUtil` | `mean(KVCacheUsagePercent/100)` | Computed from endpoint metrics |

All signals are available. No new metrics or infrastructure needed.

---

## Part 5: Alternative Approach — Mega-Scorer Plugin

Instead of modifying GAIE, you can put all the logic into a single scorer
plugin that lives entirely in llm-d. No framework fork needed.

### The Idea

Register one custom scorer as the **only** scorer with weight 1.0. Inside that
scorer, compute all three dimensions (cache, load, kv) yourself, run regime
detection, and return a single composite score per endpoint. From GAIE's
perspective it's just one scorer returning one number -- it doesn't know
there's regime detection happening inside.

```
GAIE sees:         [AdaptiveV2Scorer] x weight 1.0  -->  pick max
                         |
Inside the scorer: compute cache + load + kv scores
                   detect regime
                   apply regime weights
                   return composite score
```

### What the Scorer Computes Internally

The scorer reads endpoint metrics directly and computes two of the three
dimensions itself:

**Load-aware** (from metrics):
```go
// Same formula as llm-d's LoadAware scorer
queue := endpoint.GetMetrics().WaitingQueueSize
loadScore = 0.5 * (1.0 - min(queue, 128) / 128.0)
```

**KV-utilization** (from metrics):
```go
kvUtil := endpoint.GetMetrics().KVCacheUsagePercent / 100.0
kvScore = 1.0 - kvUtil
```

**Cache** (the hard part): The prefix cache score requires the `kvCacheIndexer`
-- a complex subsystem with ZMQ event streams, tokenizers, and a block index.
You can't just read it from `GetMetrics()`.

### Solving the Cache Signal

Two options:

**Option A: Inject PrecisePrefixCacheScorer as a dependency.** Your scorer's
factory creates a PrecisePrefixCacheScorer internally and calls its `Score()`
method directly:

```go
type AdaptiveV2Scorer struct {
    prefixScorer *PrecisePrefixCacheScorer  // created in factory
    loadThreshold float64
    // ...
}

func (s *AdaptiveV2Scorer) Score(ctx, cycleState, request, endpoints) map[Endpoint]float64 {
    // Get real cache scores from the prefix scorer
    cacheScores := s.prefixScorer.Score(ctx, cycleState, request, endpoints)

    // Compute load-aware and kv-util from metrics
    // ... (formulas above)

    // Regime detection + composite scoring
    // ... (same logic as BLIS adaptive-v2)
}
```

This gives you real cache scores, so all three regimes work correctly.

**Option B: Run without cache signal.** Skip the cache dimension entirely.
The scorer computes load-aware and kv-utilization from metrics. Cache scores
are always 0, so cacheSpread is always 0, and the cache-affinity regime
never activates. You effectively run in load-balance or memory-aware regime
only.

This still beats the baseline on FM-3 (burst) and FM-6 (cold traffic) where
the wins come from load balancing and memory awareness -- not cache affinity.
You lose the FM-2a wins (multi-tenant prefix routing) until you wire in the
cache signal.

### YAML Config for Mega-Scorer

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: single-profile-handler
  - type: decode-filter
  - type: blis-weighted-scoring          # the mega-scorer
    parameters:
      loadThreshold: 128
      cacheSpreadThreshold: 0.1
      kvPressureThreshold: 0.7
      enabled: true
  - type: max-score-picker
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: decode-filter
      - pluginRef: blis-weighted-scoring
        weight: 1.0
      - pluginRef: max-score-picker
```

### Code Changes for Mega-Scorer

| File | Change |
|------|--------|
| `pkg/plugins/scorer/blis_weighted_scoring.go` | New file (~250 lines) -- the scorer plugin |
| `pkg/plugins/register.go` | Add one line: `plugin.Register(scorer.BLISWeightedScoringType, scorer.BLISWeightedScoringFactory)` |

No GAIE changes. No other llm-d changes.

### Trade-offs: GAIE Modification vs Mega-Scorer

| | GAIE Modification (Part 2) | Mega-Scorer (Part 5) |
|---|---|---|
| **Cache signal** | Real scores from PrecisePrefixCache | Needs manual wiring (Option A) or missing (Option B) |
| **Code location** | GAIE fork (1 file, ~60 lines) | llm-d only (1 new file, ~250 lines) |
| **Scorer accuracy** | Uses the real scorers as-is | Reimplements load-aware and kv-util formulas |
| **Framework changes** | Yes (fork GAIE) | No |
| **All regimes work** | Yes | Only with Option A (cache injected) |
| **Upstreaming** | Harder (shared framework) | Easier (llm-d plugin only) |
| **Confidence** | Higher -- same scorers, same pipeline | Slightly lower -- reimplemented formulas |
| **Speed to results** | Fast (if you can fork GAIE) | Fast (no external dependency) |

**Recommendation:** Use the GAIE modification (Part 2) for experiments where you
want high-confidence results matching BLIS simulation. Use the mega-scorer if
you cannot fork GAIE or need to ship something entirely within llm-d. Or
read Part 6 below for the best of both worlds.

---

## Part 6: Recommended Approach — Custom SchedulerProfile (no GAIE fork)

This is the cleanest path. It uses real scorers (no reimplementation), has
access to all endpoints and metrics, supports all three regimes including
cache-affinity, and lives entirely in llm-d with zero GAIE changes.

### Why This Works

`SchedulerProfile` is an **interface** in GAIE, not a concrete struct:

```go
// In gateway-api-inference-extension/pkg/epp/framework/interface/scheduling/types.go
type SchedulerProfile interface {
    Run(ctx context.Context, request *LLMRequest, cycleState *CycleState,
        candidateEndpoints []Endpoint) (*ProfileRunResult, error)
}
```

The scheduler just calls `profile.Run(...)` — it doesn't care what's inside.
Anyone can implement this interface.

GAIE's runner also has a builder method to inject a custom scheduler config:

```go
// In gateway-api-inference-extension/cmd/epp/runner/runner.go:136
func (r *Runner) WithSchedulerConfig(schedulerConfig *scheduling.SchedulerConfig) *Runner
```

When you provide a `SchedulerConfig`, the runner uses it directly and skips
the YAML config loader for scheduling. So llm-d can build its own
`SchedulerConfig` containing a custom `SchedulerProfile` that does regime
detection — all without touching GAIE.

### How It Fits Together

```
llm-d main.go
    |
    |  builds AdaptiveV2Profile (implements SchedulerProfile interface)
    |  with real scorer instances: PrecisePrefixCache, LoadAware, KVUtilization
    |
    v
runner.NewRunner().
    WithSchedulerConfig(config with AdaptiveV2Profile).
    Run(ctx)
    |
    v
On each request, scheduler calls AdaptiveV2Profile.Run():
    1. Run filters (same as baseline)
    2. Run all 3 real scorers, collect their outputs
    3. Compute cacheSpread + avgKVUtil → pick regime
    4. Apply regime weights to scorer outputs
    5. Pick endpoint with highest composite score
```

### Step 1: Create the Adaptive Profile

Create one new file in llm-d:

**File:** `pkg/plugins/profile/adaptive_v2_profile.go`

```go
package profile

import (
    "context"
    "time"

    "sigs.k8s.io/controller-runtime/pkg/log"
    logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
    fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
    "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
    "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
)

// AdaptiveV2Profile implements SchedulerProfile with per-request regime detection.
// It runs the same Filter → Score → Pick pipeline as the default profile,
// but replaces the static weighted sum with adaptive regime-specific weights.
type AdaptiveV2Profile struct {
    filters []fwksched.Filter
    scorers []*scheduling.WeightedScorer   // [0]=cache, [1]=load-aware, [2]=kv-util
    picker  fwksched.Picker

    cacheSpreadThreshold float64  // default 0.1
    kvPressureThreshold  float64  // default 0.7
}

// NewAdaptiveV2Profile creates a new adaptive-v2 profile.
// Scorers must be in order: [precise-prefix-cache, load-aware, kv-utilization].
func NewAdaptiveV2Profile(
    filters []fwksched.Filter,
    scorers []*scheduling.WeightedScorer,
    picker fwksched.Picker,
) *AdaptiveV2Profile {
    return &AdaptiveV2Profile{
        filters:              filters,
        scorers:              scorers,
        picker:               picker,
        cacheSpreadThreshold: 0.1,
        kvPressureThreshold:  0.7,
    }
}

// Run implements SchedulerProfile. Runs filters, then adaptive scoring, then picker.
func (p *AdaptiveV2Profile) Run(ctx context.Context, request *fwksched.LLMRequest,
    cycleState *fwksched.CycleState, candidateEndpoints []fwksched.Endpoint,
) (*fwksched.ProfileRunResult, error) {
    logger := log.FromContext(ctx)

    // --- Filters (identical to default profile) ---
    endpoints := candidateEndpoints
    for _, filter := range p.filters {
        endpoints = filter.Filter(ctx, cycleState, request, endpoints)
        if len(endpoints) == 0 {
            return nil, fmt.Errorf("no endpoints after filter %s", filter.TypedName())
        }
    }

    // --- Adaptive Scoring ---
    weightedScores := make(map[fwksched.Endpoint]float64, len(endpoints))
    for _, ep := range endpoints {
        weightedScores[ep] = 0
    }

    // Step 1: Run all scorers, collect their per-endpoint scores.
    allScores := make([]map[fwksched.Endpoint]float64, len(p.scorers))
    for i, scorer := range p.scorers {
        before := time.Now()
        allScores[i] = scorer.Score(ctx, cycleState, request, endpoints)
        metrics.RecordPluginProcessingLatency("Scorer",
            scorer.TypedName().Type, scorer.TypedName().Name, time.Since(before))
    }

    // Step 2: Regime detection.
    regimeWeights, regime := p.detectRegime(endpoints, allScores)
    logger.V(logutil.VERBOSE).Info("Adaptive-v2 regime selected", "regime", regime)

    // Step 3: Apply regime weights to scorer outputs.
    for i, scores := range allScores {
        for ep, score := range scores {
            // Clamp to [0, 1] like GAIE's enforceScoreRange.
            if score < 0 {
                score = 0
            }
            if score > 1 {
                score = 1
            }
            weightedScores[ep] += score * regimeWeights[i]
        }
    }

    // --- Picker (identical to default profile) ---
    scoredEndpoints := make([]*fwksched.ScoredEndpoint, 0, len(weightedScores))
    for ep, score := range weightedScores {
        scoredEndpoints = append(scoredEndpoints, &fwksched.ScoredEndpoint{
            Endpoint: ep, Score: score,
        })
    }

    result := p.picker.Pick(ctx, cycleState, scoredEndpoints)
    return result, nil
}

// detectRegime checks cache spread and KV pressure, returns regime weights and name.
// Scorer ordering: [0]=precise-prefix-cache, [1]=load-aware, [2]=kv-utilization.
func (p *AdaptiveV2Profile) detectRegime(endpoints []fwksched.Endpoint,
    allScores []map[fwksched.Endpoint]float64) ([]float64, string) {

    // If we don't have exactly 3 scorers, fall back to equal weights.
    if len(p.scorers) != 3 {
        w := make([]float64, len(p.scorers))
        for i := range w {
            w[i] = 1.0 / float64(len(p.scorers))
        }
        return w, "fallback"
    }

    // Cache spread: max - min of prefix-cache scores (index 0).
    cacheSpread := 0.0
    if len(allScores[0]) > 0 {
        minC, maxC := 1.0, 0.0
        for _, s := range allScores[0] {
            if s < minC {
                minC = s
            }
            if s > maxC {
                maxC = s
            }
        }
        cacheSpread = maxC - minC
    }

    // Average KV utilization from endpoint metrics.
    avgKVUtil := 0.0
    count := 0
    for _, ep := range endpoints {
        if m := ep.GetMetrics(); m != nil {
            avgKVUtil += m.KVCacheUsagePercent / 100.0
            count++
        }
    }
    if count > 0 {
        avgKVUtil /= float64(count)
    }

    // Select regime.
    var weights [3]float64
    var regime string
    switch {
    case cacheSpread > p.cacheSpreadThreshold:
        weights = [3]float64{4.0, 1.0, 0.0}
        regime = "cache-affinity"
    case avgKVUtil > p.kvPressureThreshold:
        weights = [3]float64{0.0, 1.0, 1.0}
        regime = "memory-aware"
    default:
        weights = [3]float64{0.0, 1.0, 0.0}
        regime = "load-balance"
    }

    // Normalize to sum to 1.
    sum := 0.0
    for _, w := range weights {
        sum += w
    }
    result := make([]float64, 3)
    for i := range weights {
        result[i] = weights[i] / sum
    }
    return result, regime
}
```

### Step 2: Wire It Up in main.go

Modify `cmd/epp/main.go` to build the adaptive profile and inject it via
`WithSchedulerConfig`:

**File:** `cmd/epp/main.go`

```go
package main

import (
    "context"
    "os"

    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/gateway-api-inference-extension/cmd/epp/runner"
    fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
    "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/picker"
    "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"

    "github.com/llm-d/llm-d-inference-scheduler/pkg/metrics"
    "github.com/llm-d/llm-d-inference-scheduler/pkg/plugins"
    "github.com/llm-d/llm-d-inference-scheduler/pkg/plugins/filter"
    "github.com/llm-d/llm-d-inference-scheduler/pkg/plugins/profile"
    "github.com/llm-d/llm-d-inference-scheduler/pkg/plugins/scorer"
    "github.com/llm-d/llm-d-inference-scheduler/pkg/telemetry"
)

func main() {
    os.Exit(run())
}

func run() int {
    ctx := ctrl.SetupSignalHandler()

    shutdownTracing, err := telemetry.InitTracing(ctx)
    if err != nil {
        ctrl.Log.Error(err, "Failed to initialize tracing")
    }
    if shutdownTracing != nil {
        defer func() {
            if err := shutdownTracing(ctx); err != nil {
                ctrl.Log.Error(err, "Failed to shutdown tracing")
            }
        }()
    }

    // Register all plugins (still needed for non-scheduling plugins).
    plugins.RegisterAllPlugins()

    // --- Build the adaptive-v2 scheduler config ---
    schedulerConfig, err := buildAdaptiveV2Config(ctx)
    if err != nil {
        ctrl.Log.Error(err, "Failed to build adaptive-v2 scheduler config")
        return 1
    }

    if err := runner.NewRunner().
        WithSchedulerConfig(schedulerConfig).
        WithCustomCollectors(metrics.GetCollectors()...).
        Run(ctx); err != nil {
        return 1
    }
    return 0
}

// buildAdaptiveV2Config creates a SchedulerConfig with the adaptive-v2 profile.
// This bypasses the YAML config loader for scheduling -- the profile is built
// in code with the exact scorers and regime detection logic we need.
func buildAdaptiveV2Config(ctx context.Context) (*scheduling.SchedulerConfig, error) {

    // 1. Create the real scorers (same ones llm-d already uses).
    //    Adjust PrecisePrefixCacheScorer parameters to match your deployment.
    prefixCacheScorer, err := scorer.New(ctx, scorer.PrecisePrefixCachePluginConfig{
        // Fill in your deployment's tokenizer and indexer config here.
        // Same parameters as the YAML config's precise-prefix-cache-scorer section.
    })
    if err != nil {
        return nil, fmt.Errorf("failed to create prefix cache scorer: %w", err)
    }

    loadAwareScorer := scorer.NewLoadAware(ctx, scorer.QueueThresholdDefault)

    // kv-cache-utilization-scorer is a GAIE built-in, import it:
    //   "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/scorer/kvutilization"
    kvUtilScorer := kvutilization.New()

    // 2. Wrap in WeightedScorer (weights don't matter -- regime detection overrides).
    weightedScorers := []*scheduling.WeightedScorer{
        scheduling.NewWeightedScorer(prefixCacheScorer, 1.0),   // index 0
        scheduling.NewWeightedScorer(loadAwareScorer, 1.0),     // index 1
        scheduling.NewWeightedScorer(kvUtilScorer, 1.0),        // index 2
    }

    // 3. Create filter and picker.
    decodeFilter := filter.NewDecodeFilter()
    maxScorePicker := picker.NewMaxScorePicker()

    // 4. Build the adaptive profile.
    adaptiveProfile := profile.NewAdaptiveV2Profile(
        []fwksched.Filter{decodeFilter},
        weightedScorers,
        maxScorePicker,
    )

    // 5. Build a profile handler (single profile, same as baseline).
    profileHandler := profile.NewDataParallelProfileHandler(8000) // or use SingleProfileHandler

    // 6. Assemble the config.
    config := scheduling.NewSchedulerConfig(
        profileHandler,
        map[string]fwksched.SchedulerProfile{
            "default": adaptiveProfile,
        },
    )

    return config, nil
}
```

**Note:** The exact imports for GAIE built-in scorers (like `kv-cache-utilization`)
and filters may vary depending on your GAIE version. Check the GAIE source under
`pkg/epp/framework/plugins/scheduling/scorer/` for the correct import paths.

### Step 3: No YAML Config Needed for Scheduling

When you use `WithSchedulerConfig`, the runner skips the YAML config loader
for scheduling entirely. Your EPP config YAML only needs non-scheduling
settings (if any). The scorers, weights, and regime detection are all defined
in code.

If your deployment requires other YAML-configured plugins (like data sources
or pre-request handlers), those still work -- only the scheduling profile is
overridden.

### Step 4: Build and Deploy

```bash
cd your-llm-d-repo
go build ./...
# Deploy as usual
```

### Step 5: Run Baseline Comparison

To compare baseline vs adaptive-v2, deploy two EPP instances:

1. **Baseline 2:1:1:** Stock llm-d with the standard YAML config
   (precise-prefix-cache:2, queue-scorer:1, kv-cache-utilization:1)
2. **Adaptive-v2:** Modified llm-d with the `buildAdaptiveV2Config` code above

Run the same traffic against both and compare E2E and TTFT latencies.
Focus on workloads matching the simulation scenarios:
- Multi-tenant (many prefix groups, few instances) -- tests cache-affinity regime
- Burst traffic -- tests load-balance regime under stale snapshots
- High KV pressure -- tests memory-aware regime

---

## Part 7: Comparison of All Three Approaches

| | Part 2: GAIE Fork | Part 5: Mega-Scorer | Part 6: Custom Profile |
|---|---|---|---|
| **GAIE changes** | Yes (fork 1 file) | No | No |
| **llm-d changes** | YAML only | 1 new file + 1 line in register.go | 1 new file + modify main.go |
| **Uses real scorers** | Yes | No (reimplements load + kv) | Yes |
| **Cache-affinity works** | Yes | Only with manual wiring | Yes |
| **All regimes work** | Yes | Depends on cache signal | Yes |
| **Scorer accuracy** | Exact | Approximated | Exact |
| **Upstreamable** | Hard (shared framework) | Easy (llm-d plugin) | Easy (llm-d only) |
| **Confidence level** | Highest | Medium | Highest |
| **Complexity** | Low (~60 lines) | Medium (~250 lines) | Medium (~150 lines) |

**Recommendation:** Use Part 6 (Custom Profile). It gives the same confidence
as the GAIE fork (real scorers, all regimes) without touching GAIE. The code
lives entirely in llm-d and is easy to maintain.
