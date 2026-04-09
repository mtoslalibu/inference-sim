
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
