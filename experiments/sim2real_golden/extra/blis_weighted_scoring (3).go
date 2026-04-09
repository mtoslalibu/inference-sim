package scorer

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	BLISWeightedScoringType = "blis-weighted-scoring"
)

var _ scheduling.Scorer = &BLISWeightedScorer{}

// BLISWeightedScorerParameters configures the adaptive-v2 regime-detection scorer.
type BLISWeightedScorerParameters struct {
	LoadThreshold        int     `json:"loadThreshold"`
	CacheSpreadThreshold float64 `json:"cacheSpreadThreshold"`
	KVPressureThreshold  float64 `json:"kvPressureThreshold"`
	Enabled              bool    `json:"enabled"`
}

// BLISWeightedScorer implements the adaptive-v2 3-regime scoring algorithm evolved
// by BLIS. Computes three scoring dimensions (cache, load-aware, kv-utilization) and
// selects per-request regime-specific weights based on cache spread and memory pressure.
//
// Regime detection (from sim router_adaptive_v2.go:237-250):
//   - Cache-affinity (cacheSpread > threshold): weights [4, 1, 0] for [cache, load, kv]
//   - Memory-aware (avgKVUtil > threshold):     weights [0, 1, 1]
//   - Load-balance (default):                   weights [0, 1, 0]
//
// Signal mapping (sim -> production):
//   - QueueDepth       -> WaitingQueueSize
//   - BatchSize        -> zeroed in production (no separate metric)
//   - InFlightRequests -> RunningRequestsSize
//   - KVUtilization    -> KVCacheUsagePercent / 100.0
//   - EffectiveLoad    -> WaitingQueueSize + RunningRequestsSize
//     (Sim EffectiveLoad() = QueueDepth + BatchSize + InFlightRequests.
//     BatchSize is zeroed in production -- RunningRequestsSize already captures
//     the full running request count. The production equivalent is therefore
//     WaitingQueueSize + RunningRequestsSize with a 1x coefficient.)
//
// Note: CacheHitRate is not available as an endpoint metric; the cache dimension
// is always 0. Cache-affinity regime cannot activate. Deploy PrecisePrefixCache
// scorer alongside this plugin to handle cache affinity at the framework level.
//
// Approximation: The load-aware dimension uses the production LoadAware formula
// (0.5 * (1 - capped/threshold)) applied to EffectiveLoad. The sim delegates to
// individual scorerFunc instances whose internal formulas are not directly exposed
// in the EVOLVE-BLOCK. This is a best-effort approximation using the production
// reference scorer's approach.
type BLISWeightedScorer struct {
	typedName            plugin.TypedName
	enabled              bool
	loadThreshold        float64
	cacheSpreadThreshold float64
	kvPressureThreshold  float64
}

// BLISWeightedScoringFactory creates a BLISWeightedScorer from config parameters.
func BLISWeightedScoringFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	params := BLISWeightedScorerParameters{
		LoadThreshold:        128,
		CacheSpreadThreshold: 0.1,
		KVPressureThreshold:  0.7,
		Enabled:              true,
	}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &params); err != nil {
			return nil, fmt.Errorf("failed to parse parameters for '%s' scorer: %w", BLISWeightedScoringType, err)
		}
	}
	return NewBLISWeightedScorer(handle.Context(), params).WithName(name), nil
}

// NewBLISWeightedScorer creates a BLISWeightedScorer with the given parameters.
func NewBLISWeightedScorer(ctx context.Context, params BLISWeightedScorerParameters) *BLISWeightedScorer {
	if params.LoadThreshold <= 0 {
		params.LoadThreshold = 128
		log.FromContext(ctx).V(logutil.DEFAULT).Info("loadThreshold must be positive, using default 128")
	}
	if params.CacheSpreadThreshold <= 0 {
		params.CacheSpreadThreshold = 0.1
	}
	if params.KVPressureThreshold <= 0 {
		params.KVPressureThreshold = 0.7
	}
	return &BLISWeightedScorer{
		typedName:            plugin.TypedName{Type: BLISWeightedScoringType},
		enabled:              params.Enabled,
		loadThreshold:        float64(params.LoadThreshold),
		cacheSpreadThreshold: params.CacheSpreadThreshold,
		kvPressureThreshold:  params.KVPressureThreshold,
	}
}

func (s *BLISWeightedScorer) TypedName() plugin.TypedName {
	return s.typedName
}

func (s *BLISWeightedScorer) WithName(name string) *BLISWeightedScorer {
	s.typedName.Name = name
	return s
}

func (s *BLISWeightedScorer) Category() scheduling.ScorerCategory {
	return scheduling.Distribution
}

// Score implements scheduling.Scorer. Computes composite scores using adaptive-v2
// regime detection with three scoring dimensions.
func (s *BLISWeightedScorer) Score(_ context.Context, _ *scheduling.CycleState, _ *scheduling.LLMRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	if !s.enabled {
		return nil
	}

	n := len(endpoints)
	scoredEndpoints := make(map[scheduling.Endpoint]float64, n)

	// Per-endpoint scoring dimensions (matching sim scorer ordering):
	//   0 = cache (precise-prefix-cache) -- unavailable from metrics, always 0
	//   1 = load-aware -- based on EffectiveLoad (approximation, see type doc)
	//   2 = kv-utilization -- based on normalized KVCacheUsagePercent
	type dimScores struct {
		cache      float64
		load       float64
		kv         float64
		kvUtil     float64 // raw normalized KV for regime detection
		hasMetrics bool    // false when metrics is nil; excluded from regime detection
	}
	epDims := make([]dimScores, n)

	for i, endpoint := range endpoints {
		metrics := endpoint.GetMetrics()
		if metrics == nil {
			// Nil metrics: score 0.0 and exclude from regime detection.
			epDims[i] = dimScores{hasMetrics: false}
			continue
		}

		// EffectiveLoad = WaitingQueueSize + RunningRequestsSize
		// Sim: EffectiveLoad() = QueueDepth + BatchSize + InFlightRequests.
		// In production, BatchSize is zeroed -- RunningRequestsSize already captures
		// the full running request count (equivalent to InFlightRequests alone).
		// The production formula uses a 1x coefficient on RunningRequestsSize.
		// [R4-FIX-1] Corrected from R3's 2x coefficient: BatchSize is zeroed in
		// production, so the F-10 double-counting fix applies -- use 1x.
		effectiveLoad := float64(metrics.WaitingQueueSize) + float64(metrics.RunningRequestsSize)

		// KVUtilization = KVCacheUsagePercent / 100.0 (normalize 0-100 to 0.0-1.0)
		kvUtil := metrics.KVCacheUsagePercent / 100.0

		// Load-aware dimension: approximation using production LoadAware formula
		// applied to EffectiveLoad. Range [0, 0.5], 0.5 at zero load.
		// The sim delegates to scorerFunc instances whose internal formula
		// is not part of the EVOLVE-BLOCK; this uses the production reference scorer
		// approach (load_aware.go). This is a documented best-effort approximation.
		var loadScore float64
		if effectiveLoad == 0 {
			loadScore = 0.5
		} else {
			capped := min(effectiveLoad, s.loadThreshold)
			loadScore = 0.5 * (1.0 - capped/s.loadThreshold)
		}

		// KV-utilization dimension: lower utilization = higher score.
		kvScore := max(0.0, min(1.0, 1.0-kvUtil))

		epDims[i] = dimScores{load: loadScore, kv: kvScore, kvUtil: kvUtil, hasMetrics: true}
	}

	// --- Regime detection ---
	// Only endpoints with valid metrics participate in regime detection.

	// Cache spread: max - min of cache scores across endpoints with metrics.
	// Cache scores are always 0 (signal unavailable), so cacheSpread is always 0 and
	// cache-affinity regime cannot activate. This is a known functional gap; deploy
	// PrecisePrefixCache scorer alongside to handle cache affinity at framework level.
	minCache, maxCache := 1.0, 0.0
	metricsCount := 0
	for i := range n {
		if !epDims[i].hasMetrics {
			continue
		}
		minCache = min(minCache, epDims[i].cache)
		maxCache = max(maxCache, epDims[i].cache)
		metricsCount++
	}
	cacheSpread := 0.0
	if metricsCount > 0 {
		cacheSpread = maxCache - minCache
	}

	// Average KV utilization across endpoints with valid metrics.
	avgKVUtil := 0.0
	for i := range n {
		if !epDims[i].hasMetrics {
			continue
		}
		avgKVUtil += epDims[i].kvUtil
	}
	if metricsCount > 0 {
		avgKVUtil /= float64(metricsCount)
	}

	// Regime weights: [cache, load, kv].
	// Matches sim router_adaptive_v2.go:237-250 regime detection.
	var weights [3]float64
	switch {
	case cacheSpread > s.cacheSpreadThreshold:
		weights = [3]float64{4.0, 1.0, 0.0} // cache-affinity
	case avgKVUtil > s.kvPressureThreshold:
		weights = [3]float64{0.0, 1.0, 1.0} // memory-aware
	default:
		weights = [3]float64{0.0, 1.0, 0.0} // load-balance
	}

	// Normalize weights to sum to 1.
	// Matches sim router_adaptive_v2.go:253-260.
	wSum := 0.0
	for _, w := range weights {
		wSum += w
	}
	if wSum > 0 {
		for i := range weights {
			weights[i] /= wSum
		}
	}

	// Composite scores with regime-specific weights.
	for i, endpoint := range endpoints {
		if !epDims[i].hasMetrics {
			scoredEndpoints[endpoint] = 0.0
			continue
		}
		dims := [3]float64{epDims[i].cache, epDims[i].load, epDims[i].kv}
		composite := 0.0
		for d := range 3 {
			v := max(0.0, min(1.0, dims[d]))
			composite += v * weights[d]
		}
		scoredEndpoints[endpoint] = composite
	}

	return scoredEndpoints
}

// ScoreEndpoints converts Score() output from map[Endpoint]float64 to
// map[string]float64 keyed by endpoint name. Used by equivalence tests.
func ScoreEndpoints(
	ctx context.Context,
	s scheduling.Scorer,
	request *scheduling.LLMRequest,
	endpoints []scheduling.Endpoint,
) map[string]float64 {
	raw := s.Score(ctx, nil, request, endpoints)
	if raw == nil {
		return nil
	}
	result := make(map[string]float64, len(raw))
	for endpoint, score := range raw {
		name := endpoint.GetMetadata().NamespacedName.String()
		if _, exists := result[name]; exists {
			panic(fmt.Sprintf("ScoreEndpoints: duplicate endpoint name %q", name))
		}
		result[name] = score
	}
	return result
}
