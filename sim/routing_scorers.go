package sim

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// ScorerConfig describes a named scorer with a weight for weighted routing.
type ScorerConfig struct {
	Name   string  `yaml:"name"`
	Weight float64 `yaml:"weight"`
}

// scorerFunc computes per-instance scores in [0,1] for a scoring dimension.
// The req parameter provides request metadata (e.g., InputTokens for prefix matching).
// Stateless scorers may ignore it.
type scorerFunc func(req *Request, snapshots []RoutingSnapshot) map[string]float64

// cacheQueryFn maps instance IDs to functions that return the count of
// consecutive cached prefix blocks for given tokens. Used by precise
// prefix cache scoring. Nil for sim-level tests without cluster instances.
type cacheQueryFn map[string]func([]int) int

// validScorerNames maps scorer names to validity. Unexported to prevent mutation (antipattern rule 8).
var validScorerNames = map[string]bool{
	"prefix-affinity":      true,
	"precise-prefix-cache": true,
	"no-hit-lru":           true,
	"queue-depth":          true,
	"kv-utilization":       true,
	"load-balance":         true,
	"load-aware":           true,
	"inflight-requests":    true,
	"active-requests":      true,
	"running-requests":     true,
}

// IsValidScorer returns true if name is a recognized scorer.
func IsValidScorer(name string) bool { return validScorerNames[name] }

// ValidScorerNames returns sorted valid scorer names.
func ValidScorerNames() []string { return validNamesList(validScorerNames) }

// DefaultScorerConfigs returns the default scorer configuration for weighted routing.
// Default profile: precise-prefix-cache:2, queue-depth:1, kv-utilization:1 (llm-d parity).
func DefaultScorerConfigs() []ScorerConfig {
	return []ScorerConfig{
		{Name: "precise-prefix-cache", Weight: 2.0},
		{Name: "queue-depth", Weight: 1.0},
		{Name: "kv-utilization", Weight: 1.0},
	}
}

// ParseScorerConfigs parses a comma-separated string of "name:weight" pairs.
// Returns nil for empty input. Returns error for invalid names, non-positive weights,
// NaN, Inf, or malformed input.
func ParseScorerConfigs(s string) ([]ScorerConfig, error) {
	if s == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	configs := make([]ScorerConfig, 0, len(parts))
	seen := make(map[string]bool, len(parts))
	for _, part := range parts {
		kv := strings.SplitN(strings.TrimSpace(part), ":", 2)
		if len(kv) != 2 {
			return nil, fmt.Errorf("invalid scorer config %q (expected name:weight)", strings.TrimSpace(part))
		}
		name := strings.TrimSpace(kv[0])
		if !IsValidScorer(name) {
			return nil, fmt.Errorf("unknown scorer %q; valid: %s", name, strings.Join(ValidScorerNames(), ", "))
		}
		if seen[name] {
			return nil, fmt.Errorf("duplicate scorer %q; each scorer may appear at most once", name)
		}
		seen[name] = true
		weight, err := strconv.ParseFloat(strings.TrimSpace(kv[1]), 64)
		if err != nil {
			return nil, fmt.Errorf("invalid weight for scorer %q: %w", name, err)
		}
		if weight <= 0 || math.IsNaN(weight) || math.IsInf(weight, 0) {
			return nil, fmt.Errorf("scorer %q weight must be a finite positive number, got %v", name, weight)
		}
		configs = append(configs, ScorerConfig{Name: name, Weight: weight})
	}
	return configs, nil
}

// normalizeScorerWeights returns weights normalized to sum to 1.0.
// Panics if total weight is zero (should be prevented by validation).
func normalizeScorerWeights(configs []ScorerConfig) []float64 {
	total := 0.0
	for _, c := range configs {
		total += c.Weight
	}
	if total <= 0 {
		panic(fmt.Sprintf("scorer weights sum to %f; must be positive", total))
	}
	weights := make([]float64, len(configs))
	for i, c := range configs {
		weights[i] = c.Weight / total
	}
	return weights
}

// newScorerWithObserver creates a scorer function and optional observer for a named scorer.
// Returns (scorer, observer) where observer is nil for stateless scorers.
// blockSize is used by stateful scorers (e.g., prefix-affinity) for block hash computation.
// Panics on unknown name (validation should catch this before reaching here).
func newScorerWithObserver(name string, blockSize int, cacheFn cacheQueryFn) (scorerFunc, observerFunc) {
	switch name {
	case "prefix-affinity":
		return newPrefixAffinityScorer(blockSize)
	case "precise-prefix-cache":
		return newPrecisePrefixCacheScorer(cacheFn)
	case "no-hit-lru":
		return newNoHitLRUScorer(cacheFn)
	case "queue-depth":
		return scoreQueueDepth, nil
	case "kv-utilization":
		return scoreKVUtilization, nil
	case "load-balance":
		return scoreLoadBalance, nil
	case "load-aware":
		return scoreLoadAware, nil
	case "inflight-requests":
		return scoreInFlightRequests, nil
	case "active-requests":
		return scoreActiveRequests, nil
	case "running-requests":
		return scoreRunningRequests, nil
	default:
		panic(fmt.Sprintf("unknown scorer %q", name))
	}
}

// scoreQueueDepth computes per-instance queue depth scores using min-max normalization.
// Lower queue depth → higher score. All-equal depths → all score 1.0.
// Matches llm-d/GIE's queue-scorer semantics: reads QueueDepth only (WaitingQueueSize).
//
// Signal freshness (R17, INV-7):
//
//	Reads: QueueDepth (Periodic when interval>0, else Immediate).
func scoreQueueDepth(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	minDepth, maxDepth := math.MaxInt, 0
	for _, snap := range snapshots {
		depth := snap.QueueDepth
		if depth < minDepth {
			minDepth = depth
		}
		if depth > maxDepth {
			maxDepth = depth
		}
	}
	for _, snap := range snapshots {
		if maxDepth == minDepth {
			scores[snap.ID] = 1.0
		} else {
			depth := snap.QueueDepth
			scores[snap.ID] = float64(maxDepth-depth) / float64(maxDepth-minDepth)
		}
	}
	return scores
}

// scoreKVUtilization computes per-instance KV utilization scores.
// Lower utilization → higher score: score = 1 - KVUtilization.
// Matches llm-d's kv-cache-utilization-scorer semantics.
//
// Signal freshness (R17, INV-7):
//
//	Reads: KVUtilization (Periodic when interval>0, else Immediate).
//	WARNING: At high request rates with large intervals, this signal can be significantly stale.
//	Pair with a load-aware scorer (e.g., queue-depth) for robust routing.
//	See H3 experiment: 200x worse distribution uniformity at rate=5000.
func scoreKVUtilization(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		scores[snap.ID] = 1.0 - snap.KVUtilization
	}
	return scores
}

// scoreLoadBalance computes per-instance load balance scores using inverse transform.
// Lower effective load → higher score: score = 1/(1 + effectiveLoad).
// BLIS-native formula preserving absolute load differences (alternative to min-max).
//
// Signal freshness (R17, INV-7):
//
//	Reads: EffectiveLoad() — same as scoreQueueDepth (synchronous + Periodic composite).
func scoreLoadBalance(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		scores[snap.ID] = 1.0 / (1.0 + float64(snap.EffectiveLoad()))
	}
	return scores
}

// loadAwareQueueThreshold is the default queue depth threshold for the load-aware scorer.
// Matches llm-d's QueueThresholdDefault (128).
const loadAwareQueueThreshold = 128

// scoreLoadAware computes per-instance load-aware scores using llm-d's linear
// threshold-capped formula. Empty queue → 0.5, queue at threshold → 0.0.
// Matches llm-d's load-aware-scorer semantics exactly.
//
// Formula: empty = 0.5, otherwise 0.5 * (1.0 - min(QueueDepth, threshold) / threshold)
// Score range: [0.0, 0.5]
//
// Signal freshness (R17, INV-7):
//
//	Reads: QueueDepth (Periodic when interval>0, else Immediate).
func scoreLoadAware(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		if snap.QueueDepth == 0 {
			scores[snap.ID] = 0.5
		} else {
			clamped := float64(snap.QueueDepth)
			if clamped > loadAwareQueueThreshold {
				clamped = loadAwareQueueThreshold
			}
			scores[snap.ID] = 0.5 * (1.0 - clamped/loadAwareQueueThreshold)
		}
	}
	return scores
}

// scoreInFlightRequests computes per-instance scores using min-max normalization
// on InFlightRequests — the gateway-local synchronous counter of dispatched-but-
// not-completed requests. Lower in-flight count → higher score.
// All-equal counts → all score 1.0.
//
// GIE parity: InFlightLoad.Requests is populated per endpoint by GIE's data layer
// (concurrency.InFlightLoadKey in AttributeMap). Available but not consumed by any
// existing llm-d scorer — token-load-scorer reads InFlightLoad.Tokens instead.
//
// Signal freshness (R17, INV-7):
//
//	Reads: InFlightRequests (Synchronous — updated at gateway on dispatch/completion).
//	This is the only routing signal with zero staleness under periodic snapshot refresh.
func scoreInFlightRequests(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	minIFR, maxIFR := math.MaxInt, 0
	for _, snap := range snapshots {
		ifr := snap.InFlightRequests
		if ifr < minIFR {
			minIFR = ifr
		}
		if ifr > maxIFR {
			maxIFR = ifr
		}
	}
	for _, snap := range snapshots {
		if maxIFR == minIFR {
			scores[snap.ID] = 1.0
		} else {
			ifr := snap.InFlightRequests
			scores[snap.ID] = float64(maxIFR-ifr) / float64(maxIFR-minIFR)
		}
	}
	return scores
}

// scoreActiveRequests computes per-instance scores using llm-d's active-request-scorer
// formula. Instances with 0 in-flight always score 1.0. Busy instances:
// score = (maxCount - count) / maxCount.
//
// Matches llm-d's active_request.go:193-230 (simplified — no TTL cleanup needed in sim).
//
// Signal freshness (R17, INV-7):
//
//	Reads: InFlightRequests (Synchronous — updated at gateway on dispatch/completion).
func scoreActiveRequests(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	maxCount := 0
	for _, snap := range snapshots {
		if snap.InFlightRequests > maxCount {
			maxCount = snap.InFlightRequests
		}
	}
	for _, snap := range snapshots {
		if snap.InFlightRequests == 0 || maxCount == 0 {
			scores[snap.ID] = 1.0
		} else {
			scores[snap.ID] = float64(maxCount-snap.InFlightRequests) / float64(maxCount)
		}
	}
	return scores
}

// scoreRunningRequests computes per-instance scores using min-max normalization
// on BatchSize (running/in-batch request count). Lower batch size → higher score.
// All-equal sizes → all score 1.0.
//
// Matches GIE's running-requests-size-scorer (runningrequest.go:99).
//
// Signal freshness (R17, INV-7):
//
//	Reads: BatchSize (Periodic when interval>0, else Immediate).
func scoreRunningRequests(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	minBatch, maxBatch := math.MaxInt, 0
	for _, snap := range snapshots {
		if snap.BatchSize < minBatch {
			minBatch = snap.BatchSize
		}
		if snap.BatchSize > maxBatch {
			maxBatch = snap.BatchSize
		}
	}
	for _, snap := range snapshots {
		if maxBatch == minBatch {
			scores[snap.ID] = 1.0
		} else {
			scores[snap.ID] = float64(maxBatch-snap.BatchSize) / float64(maxBatch-minBatch)
		}
	}
	return scores
}
