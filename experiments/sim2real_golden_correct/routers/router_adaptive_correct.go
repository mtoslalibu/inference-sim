package sim

// router_adaptive_correct.go — Drop-in replacement for sim/routing.go
//
// Adaptive-correct v3: Adaptive-expanded + load-gated cache affinity.
// Same 5 scorers as expanded, same regimes, but cache-affinity is GATED
// on load balance to prevent pile-on under unbalanced conditions.
//
// CHANGES from adaptive-expanded:
//   1. Cache-affinity regime requires loadSpread < threshold (load gate).
//      When cache spread is high but load is unbalanced, falls through to
//      load-balance regime instead of chasing cache.
//   2. KV pressure threshold lowered to 0.6 (from 0.7) for earlier response.
//   3. KV pressure checked FIRST (highest priority) — overrides cache affinity.
//
// SCORERS (5, same as expanded):
//   precise-prefix-cache, load-aware, active-requests, running-requests, kv-utilization
//
// REGIMES:
//   Regime 1 — Memory-aware (avgKVUtil > 0.6): PRIORITY 1
//     ppc:0, la:1, ar:2, rr:1, kvu:1  — protect against KV exhaustion
//   Regime 2 — Cache-affinity (cacheSpread > 0.1 AND loadSpread < 4): PRIORITY 2
//     ppc:4, la:1, ar:1, rr:0, kvu:0  — cache when load is balanced
//   Regime 3 — Load-balance (default):
//     ppc:0, la:1, ar:2, rr:1, kvu:0  — balanced distribution
//
// USAGE: Copy this file over sim/routing.go, build BLIS, run with policy YAML:
//   routing:
//     policy: weighted
//     scorers:
//       - name: precise-prefix-cache
//         weight: 1
//       - name: load-aware
//         weight: 1
//       - name: active-requests
//         weight: 1
//       - name: running-requests
//         weight: 1
//       - name: kv-utilization
//         weight: 1

import (
	"fmt"
	"math/rand"
)

// RoutingSnapshot is a lightweight view of instance state for policy decisions.
type RoutingSnapshot struct {
	ID               string
	QueueDepth       int
	BatchSize        int
	KVUtilization    float64
	FreeKVBlocks     int64
	CacheHitRate     float64
	InFlightRequests int
	Model            string
	GPUType          string
	TPDegree         int
	CostPerHour      float64
}

// EffectiveLoad returns the total effective load on this instance.
func (s RoutingSnapshot) EffectiveLoad() int {
	return s.QueueDepth + s.BatchSize + s.InFlightRequests
}

// NewRoutingSnapshot creates a RoutingSnapshot with the given instance ID.
func NewRoutingSnapshot(id string) RoutingSnapshot {
	if id == "" {
		panic("NewRoutingSnapshot: id must not be empty")
	}
	return RoutingSnapshot{ID: id}
}

// RoutingDecision encapsulates the routing decision for a request.
type RoutingDecision struct {
	TargetInstance string
	Reason         string
	Scores         map[string]float64
	Priority       float64
}

// NewRoutingDecision creates a RoutingDecision with the given target and reason.
func NewRoutingDecision(target string, reason string) RoutingDecision {
	if target == "" {
		panic("NewRoutingDecision: target must not be empty")
	}
	return RoutingDecision{
		TargetInstance: target,
		Reason:         reason,
	}
}

// NewRoutingDecisionWithScores creates a RoutingDecision with target, reason, and per-instance scores.
func NewRoutingDecisionWithScores(target string, reason string, scores map[string]float64) RoutingDecision {
	if target == "" {
		panic("NewRoutingDecisionWithScores: target must not be empty")
	}
	return RoutingDecision{
		TargetInstance: target,
		Reason:         reason,
		Scores:         scores,
	}
}

// RoutingPolicy decides which instance should handle a request.
type RoutingPolicy interface {
	Route(req *Request, state *RouterState) RoutingDecision
}

// RoundRobin routes requests in round-robin order across instances.
type RoundRobin struct {
	counter int
}

// Route implements RoutingPolicy for RoundRobin.
func (rr *RoundRobin) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("RoundRobin.Route: empty snapshots")
	}
	target := snapshots[rr.counter%len(snapshots)]
	rr.counter++
	return NewRoutingDecision(target.ID, fmt.Sprintf("round-robin[%d]", rr.counter-1))
}

// LeastLoaded routes requests to the instance with minimum EffectiveLoad.
type LeastLoaded struct {
	rng *rand.Rand
}

// Route implements RoutingPolicy for LeastLoaded.
func (ll *LeastLoaded) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("LeastLoaded.Route: empty snapshots")
	}

	minLoad := snapshots[0].EffectiveLoad()
	for i := 1; i < len(snapshots); i++ {
		if load := snapshots[i].EffectiveLoad(); load < minLoad {
			minLoad = load
		}
	}

	var tied []int
	for i, snap := range snapshots {
		if snap.EffectiveLoad() == minLoad {
			tied = append(tied, i)
		}
	}

	idx := tied[0]
	if len(tied) > 1 && ll.rng != nil {
		idx = tied[ll.rng.Intn(len(tied))]
	}

	return NewRoutingDecision(snapshots[idx].ID, fmt.Sprintf("least-loaded (load=%d)", minLoad))
}

// observerFunc is called after each routing decision to update stateful scorer state.
type observerFunc func(req *Request, targetInstance string)

// WeightedScoring routes requests using a composable scorer pipeline.
//
// ADAPTIVE-CORRECT v3: Expanded + load-gated cache affinity.
type WeightedScoring struct {
	scorers   []scorerFunc
	weights   []float64
	observers []observerFunc
	rng       *rand.Rand
}

// Route implements RoutingPolicy for WeightedScoring.
func (ws *WeightedScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("WeightedScoring.Route: empty snapshots")
	}

	// ###########################################################################
	// ### EVOLVED ALGO LOGIC — adaptive-correct v3
	// ### Base: adaptive-expanded. Change: load-gated cache affinity + lower KV.
	// ###########################################################################

	allDimScores := make([]map[string]float64, len(ws.scorers))
	for i, scorer := range ws.scorers {
		allDimScores[i] = scorer(req, snapshots)
	}

	// --- Regime detection signals ---

	// Cache spread: max(ppc) - min(ppc).
	cacheSpread := 0.0
	if len(allDimScores) > 0 {
		minPPC, maxPPC := 1.0, 0.0
		for _, snap := range snapshots {
			s := allDimScores[0][snap.ID]
			if s < minPPC {
				minPPC = s
			}
			if s > maxPPC {
				maxPPC = s
			}
		}
		cacheSpread = maxPPC - minPPC
	}

	// Load spread: max(EffectiveLoad) - min(EffectiveLoad).
	minLoad, maxLoad := snapshots[0].EffectiveLoad(), snapshots[0].EffectiveLoad()
	for _, snap := range snapshots[1:] {
		load := snap.EffectiveLoad()
		if load < minLoad {
			minLoad = load
		}
		if load > maxLoad {
			maxLoad = load
		}
	}
	loadSpread := maxLoad - minLoad

	// Average KV utilization.
	avgKVUtil := 0.0
	for _, snap := range snapshots {
		avgKVUtil += snap.KVUtilization
	}
	avgKVUtil /= float64(len(snapshots))

	// --- Regime weights ---
	// Index: 0=ppc, 1=load-aware, 2=active-requests, 3=running-requests, 4=kv-utilization
	var regimeWeights []float64
	var regime string

	const (
		cacheSpreadThreshold = 0.1
		loadSpreadThreshold  = 4
		kvPressureThreshold  = 0.6
	)

	switch {
	case avgKVUtil > kvPressureThreshold:
		// Memory-aware: KV pressure overrides all. Disable cache, heavy kvu.
		regimeWeights = []float64{0.0, 1.0, 2.0, 1.0, 1.0}
		regime = "memory-aware"
	case cacheSpread > cacheSpreadThreshold && loadSpread < loadSpreadThreshold:
		// Cache-affinity: cache signal strong AND load is balanced.
		// Same weights as expanded's cache-affinity regime.
		regimeWeights = []float64{4.0, 1.0, 1.0, 0.0, 0.0}
		regime = "cache-affinity"
	default:
		// Load-balance: no cache signal OR load is too unbalanced.
		// Same weights as expanded's load-balance regime.
		regimeWeights = []float64{0.0, 1.0, 2.0, 1.0, 0.0}
		regime = "load-balance"
	}

	// Normalize.
	weightSum := 0.0
	for _, w := range regimeWeights {
		weightSum += w
	}
	if weightSum > 0 {
		for i := range regimeWeights {
			regimeWeights[i] /= weightSum
		}
	}

	// Composite scores.
	scores := make(map[string]float64, len(snapshots))
	for i, dimScores := range allDimScores {
		if i >= len(regimeWeights) {
			break
		}
		for _, snap := range snapshots {
			s := dimScores[snap.ID]
			if s < 0 {
				s = 0
			}
			if s > 1 {
				s = 1
			}
			scores[snap.ID] += s * regimeWeights[i]
		}
	}

	// Argmax with random tie-breaking.
	bestScore := -1.0
	for _, snap := range snapshots {
		if scores[snap.ID] > bestScore {
			bestScore = scores[snap.ID]
		}
	}

	var tied []int
	for i, snap := range snapshots {
		if scores[snap.ID] == bestScore {
			tied = append(tied, i)
		}
	}

	bestIdx := tied[0]
	if len(tied) > 1 && ws.rng != nil {
		bestIdx = tied[ws.rng.Intn(len(tied))]
	}

	for _, obs := range ws.observers {
		obs(req, snapshots[bestIdx].ID)
	}

	// ###########################################################################
	// ### END EVOLVED ALGO LOGIC
	// ###########################################################################

	return NewRoutingDecisionWithScores(
		snapshots[bestIdx].ID,
		fmt.Sprintf("adaptive-correct/%s (score=%.3f, ls=%d)", regime, bestScore, loadSpread),
		scores,
	)
}

// AlwaysBusiest routes requests to the instance with maximum load.
type AlwaysBusiest struct{}

// Route implements RoutingPolicy for AlwaysBusiest.
func (ab *AlwaysBusiest) Route(_ *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("AlwaysBusiest.Route: empty snapshots")
	}

	maxLoad := snapshots[0].EffectiveLoad()
	target := snapshots[0]

	for i := 1; i < len(snapshots); i++ {
		load := snapshots[i].EffectiveLoad()
		if load > maxLoad {
			maxLoad = load
			target = snapshots[i]
		}
	}

	return NewRoutingDecision(target.ID, fmt.Sprintf("always-busiest (load=%d)", maxLoad))
}

// NewRoutingPolicy creates a routing policy by name.
func NewRoutingPolicy(name string, scorerConfigs []ScorerConfig, blockSize int64, rng *rand.Rand) RoutingPolicy {
	return newRoutingPolicyInternal(name, scorerConfigs, blockSize, rng, nil)
}

// NewRoutingPolicyWithCache is like NewRoutingPolicy but enables cache-aware scorers.
func NewRoutingPolicyWithCache(name string, scorerConfigs []ScorerConfig, blockSize int64, rng *rand.Rand, cacheFn map[string]func([]int) int) RoutingPolicy {
	return newRoutingPolicyInternal(name, scorerConfigs, blockSize, rng, cacheQueryFn(cacheFn))
}

// newRoutingPolicyInternal creates a routing policy, shared by both public constructors.
func newRoutingPolicyInternal(name string, scorerConfigs []ScorerConfig, blockSize int64, rng *rand.Rand, cacheFn cacheQueryFn) RoutingPolicy {
	if !IsValidRoutingPolicy(name) {
		panic(fmt.Sprintf("unknown routing policy %q", name))
	}
	switch name {
	case "", "round-robin":
		return &RoundRobin{}
	case "least-loaded":
		return &LeastLoaded{rng: rng}
	case "weighted":
		if len(scorerConfigs) == 0 {
			scorerConfigs = DefaultScorerConfigs()
		}
		scorers := make([]scorerFunc, len(scorerConfigs))
		var observers []observerFunc
		for i, cfg := range scorerConfigs {
			scorer, obs := newScorerWithObserver(cfg.Name, int(blockSize), cacheFn)
			scorers[i] = scorer
			if obs != nil {
				observers = append(observers, obs)
			}
		}
		weights := normalizeScorerWeights(scorerConfigs)
		return &WeightedScoring{scorers: scorers, weights: weights, observers: observers, rng: rng}
	case "always-busiest":
		return &AlwaysBusiest{}
	default:
		panic(fmt.Sprintf("unhandled routing policy %q", name))
	}
}
