package sim

// router_adaptive_v2.go — Drop-in replacement for sim/routing.go
//
// Adaptive-v2: 3-scorer regime-detection router using precise-prefix-cache,
// load-aware, and kv-utilization scorers. Detects per-request cache spread
// and memory pressure to dynamically adjust scorer weights.
//
// USAGE: Copy this file over sim/routing.go, build BLIS, run with policy YAML:
//   routing:
//     policy: weighted
//     scorers:
//       - name: precise-prefix-cache
//         weight: 1
//       - name: load-aware
//         weight: 1
//       - name: kv-utilization
//         weight: 1
//
// The YAML weights are used for scorer instantiation only. The adaptive logic
// in Route() overrides them per-request based on regime detection.

import (
	"fmt"
	"math/rand"
)

// RoutingSnapshot is a lightweight view of instance state for policy decisions.
// Populated by CachedSnapshotProvider reading InstanceSimulator query methods,
// with InFlightRequests injected by buildRouterState() at the cluster level.
// Used by both AdmissionPolicy and RoutingPolicy.
// Timestamp is intentionally excluded: snapshot freshness is managed by
// CachedSnapshotProvider and is not a policy concern.
type RoutingSnapshot struct {
	ID               string
	QueueDepth       int
	BatchSize        int
	KVUtilization    float64
	FreeKVBlocks     int64
	CacheHitRate     float64
	InFlightRequests int    // Requests dispatched to this instance but not yet completed
	Model            string // Model served by this instance; used by buildRouterState() for per-model filtering
	GPUType          string  // GPU hardware type (e.g. "A100-80GB"); populated by buildRouterState() from instance config
	TPDegree         int     // Tensor-parallel degree; populated by buildRouterState() from instance config
	CostPerHour      float64 // Node pool cost in $/hr; populated by buildRouterState() from NodePool.CostPerHour
}

// EffectiveLoad returns the total effective load on this instance:
// QueueDepth + BatchSize + InFlightRequests.
// Used by routing policies and counterfactual scoring for consistent load calculations.
func (s RoutingSnapshot) EffectiveLoad() int {
	return s.QueueDepth + s.BatchSize + s.InFlightRequests
}

// NewRoutingSnapshot creates a RoutingSnapshot with the given instance ID.
// All numeric fields are zero-valued. Used for initial snapshot creation;
// field-by-field refresh via CachedSnapshotProvider.Snapshot() is a separate concern.
func NewRoutingSnapshot(id string) RoutingSnapshot {
	if id == "" {
		panic("NewRoutingSnapshot: id must not be empty")
	}
	return RoutingSnapshot{ID: id}
}

// RoutingDecision encapsulates the routing decision for a request.
type RoutingDecision struct {
	TargetInstance string             // Instance ID to route to (must match a snapshot ID)
	Reason         string             // Human-readable explanation
	Scores         map[string]float64 // Instance ID → composite score (nil for policies without scoring)
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

// LeastLoaded routes requests to the instance with minimum (QueueDepth + BatchSize + InFlightRequests).
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
// ADAPTIVE-V2 MODIFICATION: Instead of static weight combination, this router
// detects the current regime per-request and adjusts weights dynamically:
//
//   Regime 1 — Cache-affinity (cacheSpread > 0.1):
//     precise-prefix-cache:4, load-aware:1, kv-utilization:0
//     When prefix cache hits vary across instances, lean into cache locality.
//
//   Regime 2 — Memory-aware (avgKVUtil > 0.7):
//     precise-prefix-cache:0, load-aware:1, kv-utilization:1
//     Under memory pressure, disable prefix affinity to prevent pile-on.
//
//   Regime 3 — Load-balance (default):
//     precise-prefix-cache:0, load-aware:1, kv-utilization:0
//     When cache signals are weak and memory is fine, just balance load.
//
// Scorers are instantiated from the policy YAML (precise-prefix-cache, load-aware,
// kv-utilization). The YAML weights are ignored at runtime; regime detection
// overrides them.
type WeightedScoring struct {
	scorers   []scorerFunc
	weights   []float64 // normalized to sum to 1.0
	observers []observerFunc
	rng       *rand.Rand
}

// Route implements RoutingPolicy for WeightedScoring.
// ADAPTIVE-V2: Per-request regime detection replaces static weight combination.
func (ws *WeightedScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("WeightedScoring.Route: empty snapshots")
	}

	// ###########################################################################
	// ### EVOLVED ALGO LOGIC — adaptive-v2 (3-scorer regime detection)
	// ###
	// ### This section replaces the stock WeightedScoring static weight combination.
	// ### Everything below until the closing marker is the adaptive algorithm.
	// ### The rest of the file is identical to BLIS's sim/routing.go.
	// ###########################################################################

	// --- Regime detection ---
	// Compute all scorer dimensions first, then combine with regime-specific weights.

	allDimScores := make([]map[string]float64, len(ws.scorers))
	for i, scorer := range ws.scorers {
		allDimScores[i] = scorer(req, snapshots)
	}

	// Detect cache spread: max(ppc) - min(ppc) across instances.
	// High spread means prefix cache hits differ → cache-affinity regime.
	// Scorer index 0 = precise-prefix-cache (per YAML ordering).
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

	// Detect memory pressure: average KV utilization across instances.
	avgKVUtil := 0.0
	for _, snap := range snapshots {
		avgKVUtil += snap.KVUtilization
	}
	avgKVUtil /= float64(len(snapshots))

	// Select regime-specific weights.
	// Index mapping (must match policy YAML scorer order):
	//   0 = precise-prefix-cache
	//   1 = load-aware
	//   2 = kv-utilization
	var regimeWeights []float64
	var regime string
	switch {
	case cacheSpread > 0.1:
		// Regime 1: Cache-affinity — strong prefix signal, lean into it.
		regimeWeights = []float64{4.0, 1.0, 0.0}
		regime = "cache-affinity"
	case avgKVUtil > 0.7:
		// Regime 2: Memory-aware — disable prefix to prevent pile-on under pressure.
		regimeWeights = []float64{0.0, 1.0, 1.0}
		regime = "memory-aware"
	default:
		// Regime 3: Load-balance — weak cache signal, balance load.
		regimeWeights = []float64{0.0, 1.0, 0.0}
		regime = "load-balance"
	}

	// Normalize regime weights.
	weightSum := 0.0
	for _, w := range regimeWeights {
		weightSum += w
	}
	if weightSum > 0 {
		for i := range regimeWeights {
			regimeWeights[i] /= weightSum
		}
	}

	// Compute composite scores with regime-specific weights.
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
	// ### END EVOLVED ALGO LOGIC — adaptive-v2
	// ###########################################################################

	return NewRoutingDecisionWithScores(
		snapshots[bestIdx].ID,
		fmt.Sprintf("adaptive-v2/%s (score=%.3f)", regime, bestScore),
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
