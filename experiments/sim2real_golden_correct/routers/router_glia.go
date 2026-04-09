package sim

// router_glia.go — Drop-in replacement for sim/routing.go
//
// Glia HRA (Head-Room Allocator): KV-cache headroom projection router.
// Instead of using the scorer pipeline, this router estimates per-instance
// KV-cache headroom after hypothetically placing the request. It projects
// block usage from input tokens (with a decode-to-prompt ratio), checks
// admissibility with a safety margin, and scores based on projected
// utilization + queue load.
//
// USAGE: Copy this file over sim/routing.go, build BLIS, run with policy YAML:
//   routing:
//     policy: weighted
//     scorers:
//       - name: queue-depth
//         weight: 1
//
// The YAML scorer is a dummy — needed to trigger the "weighted" path.
// Route() ignores the scorer pipeline entirely.
//
// Origin: Adapted from sim2real/blis_router_sweet/baselines/baseline_glia.go.

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
	CostPerHour            float64 // Node pool cost in $/hr; populated by buildRouterState() from NodePool.CostPerHour
	TotalKvCapacityTokens  int64   // Total KV cache capacity in tokens
	KvTokensInUse          int64   // KV cache tokens currently in use
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

// Glia HRA parameters.
const (
	gliaDecodeToPromptRatio = 0.6
	gliaSafetyFraction      = 0.03
	gliaBlockSize           = 16.0
)

// WeightedScoring routes requests using a composable scorer pipeline.
//
// GLIA HRA MODIFICATION: This router completely bypasses the scorer pipeline.
// Instead, it estimates KV-cache headroom for each instance:
//
//   1. Project block usage: reqBlocks = ceil(inputTokens * (1 + decodeToPromptRatio) / blockSize)
//   2. Estimate total blocks from FreeKVBlocks and KVUtilization
//   3. Check admissibility: freeAfter >= totalBlocks * safetyFraction
//   4. Score: -projectedUsage/totalBlocks - 0.001*queueLoad (admissible)
//            -10.0 - projectedUsage/totalBlocks - 0.001*queueLoad (inadmissible)
//   5. Select instance with highest score (least projected utilization)
//
// Signals used: FreeKVBlocks, KVUtilization, QueueDepth, BatchSize,
// InFlightRequests, req.InputTokens.
type WeightedScoring struct {
	scorers   []scorerFunc
	weights   []float64 // normalized to sum to 1.0
	observers []observerFunc
	rng       *rand.Rand
}

// Route implements RoutingPolicy for WeightedScoring.
// GLIA HRA: Bypasses scorer pipeline, projects KV headroom directly.
func (ws *WeightedScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("WeightedScoring.Route: empty snapshots")
	}

	// ###########################################################################
	// ### EVOLVED ALGO LOGIC — Glia HRA (KV-cache headroom projection)
	// ###
	// ### This section replaces the stock WeightedScoring static weight combination.
	// ### Instead of using the scorer pipeline, it projects KV-cache headroom
	// ### per instance and scores based on projected utilization + queue load.
	// ### Everything below until the closing marker is the Glia algorithm.
	// ### The rest of the file is identical to BLIS's sim/routing.go.
	// ###########################################################################

	inputTokens := float64(len(req.InputTokens))
	reqBlocks := (inputTokens*(1.0+gliaDecodeToPromptRatio) + gliaBlockSize - 1.0) / gliaBlockSize

	scores := make(map[string]float64, len(snapshots))
	bestIdx := 0
	bestScore := -1e18

	for i, snap := range snapshots {
		freeBlocks := float64(snap.FreeKVBlocks)
		kvUtil := snap.KVUtilization

		// Estimate total blocks from utilization ratio.
		var totalBlocks float64
		if kvUtil > 0.001 && kvUtil < 0.999 {
			totalBlocks = freeBlocks / (1.0 - kvUtil)
		} else if kvUtil <= 0.001 {
			totalBlocks = freeBlocks
		} else {
			totalBlocks = freeBlocks * 1000.0
		}
		if totalBlocks < 1.0 {
			totalBlocks = 1.0
		}

		// Project usage after placing this request.
		minFreeBlocks := totalBlocks * gliaSafetyFraction
		allocatedBlocks := totalBlocks - freeBlocks
		projectedUsage := allocatedBlocks + reqBlocks
		freeAfter := totalBlocks - projectedUsage
		admissible := freeAfter >= minFreeBlocks
		queueLoad := float64(snap.QueueDepth + snap.BatchSize + snap.InFlightRequests)

		// Score: prefer low projected utilization; penalize inadmissible instances.
		var score float64
		if admissible {
			score = -projectedUsage/totalBlocks - 0.001*queueLoad
		} else {
			score = -10.0 - projectedUsage/totalBlocks - 0.001*queueLoad
		}

		scores[snap.ID] = score
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	// Notify observers (unlikely to be meaningful for Glia, but maintain contract).
	for _, obs := range ws.observers {
		obs(req, snapshots[bestIdx].ID)
	}

	// ###########################################################################
	// ### END EVOLVED ALGO LOGIC — Glia HRA
	// ###########################################################################

	return NewRoutingDecisionWithScores(
		snapshots[bestIdx].ID,
		fmt.Sprintf("glia-hra (score=%.3f)", bestScore),
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
