// Glia HRA (Head-Room Allocator) — complete routing.go with Glia algorithm embedded.
//
// This is a standalone copy of inference-sim/sim/routing.go with the static-weight
// scoring block in WeightedScoring.Route() replaced by the Glia HRA algorithm.
// The Glia section is clearly marked with GLIA-START / GLIA-END comments.
//
// What Glia does:
//   Instead of using the scorer pipeline with weighted combination, Glia estimates
//   the KV-cache headroom for each instance after hypothetically placing the request.
//   It projects block usage from input tokens (with a decode-to-prompt ratio),
//   checks if the instance can fit the request with a safety margin, and scores
//   based on projected utilization + queue load. Inadmissible instances get a
//   heavy penalty.
//
//   Key signals: FreeKVBlocks, KVUtilization (both stale ~5s), QueueDepth,
//   BatchSize, InFlightRequests, and request InputTokens.
//
// Origin: Adapted from sim2real/blis_router_sweet/baselines/baseline_glia.go,
//   updated to match the latest BLIS routing.go (Model field, cacheFn constructor).
//
// See hypo-oracle.md for comparison with the oracle adaptive router.

package sim

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
	Scores         map[string]float64 // Instance ID -> composite score (nil for policies without scoring)
	// Priority is a one-shot cluster-level priority hint applied before instance injection.
	// Zero (default) means defer to instance-level PriorityPolicy entirely.
	// Non-zero value sets req.Priority for initial queue ordering only -- the instance-level
	// PriorityPolicy recomputes priority each step, so this hint affects first-step scheduling
	// but does not persist. This is intentional: it allows priority to evolve over time
	// (e.g., SLOBasedPriority ages requests) while giving routing a way to influence initial placement.
	Priority float64
}

// NewRoutingDecision creates a RoutingDecision with the given target and reason.
// Scores is nil and Priority is 0.0 (defer to instance-level PriorityPolicy).
// This is the canonical constructor for policies that do not produce per-instance scores.
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
// Priority is 0.0 (defer to instance-level PriorityPolicy).
// Used by scoring-based routing policies (e.g., WeightedScoring).
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
// Implementations receive request and cluster-wide state via *RouterState.
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
// InFlightRequests prevents pile-on at high request rates where multiple routing decisions
// occur at the same timestamp before instance events process (#175).
// Ties are broken randomly when rng is non-nil; by first occurrence (lowest index) when rng is nil.
type LeastLoaded struct {
	rng *rand.Rand
}

// Route implements RoutingPolicy for LeastLoaded.
func (ll *LeastLoaded) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("LeastLoaded.Route: empty snapshots")
	}

	// Pass 1: find minimum load
	minLoad := snapshots[0].EffectiveLoad()
	for i := 1; i < len(snapshots); i++ {
		if load := snapshots[i].EffectiveLoad(); load < minLoad {
			minLoad = load
		}
	}

	// Pass 2: collect all instances tied at minimum load
	var tied []int
	for i, snap := range snapshots {
		if snap.EffectiveLoad() == minLoad {
			tied = append(tied, i)
		}
	}

	// Random tie-breaking when rng is non-nil; positional (first) when nil.
	idx := tied[0]
	if len(tied) > 1 && ll.rng != nil {
		idx = tied[ll.rng.Intn(len(tied))]
	}

	return NewRoutingDecision(snapshots[idx].ID, fmt.Sprintf("least-loaded (load=%d)", minLoad))
}

// observerFunc is called after each routing decision to update stateful scorer state.
// Used by scorers like prefix-affinity that track routing history.
type observerFunc func(req *Request, targetInstance string)

// WeightedScoring routes requests using a composable scorer pipeline.
//
// Each scorer evaluates all instances on a [0,1] scale. Scores are combined
// with configurable weights: composite = sum clamp(s_i) * w_i, then argmax.
//
// Available scorers: prefix-affinity (proportional prefix match ratio),
// precise-prefix-cache (min-max normalization of actual KV cache hits),
// no-hit-lru (cold request distribution to least-recently-used instances),
// queue-depth (min-max normalization of EffectiveLoad),
// kv-utilization (1 - KVUtilization), load-balance (1/(1 + EffectiveLoad)).
// See sim/routing_*.go for scorer implementations.
//
// Stateful scorers (prefix-affinity, no-hit-lru) register observers that update internal
// state after each routing decision. Observers are called after argmax selection.
//
// Higher scores are preferred. Ties broken randomly when rng is non-nil;
// by first occurrence (lowest index) when rng is nil.
//
// GLIA MODIFICATION: The Route() method below uses Glia HRA (Head-Room Allocator)
// instead of the scorer pipeline. See the GLIA-START / GLIA-END block.
type WeightedScoring struct {
	scorers   []scorerFunc
	weights   []float64 // normalized to sum to 1.0
	observers []observerFunc
	rng       *rand.Rand
}

// Route implements RoutingPolicy for WeightedScoring.
//
// GLIA VERSION: Uses Glia HRA (Head-Room Allocator) for KV-cache-aware routing
// instead of the default scorer pipeline. The Glia section replaces the entire
// scoring + argmax block.
func (ws *WeightedScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("WeightedScoring.Route: empty snapshots")
	}

	// ========================================================================
	// GLIA-START: Glia HRA (Head-Room Allocator) for KV-cache-aware routing.
	//
	// Replaces the default scorer pipeline. Instead of weighted scorer combination,
	// Glia estimates per-instance KV headroom after hypothetically placing this
	// request, then picks the instance with the most headroom remaining.
	//
	// Signals used:
	//   - FreeKVBlocks (stale ~5s via Prometheus)
	//   - KVUtilization (stale ~5s via Prometheus)
	//   - QueueDepth + BatchSize + InFlightRequests (mixed freshness)
	//   - req.InputTokens (request metadata)
	//
	// Parameters:
	//   - decodeToPromptRatio (0.6): anticipated decode overhead
	//   - safetyFraction (0.03): minimum free block fraction to remain admissible
	//   - blockSize (16): KV block size in tokens
	// ========================================================================

	decodeToPromptRatio := 0.6
	safetyFraction := 0.03
	blockSize := 16.0

	inputTokens := float64(len(req.InputTokens))
	reqBlocks := (inputTokens*(1.0+decodeToPromptRatio) + blockSize - 1.0) / blockSize

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
		minFreeBlocks := totalBlocks * safetyFraction
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

	// ========================================================================
	// GLIA-END
	// ========================================================================

	// Notify observers of routing decision (stateful scorers update their state).
	// Uses post-tie-breaking bestIdx so prefix-affinity records the actual target.
	for _, obs := range ws.observers {
		obs(req, snapshots[bestIdx].ID)
	}

	return NewRoutingDecisionWithScores(
		snapshots[bestIdx].ID,
		fmt.Sprintf("weighted-scoring (score=%.3f)", bestScore),
		scores,
	)
}

// AlwaysBusiest routes requests to the instance with maximum (QueueDepth + BatchSize + InFlightRequests).
// Pathological template for testing load imbalance detection.
// Ties broken by first occurrence in snapshot order (lowest index).
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
// Valid names are defined in validRoutingPolicies (bundle.go).
// Empty string defaults to round-robin.
// For weighted scoring, scorerConfigs configures the scorer pipeline.
// If scorerConfigs is nil/empty for "weighted", DefaultScorerConfigs() is used.
// Non-weighted policies ignore scorerConfigs.
// The rng parameter enables random tie-breaking for least-loaded and weighted policies;
// nil preserves positional tie-breaking. Ignored by round-robin and always-busiest.
// Panics on unrecognized names.
func NewRoutingPolicy(name string, scorerConfigs []ScorerConfig, blockSize int64, rng *rand.Rand) RoutingPolicy {
	return newRoutingPolicyInternal(name, scorerConfigs, blockSize, rng, nil)
}

// NewRoutingPolicyWithCache is like NewRoutingPolicy but enables the precise-prefix-cache
// and no-hit-lru scorers. cacheFn maps instance ID to a function returning the count of
// consecutive cached prefix blocks for given tokens; pass nil to disable those scorers
// (equivalent to calling NewRoutingPolicy).
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
