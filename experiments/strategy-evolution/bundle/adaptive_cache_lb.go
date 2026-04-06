// Adaptive Cache-Aware Load Balancing — drop-in replacement for sim/routing.go
//
// Two-stage routing algorithm:
//
// Stage 1: Cache Triage
//   Query precise prefix cache for each instance. Compute cache hit ratio
//   (cached_blocks / total_request_blocks). If the best cache ratio exceeds
//   a minimum threshold AND the spread (max - min) exceeds a significance
//   threshold, filter to instances within 90% of the best cache hit ratio.
//   Otherwise, all instances are eligible (cache provides no useful signal).
//
// Stage 2: Load Balance
//   Among eligible instances, pick the one with lowest EffectiveLoad().
//   Ties broken randomly.
//
// This design addresses three key principles from prior iterations:
//   P1: Min-max normalization makes weights irrelevant at N=2 → uses raw counts
//   P2: Need magnitude-aware decisions → stage 1 uses absolute cache block counts
//   P3: Cache-aware routing creates load imbalance → cache is used as a FILTER,
//       not a score. Load balance still governs the final decision within the
//       filtered set. This prevents pile-on: even if instance A has 100% cache
//       hits, we still consider A AND other instances with similar cache, then
//       pick the least loaded among them.

package sim

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
}

func (s RoutingSnapshot) EffectiveLoad() int {
	return s.QueueDepth + s.BatchSize + s.InFlightRequests
}

func NewRoutingSnapshot(id string) RoutingSnapshot {
	if id == "" {
		panic("NewRoutingSnapshot: id must not be empty")
	}
	return RoutingSnapshot{ID: id}
}

type RoutingDecision struct {
	TargetInstance string
	Reason         string
	Scores         map[string]float64
	Priority       float64
}

func NewRoutingDecision(target string, reason string) RoutingDecision {
	if target == "" {
		panic("NewRoutingDecision: target must not be empty")
	}
	return RoutingDecision{TargetInstance: target, Reason: reason}
}

func NewRoutingDecisionWithScores(target string, reason string, scores map[string]float64) RoutingDecision {
	if target == "" {
		panic("NewRoutingDecisionWithScores: target must not be empty")
	}
	return RoutingDecision{TargetInstance: target, Reason: reason, Scores: scores}
}

type RoutingPolicy interface {
	Route(req *Request, state *RouterState) RoutingDecision
}

type RoundRobin struct{ counter int }

func (rr *RoundRobin) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("RoundRobin.Route: empty snapshots")
	}
	target := snapshots[rr.counter%len(snapshots)]
	rr.counter++
	return NewRoutingDecision(target.ID, fmt.Sprintf("round-robin[%d]", rr.counter-1))
}

type LeastLoaded struct{ rng *rand.Rand }

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

type observerFunc func(req *Request, targetInstance string)

type WeightedScoring struct {
	scorers   []scorerFunc
	weights   []float64
	observers []observerFunc
	rng       *rand.Rand
	cacheFn   cacheQueryFn // available for all routers; used by oracle for direct cache triage
}

// Route implements Adaptive Cache-Aware Load Balancing.
//
// Two stages:
//   1. Cache triage: filter to instances with significant cache advantage
//   2. Load balance: among eligible, pick lowest EffectiveLoad
func (ws *WeightedScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("WeightedScoring.Route: empty snapshots")
	}

	// ========================================================================
	// ADAPTIVE-CACHE-LB-START
	// ========================================================================

	// Stage 1: Cache Triage
	// Query cache for each instance, compute cached block count.
	type instanceInfo struct {
		idx         int
		cachedBlocks int
		load        int
	}

	candidates := make([]instanceInfo, len(snapshots))
	maxCache := 0
	minCache := int(^uint(0) >> 1) // MaxInt

	for i, snap := range snapshots {
		cached := 0
		if ws.cacheFn != nil && req != nil {
			if fn, ok := ws.cacheFn[snap.ID]; ok && fn != nil {
				cached = fn(req.InputTokens)
			}
		}
		candidates[i] = instanceInfo{
			idx:         i,
			cachedBlocks: cached,
			load:        snap.EffectiveLoad(),
		}
		if cached > maxCache {
			maxCache = cached
		}
		if cached < minCache {
			minCache = cached
		}
	}

	// Determine if cache signal is meaningful:
	// - At least one instance has >0 cached blocks
	// - The spread (max-min) represents at least 2 blocks difference
	cacheSpread := maxCache - minCache
	cacheSignificant := maxCache > 0 && cacheSpread >= 2

	// Filter eligible instances
	var eligible []instanceInfo
	if cacheSignificant {
		// Keep instances within 50% of best cache performance
		// This means: if best has 20 blocks, keep instances with >= 10 blocks
		threshold := maxCache / 2
		for _, c := range candidates {
			if c.cachedBlocks >= threshold {
				eligible = append(eligible, c)
			}
		}
	}

	// If no cache differentiation or filter too aggressive, use all instances
	if len(eligible) == 0 {
		eligible = candidates
	}

	// Stage 2: Load Balance among eligible
	minLoad := eligible[0].load
	for _, c := range eligible[1:] {
		if c.load < minLoad {
			minLoad = c.load
		}
	}

	var tied []int
	for _, c := range eligible {
		if c.load == minLoad {
			tied = append(tied, c.idx)
		}
	}

	bestIdx := tied[0]
	if len(tied) > 1 && ws.rng != nil {
		bestIdx = tied[ws.rng.Intn(len(tied))]
	}

	// Build scores for diagnostics
	scores := make(map[string]float64, len(snapshots))
	for _, c := range candidates {
		// Composite: cache benefit (normalized) + load balance
		cacheNorm := 0.0
		if maxCache > 0 {
			cacheNorm = float64(c.cachedBlocks) / float64(maxCache)
		}
		loadNorm := 1.0 / (1.0 + float64(c.load))
		scores[snapshots[c.idx].ID] = cacheNorm + loadNorm
	}

	// ========================================================================
	// ADAPTIVE-CACHE-LB-END
	// ========================================================================

	for _, obs := range ws.observers {
		obs(req, snapshots[bestIdx].ID)
	}

	reason := "adaptive-cache-lb"
	if cacheSignificant {
		reason = fmt.Sprintf("adaptive-cache-lb (cache-filtered, spread=%d, eligible=%d/%d)",
			cacheSpread, len(eligible), len(snapshots))
	}

	return NewRoutingDecisionWithScores(
		snapshots[bestIdx].ID,
		reason,
		scores,
	)
}

type AlwaysBusiest struct{}

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

func NewRoutingPolicy(name string, scorerConfigs []ScorerConfig, blockSize int64, rng *rand.Rand) RoutingPolicy {
	return newRoutingPolicyInternal(name, scorerConfigs, blockSize, rng, nil)
}

func NewRoutingPolicyWithCache(name string, scorerConfigs []ScorerConfig, blockSize int64, rng *rand.Rand, cacheFn map[string]func([]int) int) RoutingPolicy {
	return newRoutingPolicyInternal(name, scorerConfigs, blockSize, rng, cacheQueryFn(cacheFn))
}

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
		return &WeightedScoring{scorers: scorers, weights: weights, observers: observers, rng: rng, cacheFn: cacheFn}
	case "always-busiest":
		return &AlwaysBusiest{}
	default:
		panic(fmt.Sprintf("unhandled routing policy %q", name))
	}
}
