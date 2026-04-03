// Marginal-Gain Scoring — drop-in replacement for sim/routing.go
//
// Replaces the static weighted scorer block in WeightedScoring.Route() with
// marginal-gain scoring: score_i = normCache_i - α * normLoad_i
//
// The algorithm automatically adapts to the operating regime:
// - When cache differs across instances → cache term dominates → route to cached instance
// - When cache is equal → cache term cancels → pure load balance
// - When load is very asymmetric → load cost dominates → avoid overloaded instance
//
// α controls the cache-vs-load tradeoff. Default α=1.0 (equal weight).
//
// Signals used:
//   - cacheQueryFn (precise prefix cache, ~2s stale): actual cached block count
//   - EffectiveLoad() (synchronous InFlightRequests + stale QueueDepth + BatchSize)
//
// To use: copy this file over sim/routing.go and rebuild.

package sim

import (
	"fmt"
	"math"
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

// WeightedScoring with Marginal-Gain algorithm.
type WeightedScoring struct {
	scorers   []scorerFunc
	weights   []float64
	observers []observerFunc
	rng       *rand.Rand
	cacheFn   cacheQueryFn // direct cache access for marginal-gain
}

// Route implements RoutingPolicy using Marginal-Gain Scoring.
//
// score_i = normCache_i - α * normLoad_i
//
// Where:
//   - normCache_i = min-max normalized cached block count for this request
//   - normLoad_i  = min-max normalized effective load (inverted: lower load = lower cost)
//   - α = 1.0 (cache-vs-load tradeoff parameter)
func (ws *WeightedScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("WeightedScoring.Route: empty snapshots")
	}

	// ========================================================================
	// MARGINAL-GAIN-START
	// ========================================================================

	const alpha = 1.0 // cache benefit vs load cost tradeoff

	// --- Cache benefit: how many prefix blocks are cached per instance ---
	cacheRaw := make(map[string]float64, len(snapshots))
	cacheMin, cacheMax := math.MaxFloat64, -math.MaxFloat64
	for _, snap := range snapshots {
		count := 0.0
		if ws.cacheFn != nil {
			if fn, ok := ws.cacheFn[snap.ID]; ok && fn != nil && req != nil {
				count = float64(fn(req.InputTokens))
			}
		}
		cacheRaw[snap.ID] = count
		if count < cacheMin {
			cacheMin = count
		}
		if count > cacheMax {
			cacheMax = count
		}
	}

	// --- Load cost: effective load per instance ---
	loadRaw := make(map[string]float64, len(snapshots))
	loadMin, loadMax := math.MaxFloat64, -math.MaxFloat64
	for _, snap := range snapshots {
		load := float64(snap.EffectiveLoad())
		loadRaw[snap.ID] = load
		if load < loadMin {
			loadMin = load
		}
		if load > loadMax {
			loadMax = load
		}
	}

	// --- Compute marginal-gain scores ---
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		// Min-max normalize cache (higher = better)
		var normCache float64
		if cacheMax == cacheMin {
			if cacheMax == 0 {
				normCache = 0.0 // no cache data → no cache benefit
			} else {
				normCache = 0.5 // all equal and non-zero → neutral
			}
		} else {
			normCache = (cacheRaw[snap.ID] - cacheMin) / (cacheMax - cacheMin)
		}

		// Min-max normalize load (lower = better → invert)
		var normLoad float64
		if loadMax == loadMin {
			normLoad = 0.0 // all equal → no load cost difference
		} else {
			normLoad = (loadRaw[snap.ID] - loadMin) / (loadMax - loadMin)
		}

		// Marginal gain: cache benefit minus load cost
		scores[snap.ID] = normCache - alpha*normLoad
	}

	// ========================================================================
	// MARGINAL-GAIN-END
	// ========================================================================

	// Argmax with tie-breaking
	bestScore := -math.MaxFloat64
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

	return NewRoutingDecisionWithScores(
		snapshots[bestIdx].ID,
		fmt.Sprintf("marginal-gain (score=%.3f, α=%.1f)", bestScore, alpha),
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
