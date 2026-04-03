// Priority-Aware Cache-Adaptive Load Balancing — drop-in replacement for sim/routing.go
//
// Combines three mechanisms:
//   1. Adaptive Cache Triage (from iter3): filter by cache, then load balance
//   2. Size-Aware Priority: set RoutingDecision.Priority = 1/(1+inputTokens),
//      enabling priority-fcfs scheduler to process small requests first (SJF-like)
//   3. Token-Weighted Load Estimation: track cumulative tokens per instance to
//      break InFlightRequests ties using actual token load
//
// Why this beats 3:2:2 on size-variant workloads:
//   - 3:2:2 doesn't set priority → with priority-fcfs, all requests equal → FCFS behavior
//   - Our algorithm sets priority → small requests jump queue → dramatically lower TTFT
//   - Token-weighted tiebreaking avoids routing small requests to token-heavy instances
//
// Why this beats GLIA on prefix workloads (same as iter3):
//   - GLIA has zero prefix cache awareness
//   - Cache triage creates locality without load imbalance

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
	cacheFn   cacheQueryFn

	// Token-weighted load tracking (synchronous, maintained in Route)
	cumTokens map[string]int64 // cumulative input tokens routed to each instance
	cumCount  map[string]int64 // cumulative request count routed to each instance
}

// Route implements Priority-Aware Cache-Adaptive Load Balancing.
func (ws *WeightedScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("WeightedScoring.Route: empty snapshots")
	}

	// Initialize tracking maps on first call
	if ws.cumTokens == nil {
		ws.cumTokens = make(map[string]int64, len(snapshots))
		ws.cumCount = make(map[string]int64, len(snapshots))
	}

	n := len(snapshots)

	// ========================================================================
	// PRIORITY-CACHE-LB-START
	// ========================================================================

	// Stage 1: Cache Triage (from iter3)
	type instanceInfo struct {
		idx          int
		cachedBlocks int
		load         int
		tokenLoad    float64 // estimated token-weighted in-flight load
	}

	candidates := make([]instanceInfo, n)
	maxCache := 0
	minCache := int(^uint(0) >> 1)

	for i, snap := range snapshots {
		cached := 0
		if ws.cacheFn != nil && req != nil {
			if fn, ok := ws.cacheFn[snap.ID]; ok && fn != nil {
				cached = fn(req.InputTokens)
			}
		}

		// Estimate token-weighted in-flight load
		tokenLoad := float64(snap.EffectiveLoad()) // default: use request count
		if ws.cumCount[snap.ID] > 0 && snap.InFlightRequests > 0 {
			avgTokensPerReq := float64(ws.cumTokens[snap.ID]) / float64(ws.cumCount[snap.ID])
			tokenLoad = avgTokensPerReq * float64(snap.InFlightRequests)
		}

		candidates[i] = instanceInfo{
			idx:          i,
			cachedBlocks: cached,
			load:         snap.EffectiveLoad(),
			tokenLoad:    tokenLoad,
		}
		if cached > maxCache {
			maxCache = cached
		}
		if cached < minCache {
			minCache = cached
		}
	}

	// Cache filter: only apply when significant differentiation
	cacheSpread := maxCache - minCache
	cacheSignificant := maxCache > 0 && cacheSpread >= 2

	var eligible []instanceInfo
	if cacheSignificant {
		threshold := maxCache / 2
		for _, c := range candidates {
			if c.cachedBlocks >= threshold {
				eligible = append(eligible, c)
			}
		}
	}
	if len(eligible) == 0 {
		eligible = candidates
	}

	// Stage 2: Load Balance among eligible
	// Primary sort: lowest EffectiveLoad
	// Tiebreaker: lowest token-weighted load
	minLoad := eligible[0].load
	for _, c := range eligible[1:] {
		if c.load < minLoad {
			minLoad = c.load
		}
	}

	var tied []instanceInfo
	for _, c := range eligible {
		if c.load == minLoad {
			tied = append(tied, c)
		}
	}

	bestIdx := tied[0].idx
	if len(tied) > 1 {
		// Break ties using token-weighted load
		minTokenLoad := tied[0].tokenLoad
		for _, c := range tied[1:] {
			if c.tokenLoad < minTokenLoad {
				minTokenLoad = c.tokenLoad
			}
		}
		var tokenTied []int
		for _, c := range tied {
			if c.tokenLoad <= minTokenLoad*1.05 { // 5% tolerance
				tokenTied = append(tokenTied, c.idx)
			}
		}
		if len(tokenTied) == 1 {
			bestIdx = tokenTied[0]
		} else if len(tokenTied) > 1 && ws.rng != nil {
			bestIdx = tokenTied[ws.rng.Intn(len(tokenTied))]
		}
	}

	// Stage 3: Size-Aware Priority
	priority := 0.0
	if req != nil && len(req.InputTokens) > 0 {
		priority = 1.0 / (1.0 + float64(len(req.InputTokens)))
	}

	// Update token-weighted tracking
	if req != nil {
		targetID := snapshots[bestIdx].ID
		ws.cumTokens[targetID] += int64(len(req.InputTokens))
		ws.cumCount[targetID]++
	}

	// ========================================================================
	// PRIORITY-CACHE-LB-END
	// ========================================================================

	// Build diagnostic scores
	scores := make(map[string]float64, n)
	for _, c := range candidates {
		cacheNorm := 0.0
		if maxCache > 0 {
			cacheNorm = float64(c.cachedBlocks) / float64(maxCache)
		}
		loadNorm := 1.0 / (1.0 + float64(c.load))
		scores[snapshots[c.idx].ID] = cacheNorm + loadNorm
	}

	for _, obs := range ws.observers {
		obs(req, snapshots[bestIdx].ID)
	}

	reason := "priority-cache-lb"
	if cacheSignificant {
		reason = fmt.Sprintf("priority-cache-lb (cache-filtered, spread=%d, eligible=%d/%d)",
			cacheSpread, len(eligible), n)
	}

	decision := NewRoutingDecisionWithScores(snapshots[bestIdx].ID, reason, scores)
	decision.Priority = priority
	return decision
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
