// Hash-Affinity Load Balancing — drop-in replacement for sim/routing.go
//
// Three-mode routing algorithm that adapts based on request characteristics:
//
// Mode A: Prefix Affinity (request has PrefixGroup)
//   Hash(PrefixGroup) determines a 2-instance "affinity set" from the pool.
//   Route to least-loaded within affinity set. If both affinity instances are
//   overloaded (EffectiveLoad > 2x global average), overflow to all instances.
//   This creates deterministic cache locality without relying on stale signals.
//
// Mode B: Cache Triage (request has no PrefixGroup, but cache signal exists)
//   Same as Adaptive-Cache-LB iter3: filter by cache hits, then load balance.
//
// Mode C: Pure Load Balance (no prefix, no cache signal)
//   Pick lowest EffectiveLoad() across all instances.
//
// Why this beats 3:2:2 on prefix-heavy workloads:
//   - 3:2:2 uses 2s-stale cache signal to guess which instance has prefix cached
//   - Hash affinity GUARANTEES stable prefix-to-instance mapping
//   - Each instance caches only its assigned groups (less eviction pressure)
//   - 3:2:2 scatters same-group requests across instances → redundant prefill
//
// Why this beats GLIA:
//   - GLIA has zero prefix cache awareness
//   - Hash affinity creates perfect cache locality

package sim

import (
	"fmt"
	"hash/fnv"
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
}

// hashPrefixGroup returns a deterministic hash for a prefix group name.
func hashPrefixGroup(group string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(group))
	return h.Sum32()
}

// Route implements Hash-Affinity Load Balancing.
//
// Three modes based on request characteristics:
//   Mode A: Hash affinity for prefixed requests
//   Mode B: Cache triage for non-prefixed requests with cache signal
//   Mode C: Pure load balance fallback
func (ws *WeightedScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("WeightedScoring.Route: empty snapshots")
	}

	n := len(snapshots)
	var bestIdx int
	var reason string

	// ========================================================================
	// HASH-AFFINITY-LB-START
	// ========================================================================

	if req != nil && req.PrefixGroup != "" && n >= 2 {
		// === Mode A: Prefix Affinity ===
		// Hash prefix group to determine 2-instance affinity set.
		h := hashPrefixGroup(req.PrefixGroup)
		primary := int(h) % n
		if primary < 0 {
			primary = -primary
		}
		secondary := (primary + 1) % n

		// Compute global average load for overflow detection
		totalLoad := 0
		for _, snap := range snapshots {
			totalLoad += snap.EffectiveLoad()
		}
		avgLoad := float64(totalLoad) / float64(n)

		pLoad := snapshots[primary].EffectiveLoad()
		sLoad := snapshots[secondary].EffectiveLoad()

		// Overflow check: if BOTH affinity instances are overloaded (>2x avg),
		// fall through to load balance across all instances.
		overflowThreshold := int(avgLoad*2.0) + 1
		if pLoad > overflowThreshold && sLoad > overflowThreshold {
			// Overflow: use all instances with cache triage
			bestIdx = ws.cacheThenLoadBalance(req, snapshots)
			reason = fmt.Sprintf("hash-affinity-overflow (primary=%d,secondary=%d,avgLoad=%.1f)",
				primary, secondary, avgLoad)
		} else {
			// Normal affinity: pick least loaded of {primary, secondary}
			if pLoad <= sLoad {
				bestIdx = primary
			} else {
				bestIdx = secondary
			}
			// Tie-break randomly
			if pLoad == sLoad && ws.rng != nil {
				if ws.rng.Intn(2) == 0 {
					bestIdx = secondary
				} else {
					bestIdx = primary
				}
			}
			reason = fmt.Sprintf("hash-affinity (group=%s,primary=%d,secondary=%d,chosen=%d)",
				req.PrefixGroup, primary, secondary, bestIdx)
		}
	} else {
		// === Mode B/C: Cache Triage + Load Balance ===
		bestIdx = ws.cacheThenLoadBalance(req, snapshots)
		reason = "cache-lb"
	}

	// ========================================================================
	// HASH-AFFINITY-LB-END
	// ========================================================================

	// Build diagnostic scores
	scores := make(map[string]float64, n)
	for i, snap := range snapshots {
		scores[snap.ID] = 1.0 / (1.0 + float64(snap.EffectiveLoad()))
		if i == bestIdx {
			scores[snap.ID] += 1.0 // Bonus for selected
		}
	}

	for _, obs := range ws.observers {
		obs(req, snapshots[bestIdx].ID)
	}

	return NewRoutingDecisionWithScores(snapshots[bestIdx].ID, reason, scores)
}

// cacheThenLoadBalance implements the Adaptive-Cache-LB from iter3.
// Used as fallback when no prefix group is available.
func (ws *WeightedScoring) cacheThenLoadBalance(req *Request, snapshots []RoutingSnapshot) int {
	type instanceInfo struct {
		idx          int
		cachedBlocks int
		load         int
	}

	candidates := make([]instanceInfo, len(snapshots))
	maxCache := 0
	minCache := int(^uint(0) >> 1)

	for i, snap := range snapshots {
		cached := 0
		if ws.cacheFn != nil && req != nil {
			if fn, ok := ws.cacheFn[snap.ID]; ok && fn != nil {
				cached = fn(req.InputTokens)
			}
		}
		candidates[i] = instanceInfo{idx: i, cachedBlocks: cached, load: snap.EffectiveLoad()}
		if cached > maxCache {
			maxCache = cached
		}
		if cached < minCache {
			minCache = cached
		}
	}

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

	// Load balance among eligible
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
	return bestIdx
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
