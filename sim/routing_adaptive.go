package sim

import (
	"fmt"
	"math/rand"
)

// Adaptive routing thresholds.
const (
	// adaptiveCacheSpreadThreshold is the min-max range of prefix cache scores
	// above which the router uses strong cache affinity. Below this, cache
	// scores are equalized (all cached or none cached) and ppc is irrelevant.
	adaptiveCacheSpreadThreshold = 0.1

	// adaptiveKVUtilThreshold is the average KV utilization above which the
	// router enables kv-utilization scoring (when cache is equalized). This
	// protects against preemptions when memory is tight and there's no prefix
	// affinity to exploit.
	adaptiveKVUtilThreshold = 0.7
)

// AdaptiveScoring is a zero-config routing policy that reads cache and memory
// state at each routing decision and dynamically selects the scoring regime.
//
// Three regimes derived from failure-mode experiments:
//
//  1. Cache-affinity (spread > 0.1): Some instances cached the prefix, others
//     didn't. Strong ppc weight routes to the cached instance, avoiding prefill
//     recomputation. kvu is excluded because it fights ppc by routing toward
//     uncached (emptier) instances.
//     Weights: ppc:4, qd:1
//
//  2. Memory-aware (spread <= 0.1, avgKVUtil > 0.5): Cache is equalized (no
//     affinity to exploit) but memory is tight. kvu prevents routing to
//     instances near KV exhaustion, avoiding preemptions.
//     Weights: qd:1, kvu:1
//
//  3. Load-balance (spread <= 0.1, avgKVUtil <= 0.5): Cache is equalized and
//     memory is spacious. Pure load balancing spreads requests optimally.
//     Weights: qd:1
//
// This eliminates all user-configurable weights. The policy adapts to the
// workload and system state at each routing decision.
type AdaptiveScoring struct {
	ppcScorer scorerFunc // precise-prefix-cache
	qdScorer  scorerFunc // queue-depth
	kvuScorer scorerFunc // kv-utilization
	observers []observerFunc
	rng       *rand.Rand
}

// newAdaptiveScoring creates an AdaptiveScoring policy with ppc, qd, and kvu
// scorers. The policy decides at each routing call which scorers to activate.
func newAdaptiveScoring(blockSize int, rng *rand.Rand, cacheFn cacheQueryFn) *AdaptiveScoring {
	ppc, ppcObs := newScorerWithObserver("precise-prefix-cache", blockSize, cacheFn)
	qd, _ := newScorerWithObserver("queue-depth", blockSize, cacheFn)
	kvu, _ := newScorerWithObserver("kv-utilization", blockSize, cacheFn)

	var observers []observerFunc
	if ppcObs != nil {
		observers = append(observers, ppcObs)
	}

	return &AdaptiveScoring{
		ppcScorer: ppc,
		qdScorer:  qd,
		kvuScorer: kvu,
		observers: observers,
		rng:       rng,
	}
}

// Route implements RoutingPolicy. It evaluates scorers, detects the current
// regime from cache spread and KV utilization, and applies the appropriate
// weights.
func (a *AdaptiveScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("AdaptiveScoring.Route: empty snapshots")
	}

	// Step 1: evaluate all three scorers.
	ppcScores := a.ppcScorer(req, snapshots)
	qdScores := a.qdScorer(req, snapshots)
	kvuScores := a.kvuScorer(req, snapshots)

	// Step 2: measure cache score differentiation.
	minPPC := 2.0
	maxPPC := -1.0
	for _, snap := range snapshots {
		s := ppcScores[snap.ID]
		if s < minPPC {
			minPPC = s
		}
		if s > maxPPC {
			maxPPC = s
		}
	}
	cacheSpread := maxPPC - minPPC

	// Step 3: measure average KV utilization.
	var totalKVUtil float64
	for _, snap := range snapshots {
		totalKVUtil += snap.KVUtilization
	}
	avgKVUtil := totalKVUtil / float64(len(snapshots))

	// Step 4: select regime and weights.
	var ppcW, qdW, kvuW float64
	var regime string

	if cacheSpread > adaptiveCacheSpreadThreshold {
		// Regime 1: Cache-affinity — prefix cached unevenly across instances.
		// Strong ppc routes to cached instance. kvu excluded (fights ppc).
		ppcW = 4.0 / 5.0
		qdW = 1.0 / 5.0
		kvuW = 0.0
		regime = "cache-affinity"
	} else if avgKVUtil > adaptiveKVUtilThreshold {
		// Regime 2: Memory-aware — no cache affinity but memory is tight.
		// kvu prevents routing to instances near KV exhaustion.
		ppcW = 0.0
		qdW = 1.0 / 2.0
		kvuW = 1.0 / 2.0
		regime = "memory-aware"
	} else {
		// Regime 3: Load-balance — no cache affinity, memory is spacious.
		// Pure load balancing.
		ppcW = 0.0
		qdW = 1.0
		kvuW = 0.0
		regime = "load-balance"
	}

	// Step 5: weighted combination.
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		ppc := clampScore(ppcScores[snap.ID])
		qd := clampScore(qdScores[snap.ID])
		kvu := clampScore(kvuScores[snap.ID])
		scores[snap.ID] = ppc*ppcW + qd*qdW + kvu*kvuW
	}

	// Step 6: argmax with tie-breaking.
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
	if len(tied) > 1 && a.rng != nil {
		bestIdx = tied[a.rng.Intn(len(tied))]
	}

	// Step 7: notify observers.
	for _, obs := range a.observers {
		obs(req, snapshots[bestIdx].ID)
	}

	return NewRoutingDecisionWithScores(
		snapshots[bestIdx].ID,
		fmt.Sprintf("adaptive-%s (score=%.3f, spread=%.2f, kvUtil=%.2f)", regime, bestScore, cacheSpread, avgKVUtil),
		scores,
	)
}

// clampScore clamps a scorer output to [0,1].
func clampScore(s float64) float64 {
	if s < 0 {
		return 0
	}
	if s > 1 {
		return 1
	}
	return s
}
