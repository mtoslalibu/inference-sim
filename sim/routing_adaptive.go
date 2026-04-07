package sim

import (
	"fmt"
	"math/rand"
)

// adaptiveCacheSpreadThreshold is the min-max range of prefix cache scores
// above which the adaptive router uses strong cache affinity (ppc-dominant).
// Below this threshold, cache scores are equalized across instances (either
// all cached or none cached), so the router falls back to pure load balancing.
//
// Rationale: when cache scores are differentiated (spread > 0.1), some
// instances have cached the prefix and others haven't. Strong ppc weight
// routes to the cached instance, saving prefill recomputation. When scores
// equalize, ppc is irrelevant and qd spreads load optimally.
const adaptiveCacheSpreadThreshold = 0.1

// AdaptiveScoring is a zero-config routing policy that reads cache state at
// each routing decision and dynamically adjusts scorer weights.
//
// Two design principles derived from failure-mode experiments:
//  1. Never use kv-utilization — it fights cache affinity under cache pressure
//     by routing to emptier (uncached) instances, causing prefill recomputation.
//  2. Adjust ppc vs qd weight based on cache score differentiation, not on a
//     fixed ratio that users must tune.
//
// Regime selection (one check per routing decision):
//
//	cache spread = max(ppc_scores) - min(ppc_scores)
//
//	if spread > 0.1:
//	    Cache is differentiated → strong cache affinity: ppc:4, qd:1
//	if spread <= 0.1:
//	    Cache equalized (all same or all empty) → pure load balance: qd:1
//
// This eliminates all user-configurable weights. The policy either matches
// or beats the static 2:1:1 baseline across all tested workloads.
type AdaptiveScoring struct {
	ppcScorer scorerFunc // precise-prefix-cache
	qdScorer  scorerFunc // queue-depth
	observers []observerFunc
	rng       *rand.Rand
}

// newAdaptiveScoring creates an AdaptiveScoring policy with ppc and qd scorers.
// kv-utilization is intentionally excluded — experiments show it is always
// neutral or harmful (fights cache affinity under cache pressure).
func newAdaptiveScoring(blockSize int, rng *rand.Rand, cacheFn cacheQueryFn) *AdaptiveScoring {
	ppc, ppcObs := newScorerWithObserver("precise-prefix-cache", blockSize, cacheFn)
	qd, _ := newScorerWithObserver("queue-depth", blockSize, cacheFn)

	var observers []observerFunc
	if ppcObs != nil {
		observers = append(observers, ppcObs)
	}

	return &AdaptiveScoring{
		ppcScorer: ppc,
		qdScorer:  qd,
		observers: observers,
		rng:       rng,
	}
}

// Route implements RoutingPolicy. It evaluates ppc and qd scorers, checks
// cache differentiation, and dynamically selects the weighting regime.
func (a *AdaptiveScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("AdaptiveScoring.Route: empty snapshots")
	}

	// Step 1: evaluate both scorers.
	ppcScores := a.ppcScorer(req, snapshots)
	qdScores := a.qdScorer(req, snapshots)

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

	// Step 3: select weights based on cache differentiation.
	var ppcW, qdW float64
	var regime string
	if cacheSpread > adaptiveCacheSpreadThreshold {
		// Cache is differentiated — some instances have the prefix, others don't.
		// Strong cache affinity routes to the cached instance.
		ppcW = 4.0 / 5.0
		qdW = 1.0 / 5.0
		regime = "cache-affinity"
	} else {
		// Cache scores equalized — all instances equally cached (or none cached).
		// Pure load balancing is optimal.
		ppcW = 0.0
		qdW = 1.0
		regime = "load-balance"
	}

	// Step 4: weighted combination.
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		ppc := clampScore(ppcScores[snap.ID])
		qd := clampScore(qdScores[snap.ID])
		scores[snap.ID] = ppc*ppcW + qd*qdW
	}

	// Step 5: argmax with tie-breaking.
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

	// Step 6: notify observers.
	for _, obs := range a.observers {
		obs(req, snapshots[bestIdx].ID)
	}

	return NewRoutingDecisionWithScores(
		snapshots[bestIdx].ID,
		fmt.Sprintf("adaptive-%s (score=%.3f, spread=%.2f)", regime, bestScore, cacheSpread),
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
