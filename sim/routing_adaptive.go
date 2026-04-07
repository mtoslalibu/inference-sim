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

// AdaptiveScoringV2 is a zero-config routing policy that uses llm-d/GIE-parity
// scorers only. It replaces queue-depth with load-aware (threshold-capped linear
// scorer matching llm-d's load-aware-scorer #958).
//
// Motivation: After PR #960, queue-depth reads QueueDepth only (no InFlightRequests).
// Under stale snapshots (5s refresh), min-max normalized QueueDepth produces
// identical scores for all instances within a refresh window, causing pile-on.
// The load-aware scorer's threshold cap (128) provides hard overload protection:
// once an instance's queue exceeds the threshold, its score drops to 0 regardless
// of what other instances report.
//
// Three regimes (same detection as v1, different load scorer):
//
//  1. Cache-affinity (spread > 0.1): ppc:4, la:1
//  2. Memory-aware (spread <= 0.1, avgKVUtil > 0.7): la:1, kvu:1
//  3. Load-balance (spread <= 0.1, avgKVUtil <= 0.7): la:1
type AdaptiveScoringV2 struct {
	ppcScorer scorerFunc // precise-prefix-cache
	laScorer  scorerFunc // load-aware (llm-d parity)
	kvuScorer scorerFunc // kv-utilization
	observers []observerFunc
	rng       *rand.Rand
}

// newAdaptiveScoringV2 creates an AdaptiveScoringV2 policy with ppc, load-aware,
// and kvu scorers.
func newAdaptiveScoringV2(blockSize int, rng *rand.Rand, cacheFn cacheQueryFn) *AdaptiveScoringV2 {
	ppc, ppcObs := newScorerWithObserver("precise-prefix-cache", blockSize, cacheFn)
	la, _ := newScorerWithObserver("load-aware", blockSize, cacheFn)
	kvu, _ := newScorerWithObserver("kv-utilization", blockSize, cacheFn)

	var observers []observerFunc
	if ppcObs != nil {
		observers = append(observers, ppcObs)
	}

	return &AdaptiveScoringV2{
		ppcScorer: ppc,
		laScorer:  la,
		kvuScorer: kvu,
		observers: observers,
		rng:       rng,
	}
}

// Route implements RoutingPolicy for AdaptiveScoringV2.
func (a *AdaptiveScoringV2) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("AdaptiveScoringV2.Route: empty snapshots")
	}

	// Step 1: evaluate all three scorers.
	ppcScores := a.ppcScorer(req, snapshots)
	laScores := a.laScorer(req, snapshots)
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
	var ppcW, laW, kvuW float64
	var regime string

	if cacheSpread > adaptiveCacheSpreadThreshold {
		// Regime 1: Cache-affinity — prefix cached unevenly across instances.
		ppcW = 4.0 / 5.0
		laW = 1.0 / 5.0
		kvuW = 0.0
		regime = "cache-affinity"
	} else if avgKVUtil > adaptiveKVUtilThreshold {
		// Regime 2: Memory-aware — no cache affinity but memory is tight.
		ppcW = 0.0
		laW = 1.0 / 2.0
		kvuW = 1.0 / 2.0
		regime = "memory-aware"
	} else {
		// Regime 3: Load-balance — no cache affinity, memory is spacious.
		ppcW = 0.0
		laW = 1.0
		kvuW = 0.0
		regime = "load-balance"
	}

	// Step 5: weighted combination.
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		ppc := clampScore(ppcScores[snap.ID])
		la := clampScore(laScores[snap.ID])
		kvu := clampScore(kvuScores[snap.ID])
		scores[snap.ID] = ppc*ppcW + la*laW + kvu*kvuW
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
		fmt.Sprintf("adaptive-v2-%s (score=%.3f, spread=%.2f, kvUtil=%.2f)", regime, bestScore, cacheSpread, avgKVUtil),
		scores,
	)
}

// AdaptiveScoringV3 is a zero-config routing policy that uses the synchronous
// InFlightRequests signal (GIE's InFlightLoad.Requests) as the load scorer.
//
// Unlike queue-depth (periodic, stale under 5s refresh) and load-aware (periodic
// with threshold cap), InFlightRequests is updated synchronously at the gateway
// on every dispatch and completion — zero staleness. This makes it the only load
// signal that differentiates instances within a snapshot refresh window.
//
// GIE parity: InFlightLoad.Requests is already populated per endpoint by GIE's
// data layer (concurrency.InFlightLoadKey). No new signal plumbing needed —
// just a scorer that reads it.
//
// Three regimes (same detection as v1/v2, different load scorer):
//
//  1. Cache-affinity (spread > 0.1): ppc:4, ifr:1
//  2. Memory-aware (spread <= 0.1, avgKVUtil > 0.7): ifr:1, kvu:1
//  3. Load-balance (spread <= 0.1, avgKVUtil <= 0.7): ifr:1
type AdaptiveScoringV3 struct {
	ppcScorer scorerFunc // precise-prefix-cache
	ifrScorer scorerFunc // inflight-requests (synchronous)
	kvuScorer scorerFunc // kv-utilization
	observers []observerFunc
	rng       *rand.Rand
}

// newAdaptiveScoringV3 creates an AdaptiveScoringV3 policy with ppc, inflight-requests,
// and kvu scorers.
func newAdaptiveScoringV3(blockSize int, rng *rand.Rand, cacheFn cacheQueryFn) *AdaptiveScoringV3 {
	ppc, ppcObs := newScorerWithObserver("precise-prefix-cache", blockSize, cacheFn)
	ifr, _ := newScorerWithObserver("inflight-requests", blockSize, cacheFn)
	kvu, _ := newScorerWithObserver("kv-utilization", blockSize, cacheFn)

	var observers []observerFunc
	if ppcObs != nil {
		observers = append(observers, ppcObs)
	}

	return &AdaptiveScoringV3{
		ppcScorer: ppc,
		ifrScorer: ifr,
		kvuScorer: kvu,
		observers: observers,
		rng:       rng,
	}
}

// Route implements RoutingPolicy for AdaptiveScoringV3.
func (a *AdaptiveScoringV3) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("AdaptiveScoringV3.Route: empty snapshots")
	}

	// Step 1: evaluate all three scorers.
	ppcScores := a.ppcScorer(req, snapshots)
	ifrScores := a.ifrScorer(req, snapshots)
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
	var ppcW, ifrW, kvuW float64
	var regime string

	if cacheSpread > adaptiveCacheSpreadThreshold {
		// Regime 1: Cache-affinity — prefix cached unevenly across instances.
		ppcW = 4.0 / 5.0
		ifrW = 1.0 / 5.0
		kvuW = 0.0
		regime = "cache-affinity"
	} else if avgKVUtil > adaptiveKVUtilThreshold {
		// Regime 2: Memory-aware — no cache affinity but memory is tight.
		ppcW = 0.0
		ifrW = 1.0 / 2.0
		kvuW = 1.0 / 2.0
		regime = "memory-aware"
	} else {
		// Regime 3: Load-balance — no cache affinity, memory is spacious.
		ppcW = 0.0
		ifrW = 1.0
		kvuW = 0.0
		regime = "load-balance"
	}

	// Step 5: weighted combination.
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		ppc := clampScore(ppcScores[snap.ID])
		ifr := clampScore(ifrScores[snap.ID])
		kvu := clampScore(kvuScores[snap.ID])
		scores[snap.ID] = ppc*ppcW + ifr*ifrW + kvu*kvuW
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
		fmt.Sprintf("adaptive-v3-%s (score=%.3f, spread=%.2f, kvUtil=%.2f)", regime, bestScore, cacheSpread, avgKVUtil),
		scores,
	)
}

// AdaptiveScoringGolden is the "golden" adaptive router that combines the best
// insights from v2 (regime detection + load-aware threshold cap) with additional
// GIE-parity signals: active-requests (synchronous, llm-d #957) and
// running-requests (batch size, GIE #956).
//
// Key evolution over v2:
// - Adds active-requests as a synchronous load signal. v3's experiment proved
//   synchronous InFlightRequests gives massive TTFT wins (82-97% over v2 on FM-1).
//   But v3's min-max normalization regressed on FM-5. active-requests uses llm-d's
//   (max-count)/max formula which is gentler — idle instances score 1.0 regardless.
// - Adds running-requests (BatchSize) as a secondary periodic signal. Instances
//   with large running batches will generate more tokens and fill KV faster.
//
// Five scorers (all GIE/llm-d parity):
//   ppc:  precise-prefix-cache (periodic, cache-signal-delay)
//   la:   load-aware (periodic, threshold-capped QueueDepth)
//   ar:   active-requests (synchronous InFlightRequests, llm-d formula)
//   rr:   running-requests (periodic BatchSize, min-max)
//   kvu:  kv-utilization (periodic)
//
// Three regimes:
//  1. Cache-affinity (spread > 0.1): ppc:4, ar:1, la:1
//     — Strong prefix routing with both synchronous and periodic load guards
//  2. Memory-aware (spread <= 0.1, avgKVUtil > 0.7): ar:2, la:1, rr:1, kvu:1
//     — Multi-signal load balancing with KV protection
//  3. Load-balance (spread <= 0.1, avgKVUtil <= 0.7): ar:2, la:1, rr:1
//     — Spread load using all available signals
type AdaptiveScoringGolden struct {
	ppcScorer scorerFunc // precise-prefix-cache
	laScorer  scorerFunc // load-aware (threshold-capped)
	arScorer  scorerFunc // active-requests (synchronous, llm-d)
	rrScorer  scorerFunc // running-requests (batch size, GIE)
	kvuScorer scorerFunc // kv-utilization
	observers []observerFunc
	rng       *rand.Rand
}

// newAdaptiveScoringGolden creates an AdaptiveScoringGolden policy.
func newAdaptiveScoringGolden(blockSize int, rng *rand.Rand, cacheFn cacheQueryFn) *AdaptiveScoringGolden {
	ppc, ppcObs := newScorerWithObserver("precise-prefix-cache", blockSize, cacheFn)
	la, _ := newScorerWithObserver("load-aware", blockSize, cacheFn)
	ar, _ := newScorerWithObserver("active-requests", blockSize, cacheFn)
	rr, _ := newScorerWithObserver("running-requests", blockSize, cacheFn)
	kvu, _ := newScorerWithObserver("kv-utilization", blockSize, cacheFn)

	var observers []observerFunc
	if ppcObs != nil {
		observers = append(observers, ppcObs)
	}

	return &AdaptiveScoringGolden{
		ppcScorer: ppc,
		laScorer:  la,
		arScorer:  ar,
		rrScorer:  rr,
		kvuScorer: kvu,
		observers: observers,
		rng:       rng,
	}
}

// Route implements RoutingPolicy for AdaptiveScoringGolden.
func (a *AdaptiveScoringGolden) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("AdaptiveScoringGolden.Route: empty snapshots")
	}

	// Step 1: evaluate all five scorers.
	ppcScores := a.ppcScorer(req, snapshots)
	laScores := a.laScorer(req, snapshots)
	arScores := a.arScorer(req, snapshots)
	rrScores := a.rrScorer(req, snapshots)
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
	var ppcW, laW, arW, rrW, kvuW float64
	var regime string

	if cacheSpread > adaptiveCacheSpreadThreshold {
		// Regime 1: Cache-affinity — prefix cached unevenly.
		// Strong ppc with dual load guards (synchronous ar + periodic la).
		ppcW = 4.0 / 6.0
		arW = 1.0 / 6.0
		laW = 1.0 / 6.0
		rrW = 0.0
		kvuW = 0.0
		regime = "cache-affinity"
	} else if avgKVUtil > adaptiveKVUtilThreshold {
		// Regime 2: Memory-aware — no cache affinity, KV tight.
		// Multi-signal load balancing with KV protection.
		ppcW = 0.0
		arW = 2.0 / 5.0
		laW = 1.0 / 5.0
		rrW = 1.0 / 5.0
		kvuW = 1.0 / 5.0
		regime = "memory-aware"
	} else {
		// Regime 3: Load-balance — no cache affinity, memory spacious.
		// Spread load using synchronous + periodic signals.
		ppcW = 0.0
		arW = 2.0 / 4.0
		laW = 1.0 / 4.0
		rrW = 1.0 / 4.0
		kvuW = 0.0
		regime = "load-balance"
	}

	// Step 5: weighted combination.
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		ppc := clampScore(ppcScores[snap.ID])
		la := clampScore(laScores[snap.ID])
		ar := clampScore(arScores[snap.ID])
		rr := clampScore(rrScores[snap.ID])
		kvu := clampScore(kvuScores[snap.ID])
		scores[snap.ID] = ppc*ppcW + la*laW + ar*arW + rr*rrW + kvu*kvuW
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
		fmt.Sprintf("adaptive-golden-%s (score=%.3f, spread=%.2f, kvUtil=%.2f)", regime, bestScore, cacheSpread, avgKVUtil),
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
