package sim

import (
	"fmt"
	"math"
	"math/rand"
)

// AdaptiveScoring routes requests by dynamically blending prefix-cache affinity
// with load-balancing signals based on load imbalance across instances.
//
// Core mechanism: load spread (coefficient of variation of effective load) drives
// the blend. When instances are unevenly loaded, shift toward load balancing.
// When load is uniform, keep prefix affinity (cache hits save recomputation).
//
// Three mechanisms:
//
//  1. Spread-driven pressure: load CV across instances drives the prefix↔load blend.
//     High CV (imbalance) → load dominates. Low CV (uniform) → prefix dominates.
//  2. Prefix information gate: when prefix scores have zero variance (e.g., cold requests),
//     the blend drops to 100% load score regardless of pressure.
//  3. Base pressure floor: even with balanced load, extreme mean KV utilization or queue
//     depth contributes a minimum pressure to prevent cache-chasing into saturated clusters.
//
// Discovered through 8 iterations of strategy evolution against 3 probe workloads.
// See experiments/sim2real_arouter_evolution/strategy-evolution/findings.md for details.
//
// Sim2Real: deploys as a single GAIE scorer plugin reading the same Endpoint signals.
// Transfer path: one Go file + one plugin.Register() line in llm-d.
type AdaptiveScoring struct {
	cacheFn cacheQueryFn
	rng     *rand.Rand
}

// qdCeiling is the queue depth at which base pressure saturates to 1.0.
const qdCeiling = 10.0

// spreadSensitivity controls how quickly load CV translates to pressure.
// CV of 1/spreadSensitivity → full pressure. With sensitivity=4, a CV of 0.25
// (moderate imbalance) triggers full load-balancing.
const spreadSensitivity = 4.0

// Route implements RoutingPolicy for AdaptiveScoring.
func (as *AdaptiveScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("AdaptiveScoring.Route: empty snapshots")
	}

	n := float64(len(snapshots))

	// === Pass 1: Cluster-level signals ===
	var sumQD, sumKV, sumLoad float64
	loads := make([]float64, len(snapshots))
	for i, snap := range snapshots {
		sumQD += float64(snap.QueueDepth)
		sumKV += snap.KVUtilization
		load := float64(snap.EffectiveLoad())
		loads[i] = load
		sumLoad += load
	}
	avgQD := sumQD / n
	avgKV := sumKV / n
	meanLoad := sumLoad / n

	// Load CV: coefficient of variation = std / (mean + 1)
	var sumSq float64
	for _, l := range loads {
		diff := l - meanLoad
		sumSq += diff * diff
	}
	stdLoad := math.Sqrt(sumSq / n)
	loadCV := stdLoad / (meanLoad + 1) // +1 avoids division by zero when idle

	// Spread-driven pressure: high CV → high pressure → load balancing
	spreadPressure := loadCV * spreadSensitivity
	if spreadPressure > 1 {
		spreadPressure = 1
	}

	// Base pressure floor from mean KV and QD (dampened to 30% contribution)
	qdPressure := avgQD / qdCeiling
	if qdPressure > 1.0 {
		qdPressure = 1.0
	}
	basePressure := avgKV
	if qdPressure > basePressure {
		basePressure = qdPressure
	}
	basePressure *= 0.3

	// Effective pressure: max of spread-driven and dampened base
	pressure := spreadPressure
	if basePressure > pressure {
		pressure = basePressure
	}
	if pressure > 1 {
		pressure = 1
	}

	// === Pass 2a: Prefix scores (min-max normalized) ===
	prefixRaw := make(map[string]int, len(snapshots))
	minPrefix, maxPrefix := math.MaxInt, 0
	for _, snap := range snapshots {
		count := 0
		if as.cacheFn != nil && req != nil {
			if fn, ok := as.cacheFn[snap.ID]; ok && fn != nil {
				count = fn(req.InputTokens)
			}
		}
		prefixRaw[snap.ID] = count
		if count < minPrefix {
			minPrefix = count
		}
		if count > maxPrefix {
			maxPrefix = count
		}
	}
	prefixScores := make(map[string]float64, len(snapshots))
	prefixHasDifferentiation := (maxPrefix != minPrefix)
	for _, snap := range snapshots {
		if !prefixHasDifferentiation {
			prefixScores[snap.ID] = 1.0
		} else {
			prefixScores[snap.ID] = float64(prefixRaw[snap.ID]-minPrefix) / float64(maxPrefix-minPrefix)
		}
	}

	// === Pass 2b: Load scores (effective load, inverted min-max) ===
	minLoad, maxLoad := loads[0], loads[0]
	for _, l := range loads[1:] {
		if l < minLoad {
			minLoad = l
		}
		if l > maxLoad {
			maxLoad = l
		}
	}
	loadScores := make(map[string]float64, len(snapshots))
	for i, snap := range snapshots {
		if maxLoad == minLoad {
			loadScores[snap.ID] = 1.0
		} else {
			loadScores[snap.ID] = (maxLoad - loads[i]) / (maxLoad - minLoad)
		}
	}

	// === Pass 3: Adaptive blend ===
	// Prefix information gate: zero variance → pure load score
	alpha := 1.0 - pressure
	beta := pressure
	if !prefixHasDifferentiation {
		alpha = 0
		beta = 1.0
	}

	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		scores[snap.ID] = alpha*prefixScores[snap.ID] + beta*loadScores[snap.ID]
	}

	// === Argmax with random tie-breaking ===
	bestScore := math.Inf(-1)
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
	if len(tied) > 1 && as.rng != nil {
		bestIdx = tied[as.rng.Intn(len(tied))]
	}

	return NewRoutingDecisionWithScores(
		snapshots[bestIdx].ID,
		fmt.Sprintf("adaptive (pressure=%.2f, cv=%.2f, score=%.3f)", pressure, loadCV, bestScore),
		scores,
	)
}
