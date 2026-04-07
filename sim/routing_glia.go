package sim

import (
	"fmt"
	"math/rand"
)

// GliaHRA (Head-Room Allocator) is a KV-cache-aware routing policy.
//
// Instead of using scorer pipelines with weighted combination, Glia estimates
// the KV-cache headroom for each instance after hypothetically placing the
// request. It projects block usage from input tokens (with a decode-to-prompt
// ratio), checks if the instance can fit the request with a safety margin,
// and scores based on projected utilization + queue load. Inadmissible
// instances get a heavy penalty.
//
// Signals used:
//   - FreeKVBlocks (stale ~5s)
//   - KVUtilization (stale ~5s)
//   - QueueDepth + BatchSize + InFlightRequests (mixed freshness)
//   - req.InputTokens (request metadata)
//
// Origin: Adapted from sim2real/blis_router_sweet/baselines/baseline_glia.go.
type GliaHRA struct {
	rng *rand.Rand
}

// gliaHRA parameters.
const (
	gliaDecodeToPromptRatio = 0.6
	gliaSafetyFraction      = 0.03
	gliaBlockSize           = 16.0
)

// Route implements RoutingPolicy for GliaHRA.
func (g *GliaHRA) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("GliaHRA.Route: empty snapshots")
	}

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

	return NewRoutingDecisionWithScores(
		snapshots[bestIdx].ID,
		fmt.Sprintf("glia-hra (score=%.3f)", bestScore),
		scores,
	)
}
