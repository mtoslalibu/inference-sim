package sim

import "fmt"

// QuinticAdmission implements parameter-free binary-tier probabilistic shedding.
// Uses the standard GAIE saturation formula with a quintic (5th-power) probability ramp.
//
// Binary tier model:
//   - Protected (priority >= 0): critical, standard — always admit
//   - Droppable (priority < 0): sheddable, batch, background — quintic rejection at k=300
//
// The quintic power law creates a natural "dead zone" at low saturation (~1% shed
// at half capacity) while ramping steeply at moderate overload.
//
// Transfer to llm-d: implement requestcontrol.AdmissionPlugin, map signals per README.md.
//   - QueueDepth → pod.GetMetrics().WaitingQueueSize
//   - KVUtilization → pod.GetMetrics().KVCacheUsagePercent (already 0-1 despite the name)
//   - sloClass → request.Objectives.Priority (>=0 protected, <0 droppable)
type QuinticAdmission struct {
	totalAdmitted int
	totalRejected int
}

func NewQuinticAdmission() *QuinticAdmission {
	return &QuinticAdmission{}
}

// Admit implements AdmissionPolicy.
//
// Decision logic (binary quintic):
//   - priority >= 0 (critical, standard): always admit
//   - priority < 0  (sheddable, batch, background): p = min(sat^5 * 300, 1.0)
//
// k=300 means 100% shed at saturation ~0.34.
func (a *QuinticAdmission) Admit(req *Request, state *RouterState) (bool, string) {
	sloClass := req.SLOClass

	// Cluster saturation (GAIE formula: avg across instances of max(QD/5.0, KV/0.8))
	saturation := computeSaturation(state.Snapshots)

	switch sloClass {
	case "critical", "standard":
		// Protected tiers: never reject.

	default:
		// All droppable tiers (sheddable, batch, background): quintic at k=300
		sat5 := saturation * saturation * saturation * saturation * saturation
		p := sat5 * 300.0
		if p > 1.0 {
			p = 1.0
		}
		if p > 0 {
			if a.pseudoRandom() < p {
				a.totalRejected++
				return false, fmt.Sprintf("quintic: %s-shed sat=%.3f p=%.2f", sloClass, saturation, p)
			}
		}
	}

	a.totalAdmitted++
	return true, ""
}

// computeSaturation computes pool-average saturation per GAIE formula:
// avg across instances of max(queueDepth/5.0, kvUtil/0.8).
func computeSaturation(snapshots []RoutingSnapshot) float64 {
	n := len(snapshots)
	if n == 0 {
		return 0.0
	}
	var total float64
	for _, snap := range snapshots {
		qRatio := float64(snap.QueueDepth) / 5.0
		kvRatio := snap.KVUtilization / 0.8
		if qRatio > kvRatio {
			total += qRatio
		} else {
			total += kvRatio
		}
	}
	return total / float64(n)
}

// pseudoRandom returns a deterministic pseudo-random value in [0, 1) based on request count.
func (a *QuinticAdmission) pseudoRandom() float64 {
	ordinal := float64(a.totalAdmitted+a.totalRejected) / 100.0
	return ordinal - float64(int(ordinal))
}
