package sim

import "fmt"

// GAIEControlAdmission implements the GAIE legacy admission logic as an AdmissionPlugin.
// Same structure as QuinticAdmission (own saturation detector, same plugin interface),
// but uses the standard GAIE binary shedding rule: reject all sheddable requests
// (priority < 0) when saturation >= 1.0. No probabilistic ramp — just a hard cutoff.
//
// Purpose: deploy alongside QuinticAdmission as a control group to isolate the effect
// of quintic probabilistic shedding vs the plugin framework itself.
//
// Transfer to llm-d: same AdmissionPlugin interface and signal mapping as admission.go.
type GAIEControlAdmission struct {
	totalAdmitted int
	totalRejected int
}

func NewGAIEControlAdmission() *GAIEControlAdmission {
	return &GAIEControlAdmission{}
}

// saturationControl computes pool-average saturation per GAIE formula:
// avg across instances of max(queueDepth/5.0, kvUtil/0.8).
// Identical to computeSaturation() in admission.go and GAIE's utilization/detector.go.
// Empty snapshots → 1.0 (conservative, matches GAIE detector.go:116-118).
func saturationControl(snapshots []RoutingSnapshot) float64 {
	n := len(snapshots)
	if n == 0 {
		return 1.0
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

// Admit implements AdmissionPolicy.
//
// GAIE legacy logic:
//   - priority >= 0 (critical, standard): always admit
//   - priority < 0 (sheddable, batch, background): reject when saturation >= 1.0
//
// This is a binary cutoff — no probabilistic ramp, no graduated shedding.
func (g *GAIEControlAdmission) Admit(req *Request, state *RouterState) (bool, string) {
	sloClass := req.SLOClass

	switch sloClass {
	case "critical", "standard":
		// Protected tiers: never reject.
		g.totalAdmitted++
		return true, ""
	}

	// All other classes (sheddable, batch, background): reject at saturation >= 1.0.
	sat := saturationControl(state.Snapshots)
	if sat >= 1.0 {
		g.totalRejected++
		return false, fmt.Sprintf("gaie-control: class=%s saturation=%.3f >= 1.0", sloClass, sat)
	}

	g.totalAdmitted++
	return true, ""
}
