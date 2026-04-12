package sim

import (
	"fmt"
	"math"
)

// AdmissionPolicy decides whether a request is admitted for processing.
// Used by ClusterSimulator's online routing pipeline to gate incoming requests.
// Receives *RouterState with cluster-wide snapshots and clock.
type AdmissionPolicy interface {
	Admit(req *Request, state *RouterState) (admitted bool, reason string)
}

// AlwaysAdmit admits all requests unconditionally.
type AlwaysAdmit struct{}

func (a *AlwaysAdmit) Admit(_ *Request, _ *RouterState) (bool, string) {
	return true, ""
}

// TokenBucket implements rate-limiting admission control.
type TokenBucket struct {
	capacity      float64
	refillRate    float64 // tokens per second
	currentTokens float64
	lastRefill    int64 // last refill clock time in microseconds
}

// NewTokenBucket creates a TokenBucket with the given capacity and refill rate.
// Panics if capacity or refillRate is <= 0, NaN, or Inf (R3: validate at construction).
func NewTokenBucket(capacity, refillRate float64) *TokenBucket {
	if capacity <= 0 || math.IsNaN(capacity) || math.IsInf(capacity, 0) {
		panic(fmt.Sprintf("NewTokenBucket: capacity must be a finite value > 0, got %v", capacity))
	}
	if refillRate <= 0 || math.IsNaN(refillRate) || math.IsInf(refillRate, 0) {
		panic(fmt.Sprintf("NewTokenBucket: refillRate must be a finite value > 0, got %v", refillRate))
	}
	return &TokenBucket{
		capacity:      capacity,
		refillRate:    refillRate,
		currentTokens: capacity,
	}
}

// Admit checks whether the request can be admitted given current token availability.
func (tb *TokenBucket) Admit(req *Request, state *RouterState) (bool, string) {
	clock := state.Clock
	elapsed := clock - tb.lastRefill
	if elapsed > 0 {
		refill := float64(elapsed) * tb.refillRate / 1e6
		tb.currentTokens = min(tb.capacity, tb.currentTokens+refill)
		tb.lastRefill = clock
	}
	cost := float64(len(req.InputTokens))
	if tb.currentTokens >= cost {
		tb.currentTokens -= cost
		return true, ""
	}
	return false, "insufficient tokens"
}

// GAIELegacyAdmission reproduces the default GAIE LegacyAdmissionController behavior:
// compute utilization-based saturation (avg of max(qd/thresh, kv/thresh) per instance),
// and reject sheddable requests (SLOClass "sheddable", "batch", "background") when
// saturation >= 1.0. Non-sheddable requests (critical, standard) always pass through.
// Use NewGAIELegacyAdmission to construct with validated parameters.
type GAIELegacyAdmission struct {
	QueueDepthThreshold  float64 // per-instance queue depth threshold (GAIE default: 5)
	KVCacheUtilThreshold float64 // per-instance KV cache util threshold (GAIE default: 0.8)
}

// NewGAIELegacyAdmission creates a GAIELegacyAdmission with validated parameters.
// Panics if thresholds are <= 0, NaN, or Inf (R3).
func NewGAIELegacyAdmission(queueDepthThreshold, kvCacheUtilThreshold float64) *GAIELegacyAdmission {
	if queueDepthThreshold <= 0 || math.IsNaN(queueDepthThreshold) || math.IsInf(queueDepthThreshold, 0) {
		panic(fmt.Sprintf("NewGAIELegacyAdmission: queueDepthThreshold must be > 0, got %v", queueDepthThreshold))
	}
	if kvCacheUtilThreshold <= 0 || math.IsNaN(kvCacheUtilThreshold) || math.IsInf(kvCacheUtilThreshold, 0) {
		panic(fmt.Sprintf("NewGAIELegacyAdmission: kvCacheUtilThreshold must be > 0, got %v", kvCacheUtilThreshold))
	}
	return &GAIELegacyAdmission{
		QueueDepthThreshold:  queueDepthThreshold,
		KVCacheUtilThreshold: kvCacheUtilThreshold,
	}
}

// Admit implements AdmissionPolicy. Rejects sheddable requests (GAIE priority < 0)
// when utilization-based saturation >= 1.0. Critical and standard always pass through.
func (g *GAIELegacyAdmission) Admit(req *Request, state *RouterState) (bool, string) {
	// GAIE: priority >= 0 always admitted, priority < 0 is sheddable.
	priority := SLOTierPriorityGAIE(req.SLOClass)
	if priority >= 0 {
		return true, ""
	}

	// Compute utilization-based saturation: avg(max(qd/thresh, kv/thresh)) per instance.
	// Same formula as GAIE's utilization/detector.go.
	if len(state.Snapshots) == 0 {
		return true, "" // no instances → admit (safe default)
	}
	satSum := 0.0
	for _, snap := range state.Snapshots {
		qRatio := float64(snap.QueueDepth) / g.QueueDepthThreshold
		kvRatio := snap.KVUtilization / g.KVCacheUtilThreshold
		satSum += math.Max(qRatio, kvRatio)
	}
	saturation := satSum / float64(len(state.Snapshots))

	if saturation >= 1.0 {
		return false, fmt.Sprintf("gaie-legacy: class=%s saturation=%.3f >= 1.0", req.SLOClass, saturation)
	}
	return true, ""
}

// RejectAll rejects all requests unconditionally (pathological template for testing).
type RejectAll struct{}

func (r *RejectAll) Admit(_ *Request, _ *RouterState) (bool, string) {
	return false, "reject-all"
}

// SLOTierPriority maps an SLOClass string to an integer priority.
// Higher = more important. Background=0 … Critical=4.
// Empty or unknown string maps to Standard (3) for backward compatibility.
// Exported so sim/cluster/ can call it without a circular import.
func SLOTierPriority(class string) int {
	switch class {
	case "critical":
		return 4
	case "standard":
		return 3
	case "sheddable":
		return 2
	case "batch":
		return 1
	case "background":
		return 0
	default:
		return 3 // empty or unknown → Standard
	}
}

// SLOTierPriorityGAIE maps an SLOClass string to GAIE-compatible priority.
// In GAIE, priority < 0 is sheddable (rejected under saturation), priority >= 0
// is protected (always admitted). This mapping preserves that boundary:
//   critical=4, standard=3, sheddable=-1, batch=-2.
// Four tiers only (no background). Empty or unknown string maps to Standard (3).
func SLOTierPriorityGAIE(class string) int {
	switch class {
	case "critical":
		return 4
	case "standard":
		return 3
	case "sheddable":
		return -1
	case "batch":
		return -2
	default:
		return 3 // empty or unknown → Standard
	}
}

// TierShedAdmission sheds lower-priority requests under overload.
// Stateless: all decisions computed from RouterState at call time.
// Batch and Background always pass through (deferred queue PR handles them).
// Use NewTierShedAdmission to construct with validated parameters.
type TierShedAdmission struct {
	OverloadThreshold int // max per-instance effective load before shedding; 0 = any load triggers
	MinAdmitPriority  int // minimum tier priority admitted under overload; 0 = admit all (footgun)
}

// NewTierShedAdmission creates a TierShedAdmission with validated parameters.
// Panics if overloadThreshold < 0 or minAdmitPriority is outside [0, 4] (R3).
func NewTierShedAdmission(overloadThreshold, minAdmitPriority int) *TierShedAdmission {
	if overloadThreshold < 0 {
		panic(fmt.Sprintf("NewTierShedAdmission: overloadThreshold must be >= 0, got %d", overloadThreshold))
	}
	if minAdmitPriority < 0 || minAdmitPriority > 4 {
		panic(fmt.Sprintf("NewTierShedAdmission: minAdmitPriority must be in [0,4], got %d", minAdmitPriority))
	}
	return &TierShedAdmission{
		OverloadThreshold: overloadThreshold,
		MinAdmitPriority:  minAdmitPriority,
	}
}

// Admit rejects requests whose tier priority is below MinAdmitPriority when the
// cluster is overloaded (max effective load across instances > OverloadThreshold).
// Batch and Background classes always return admitted=true.
// Empty Snapshots (no instances) also returns admitted=true (safe default).
func (t *TierShedAdmission) Admit(req *Request, state *RouterState) (bool, string) {
	class := req.SLOClass
	// Batch/Background bypass tier-shed (deferred queue handles them in PR-2).
	if class == "batch" || class == "background" {
		return true, ""
	}
	// Compute max effective load across all instance snapshots.
	maxLoad := 0
	for _, snap := range state.Snapshots {
		if l := snap.EffectiveLoad(); l > maxLoad {
			maxLoad = l
		}
	}
	if maxLoad <= t.OverloadThreshold {
		return true, "" // under threshold: admit all
	}
	// Under overload: reject tiers below MinAdmitPriority.
	priority := SLOTierPriority(class)
	if priority < t.MinAdmitPriority {
		return false, fmt.Sprintf("tier-shed: class=%s priority=%d < min=%d load=%d",
			class, priority, t.MinAdmitPriority, maxLoad)
	}
	return true, ""
}

// AdaptiveAdmission implements cluster-aware, SLO-aware admission control.
// The Admit() logic inside the EVOLVE-BLOCK is mutated by the strategy evolution framework.
// Stateful fields enable rate estimation, tenant fairness, and per-class tracking.
type AdaptiveAdmission struct {
	tenantTokens   map[string]float64 // per-tenant token budget tracker
	tenantRequests map[string]int     // per-tenant request counter
	classCounters  map[string]int     // per-SLO-class admission counter
	windowStart    int64              // sliding window start (microseconds)
	windowCount    int                // requests in current window
	totalAdmitted  int
	totalRejected  int
	lastClock      int64
}

// NewAdaptiveAdmission creates an AdaptiveAdmission with pre-initialized state maps.
func NewAdaptiveAdmission() *AdaptiveAdmission {
	return &AdaptiveAdmission{
		tenantTokens:   make(map[string]float64),
		tenantRequests: make(map[string]int),
		classCounters:  make(map[string]int),
	}
}

// Admit implements AdmissionPolicy for AdaptiveAdmission.
func (a *AdaptiveAdmission) Admit(req *Request, state *RouterState) (bool, string) {
	// --- Derived signals (fixed, available to EVOLVE-BLOCK) ---
	numInstances := len(state.Snapshots)
	totalInFlight := 0
	totalQueueDepth := 0
	maxKVUtil := 0.0
	avgKVUtil := 0.0
	minFreeKV := int64(math.MaxInt64)
	for _, snap := range state.Snapshots {
		totalInFlight += snap.InFlightRequests
		totalQueueDepth += snap.QueueDepth
		if snap.KVUtilization > maxKVUtil {
			maxKVUtil = snap.KVUtilization
		}
		avgKVUtil += snap.KVUtilization
		if snap.FreeKVBlocks < minFreeKV {
			minFreeKV = snap.FreeKVBlocks
		}
	}
	if numInstances > 0 {
		avgKVUtil /= float64(numInstances)
	}
	inputLen := len(req.InputTokens)
	sloClass := req.SLOClass
	tenantID := req.TenantID
	clock := state.Clock

	// Suppress "unused variable" errors for signals the evolved block may or may not use.
	_ = numInstances
	_ = totalInFlight
	_ = totalQueueDepth
	_ = maxKVUtil
	_ = avgKVUtil
	_ = minFreeKV
	_ = inputLen
	_ = sloClass
	_ = tenantID
	_ = clock

	// EVOLVE-BLOCK-START
	// Iteration 11: Ultra-preemptive GAIE-formula shedding — model-agnostic.
	// Key insight: even when both algorithms eventually shed the same total,
	// WHEN they start matters enormously — early shedding prevents queue buildup
	// from compounding. This iteration starts at near-zero thresholds.
	// Transfer to llm-d: AdmissionPlugin with shedStart/shedFull parameters.

	// GAIE saturation formula (identical to production utilization/detector.go)
	saturation := 0.0
	if numInstances > 0 {
		for _, snap := range state.Snapshots {
			qRatio := float64(snap.QueueDepth) / 5.0
			kvRatio := snap.KVUtilization / 0.8
			if qRatio > kvRatio {
				saturation += qRatio
			} else {
				saturation += kvRatio
			}
		}
		saturation /= float64(numInstances)
	}

	switch sloClass {
	case "critical":
		// NEVER reject. Protected tier (priority >= 0 in GAIE terms).

	case "standard":
		// NEVER reject. Protected tier.

	case "sheddable":
		// Ultra-preemptive shedding: ramp from 0.01 to 0.10.
		// At sat=0.10 (avg QD=0.5 or KV=8%): shed ALL sheddable.
		// Tighter than iter9 (0.02→0.15) to catch early KV buildup in large models.
		if saturation >= 0.01 {
			p := (saturation - 0.01) / 0.09 // 0→1 over [0.01, 0.10]
			if p > 1.0 {
				p = 1.0
			}
			requestOrdinal := float64(a.totalAdmitted+a.totalRejected) / 100.0
			randVal := requestOrdinal - float64(int(requestOrdinal))
			if randVal < p {
				a.totalRejected++
				return false, fmt.Sprintf("adaptive: sheddable-shed sat=%.3f p=%.2f", saturation, p)
			}
		}

	case "batch":
		// Batch: most aggressive — start at 0.005, full at 0.05.
		if saturation >= 0.005 {
			p := (saturation - 0.005) / 0.045
			if p > 1.0 {
				p = 1.0
			}
			requestOrdinal := float64(a.totalAdmitted+a.totalRejected) / 100.0
			randVal := requestOrdinal - float64(int(requestOrdinal))
			if randVal < p {
				a.totalRejected++
				return false, fmt.Sprintf("adaptive: batch-shed sat=%.3f p=%.2f", saturation, p)
			}
		}
	}

	_ = totalQueueDepth
	_ = avgKVUtil
	_ = maxKVUtil
	_ = minFreeKV
	_ = totalInFlight
	_ = tenantID
	_ = inputLen

	a.classCounters[sloClass]++
	a.totalAdmitted++
	return true, ""
	// EVOLVE-BLOCK-END
}

// NewAdmissionPolicy creates an admission policy by name.
// Valid names are defined in ValidAdmissionPolicies (bundle.go).
// An empty string defaults to AlwaysAdmit (for CLI flag default compatibility).
// For token-bucket, capacity and refillRate configure the bucket.
// Panics on unrecognized names.
func NewAdmissionPolicy(name string, capacity, refillRate float64) AdmissionPolicy {
	if !IsValidAdmissionPolicy(name) {
		panic(fmt.Sprintf("unknown admission policy %q", name))
	}
	switch name {
	case "", "always-admit":
		return &AlwaysAdmit{}
	case "token-bucket":
		return NewTokenBucket(capacity, refillRate)
	case "reject-all":
		return &RejectAll{}
	case "gaie-legacy":
		// Default GAIE thresholds: queueDepth=5, kvCacheUtil=0.8.
		// Override via DeploymentConfig fields when wired through cluster.go.
		return NewGAIELegacyAdmission(5.0, 0.8)
	case "adaptive-admission":
		return NewAdaptiveAdmission()
	default:
		panic(fmt.Sprintf("unhandled admission policy %q", name))
	}
}
