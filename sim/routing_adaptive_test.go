package sim

import (
	"math/rand"
	"testing"
)

// TestAdaptiveScoring_CacheEqualized_PureLoadBalance verifies that when all
// instances have the same cache score, the adaptive policy falls back to
// pure load balancing (no ppc influence).
func TestAdaptiveScoring_CacheEqualized_PureLoadBalance(t *testing.T) {
	// GIVEN: no cache function (all ppc scores equalize to 0.5)
	snapshots := []RoutingSnapshot{
		{ID: "inst_0", QueueDepth: 5, BatchSize: 2, InFlightRequests: 1, KVUtilization: 0.3},
		{ID: "inst_1", QueueDepth: 1, BatchSize: 0, InFlightRequests: 0, KVUtilization: 0.1},
		{ID: "inst_2", QueueDepth: 8, BatchSize: 3, InFlightRequests: 2, KVUtilization: 0.5},
	}
	state := &RouterState{Snapshots: snapshots, Clock: 1000}
	req := &Request{ID: "req1", InputTokens: []int{1, 2, 3}}

	// Build pure qd policy for comparison
	qdOnly := newRoutingPolicyInternal("weighted", []ScorerConfig{
		{Name: "queue-depth", Weight: 1.0},
	}, 16, nil, nil)
	adaptive := newAdaptiveScoring(16, nil, nil)

	// WHEN: both route the same request
	qdDecision := qdOnly.Route(req, state)
	adaptiveDecision := adaptive.Route(req, state)

	// THEN: adaptive should route to the same instance as pure qd
	// (inst_1 has lowest effective load = 1)
	if adaptiveDecision.TargetInstance != qdDecision.TargetInstance {
		t.Errorf("Cache equalized: adaptive chose %q, qd-only chose %q",
			adaptiveDecision.TargetInstance, qdDecision.TargetInstance)
	}
	if adaptiveDecision.TargetInstance != "inst_1" {
		t.Errorf("Expected inst_1 (lowest load), got %q", adaptiveDecision.TargetInstance)
	}
}

// TestAdaptiveScoring_CacheDifferentiated_StrongAffinity verifies that when
// cache scores differ, the adaptive policy uses strong cache affinity.
func TestAdaptiveScoring_CacheDifferentiated_StrongAffinity(t *testing.T) {
	// GIVEN: cache function where inst_0 has many cached blocks, inst_1 has none
	snapshots := []RoutingSnapshot{
		{ID: "inst_0", QueueDepth: 3, BatchSize: 1, InFlightRequests: 2, KVUtilization: 0.7},
		{ID: "inst_1", QueueDepth: 1, BatchSize: 0, InFlightRequests: 0, KVUtilization: 0.3},
	}
	state := &RouterState{Snapshots: snapshots, Clock: 1000}
	req := &Request{ID: "req1", InputTokens: []int{1, 2, 3, 4, 5}}

	cacheFn := cacheQueryFn{
		"inst_0": func(tokens []int) int { return 5 }, // 5 cached blocks
		"inst_1": func(tokens []int) int { return 0 }, // no cache
	}

	adaptive := newAdaptiveScoring(16, nil, cacheFn)

	// WHEN: route a request
	decision := adaptive.Route(req, state)

	// THEN: should route to inst_0 (cached instance) despite higher load
	// because ppc:4 outweighs qd:1
	if decision.TargetInstance != "inst_0" {
		t.Errorf("Cache differentiated: expected inst_0 (cached), got %q", decision.TargetInstance)
	}
}

// TestAdaptiveScoring_NeverUsesKVU verifies kvu is not a factor.
func TestAdaptiveScoring_NeverUsesKVU(t *testing.T) {
	// GIVEN: two instances with extreme KV utilization difference but same load
	// If kvu were active, it would prefer inst_1 (lower utilization)
	snapshots := []RoutingSnapshot{
		{ID: "inst_0", QueueDepth: 2, BatchSize: 0, InFlightRequests: 0, KVUtilization: 0.95},
		{ID: "inst_1", QueueDepth: 2, BatchSize: 0, InFlightRequests: 0, KVUtilization: 0.05},
	}
	state := &RouterState{Snapshots: snapshots, Clock: 1000}
	req := &Request{ID: "req1", InputTokens: []int{1, 2}}

	adaptive := newAdaptiveScoring(16, nil, nil)

	// WHEN: route (no cache fn, ppc scores equalize, pure qd mode)
	decision := adaptive.Route(req, state)

	// THEN: both instances have same load, kvu should NOT break the tie
	// deterministically toward inst_1. With nil rng, first instance wins.
	if decision.TargetInstance != "inst_0" {
		t.Errorf("KVU influence detected: expected inst_0 (first tie), got %q",
			decision.TargetInstance)
	}
}

// TestAdaptiveScoring_EmptySnapshots_Panics verifies panic on empty input.
func TestAdaptiveScoring_EmptySnapshots_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic on empty snapshots, got none")
		}
	}()

	adaptive := newAdaptiveScoring(16, nil, nil)
	req := &Request{ID: "req1"}
	adaptive.Route(req, &RouterState{Snapshots: []RoutingSnapshot{}, Clock: 1000})
}

// TestAdaptiveScoring_TieBreaking verifies random tie-breaking with rng.
func TestAdaptiveScoring_TieBreaking(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "inst_0", QueueDepth: 0, KVUtilization: 0.1},
		{ID: "inst_1", QueueDepth: 0, KVUtilization: 0.1},
	}
	state := &RouterState{Snapshots: snapshots, Clock: 1000}

	rng := rand.New(rand.NewSource(42))
	adaptive := newAdaptiveScoring(16, rng, nil)

	counts := map[string]int{}
	for i := 0; i < 100; i++ {
		req := &Request{ID: "req", InputTokens: []int{1}}
		decision := adaptive.Route(req, state)
		counts[decision.TargetInstance]++
	}

	if counts["inst_0"] == 0 || counts["inst_1"] == 0 {
		t.Errorf("Expected both instances selected with rng, got: %v", counts)
	}
}

// TestAdaptiveScoring_FactoryCreation verifies the policy can be created via the factory.
func TestAdaptiveScoring_FactoryCreation(t *testing.T) {
	policy := NewRoutingPolicy("adaptive", nil, 16, nil)
	if policy == nil {
		t.Fatal("Expected non-nil policy from factory")
	}

	snapshots := []RoutingSnapshot{
		{ID: "inst_0", QueueDepth: 1, KVUtilization: 0.3},
	}
	decision := policy.Route(
		&Request{ID: "req1", InputTokens: []int{1}},
		&RouterState{Snapshots: snapshots, Clock: 1000},
	)
	if decision.TargetInstance != "inst_0" {
		t.Errorf("Expected inst_0, got %q", decision.TargetInstance)
	}
}
