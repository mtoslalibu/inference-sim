package sim

import (
	"math/rand"
	"testing"
)

// TestAdaptiveScoring_Regime1_CacheAffinity verifies that when cache scores
// differ, the adaptive policy uses strong cache affinity (ppc:4, qd:1, no kvu).
func TestAdaptiveScoring_Regime1_CacheAffinity(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "inst_0", QueueDepth: 3, BatchSize: 1, InFlightRequests: 2, KVUtilization: 0.7},
		{ID: "inst_1", QueueDepth: 1, BatchSize: 0, InFlightRequests: 0, KVUtilization: 0.3},
	}
	state := &RouterState{Snapshots: snapshots, Clock: 1000}
	req := &Request{ID: "req1", InputTokens: []int{1, 2, 3, 4, 5}}

	cacheFn := cacheQueryFn{
		"inst_0": func(tokens []int) int { return 5 },
		"inst_1": func(tokens []int) int { return 0 },
	}

	adaptive := newAdaptiveScoring(16, nil, cacheFn)
	decision := adaptive.Route(req, state)

	// inst_0 has cache hits, ppc:4 should outweigh qd:1 despite higher load
	if decision.TargetInstance != "inst_0" {
		t.Errorf("Regime 1: expected inst_0 (cached), got %q", decision.TargetInstance)
	}
}

// TestAdaptiveScoring_Regime2_MemoryAware verifies that when cache is
// equalized but KV utilization is high, kvu is activated.
func TestAdaptiveScoring_Regime2_MemoryAware(t *testing.T) {
	// No cache fn → ppc scores equalize. High KV util → regime 2.
	// inst_0 has high KV util (nearly full), inst_1 has moderate.
	// Same queue depth → kvu should prefer inst_1 (more room).
	snapshots := []RoutingSnapshot{
		{ID: "inst_0", QueueDepth: 2, BatchSize: 0, InFlightRequests: 0, KVUtilization: 0.9},
		{ID: "inst_1", QueueDepth: 2, BatchSize: 0, InFlightRequests: 0, KVUtilization: 0.6},
	}
	state := &RouterState{Snapshots: snapshots, Clock: 1000}
	req := &Request{ID: "req1", InputTokens: []int{1, 2}}

	adaptive := newAdaptiveScoring(16, nil, nil)
	decision := adaptive.Route(req, state)

	// Both have same load (qd ties), kvu should break tie toward inst_1
	if decision.TargetInstance != "inst_1" {
		t.Errorf("Regime 2: expected inst_1 (lower KV util), got %q", decision.TargetInstance)
	}
}

// TestAdaptiveScoring_Regime3_LoadBalance verifies that when cache is
// equalized and memory is spacious, pure load balancing is used.
func TestAdaptiveScoring_Regime3_LoadBalance(t *testing.T) {
	// No cache fn → ppc equalizes. Low KV util → regime 3. Pure qd.
	snapshots := []RoutingSnapshot{
		{ID: "inst_0", QueueDepth: 5, BatchSize: 2, InFlightRequests: 1, KVUtilization: 0.2},
		{ID: "inst_1", QueueDepth: 1, BatchSize: 0, InFlightRequests: 0, KVUtilization: 0.1},
		{ID: "inst_2", QueueDepth: 8, BatchSize: 3, InFlightRequests: 2, KVUtilization: 0.3},
	}
	state := &RouterState{Snapshots: snapshots, Clock: 1000}
	req := &Request{ID: "req1", InputTokens: []int{1, 2, 3}}

	adaptive := newAdaptiveScoring(16, nil, nil)
	decision := adaptive.Route(req, state)

	// inst_1 has lowest effective load (1), should be selected
	if decision.TargetInstance != "inst_1" {
		t.Errorf("Regime 3: expected inst_1 (lowest load), got %q", decision.TargetInstance)
	}
}

// TestAdaptiveScoring_Regime2_KVUDoesNotFightPPC verifies that kvu only
// activates when ppc is irrelevant (cache equalized).
func TestAdaptiveScoring_Regime2_KVUDoesNotFightPPC(t *testing.T) {
	// When cache IS differentiated, kvu should NOT be active even with high KV util
	snapshots := []RoutingSnapshot{
		{ID: "inst_0", QueueDepth: 2, BatchSize: 0, InFlightRequests: 0, KVUtilization: 0.9},
		{ID: "inst_1", QueueDepth: 2, BatchSize: 0, InFlightRequests: 0, KVUtilization: 0.3},
	}
	state := &RouterState{Snapshots: snapshots, Clock: 1000}
	req := &Request{ID: "req1", InputTokens: []int{1, 2, 3}}

	cacheFn := cacheQueryFn{
		"inst_0": func(tokens []int) int { return 5 },
		"inst_1": func(tokens []int) int { return 0 },
	}

	adaptive := newAdaptiveScoring(16, nil, cacheFn)
	decision := adaptive.Route(req, state)

	// Despite inst_0 having 0.9 KV util, cache affinity (regime 1) should win
	if decision.TargetInstance != "inst_0" {
		t.Errorf("Expected regime 1 (cache-affinity) to inst_0, got %q", decision.TargetInstance)
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
