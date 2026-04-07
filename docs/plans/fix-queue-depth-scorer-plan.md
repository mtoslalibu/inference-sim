# Fix queue-depth scorer to read QueueDepth only (GIE parity)

**Goal:** Make the `queue-depth` scorer read only `QueueDepth` instead of `EffectiveLoad()` (which adds `BatchSize + InFlightRequests`), matching the real GIE `queue-scorer` which reads `WaitingQueueSize` only.

**Source:** [#955](https://github.com/inference-sim/inference-sim/issues/955), parent [#953](https://github.com/inference-sim/inference-sim/issues/953)

**Closes:** `Fixes #955`

**PR Size Tier:** Small (4 files changed — `sim/routing_scorers.go`, `sim/routing_scorers_test.go`, `sim/routing.go`, `docs/guide/routing.md` — no new interfaces/types/CLI flags)

**Source Document Audit:** The issue references `docs/routing-study.md` for a documentation update, but this file does not exist in the codebase. The parity comparison tables live only in the GitHub issues (#953). No documentation file requires updating. This is logged as CLARIFICATION — the doc update acceptance criterion is satisfied by updating the code comments only.

## Behavioral Contracts

**BC-1: QueueDepth-only scoring**
- GIVEN snapshots with varying `QueueDepth`, `BatchSize`, and `InFlightRequests`
- WHEN `scoreQueueDepth` is called
- THEN scores depend ONLY on `QueueDepth` values — identical `QueueDepth` produces identical scores regardless of `BatchSize` or `InFlightRequests`

**BC-2: Min-max normalization preserved**
- GIVEN snapshots with different `QueueDepth` values
- WHEN `scoreQueueDepth` is called
- THEN the instance with the lowest `QueueDepth` scores 1.0, the highest scores 0.0, and all others are linearly interpolated between 0.0 and 1.0

**BC-3: Uniform queue depth**
- GIVEN all snapshots have the same `QueueDepth`
- WHEN `scoreQueueDepth` is called
- THEN all instances score 1.0 (regardless of differing `BatchSize` or `InFlightRequests`)

**BC-4: EffectiveLoad still available**
- GIVEN existing callers of `EffectiveLoad()` (load-balance scorer, LeastLoaded policy, admission, counterfactual)
- WHEN the queue-depth scorer is changed
- THEN `EffectiveLoad()` remains unchanged and all existing callers continue to work

## Tasks

### Task 1: Update tests to verify QueueDepth-only scoring (BC-1, BC-3)

**Files:** modify `sim/routing_scorers_test.go`

**What to change:**

The existing test `TestScoreQueueDepth_IncludesInFlightRequests` (line 179) currently **asserts** that `InFlightRequests` affects scores. This test must be replaced with a test that asserts the **opposite** — that `BatchSize` and `InFlightRequests` do NOT affect scores.

The existing test `TestScoreQueueDepth_UniformLoad_AllScoreOne` (line 169) uses snapshots with equal `QueueDepth=5` and equal `BatchSize=3`. It passes today and will still pass after the fix, but it doesn't prove BC-3 (that differing `BatchSize`/`InFlightRequests` don't matter). We should add a scenario where `QueueDepth` is uniform but `BatchSize` and `InFlightRequests` differ.

**Test code — replace `TestScoreQueueDepth_IncludesInFlightRequests` (lines 179-188) with:**

```go
// TestScoreQueueDepth_IgnoresBatchSizeAndInFlight verifies BC-1:
// GIVEN snapshots with identical QueueDepth but different BatchSize and InFlightRequests
// WHEN scoreQueueDepth is called
// THEN all instances score identically (BatchSize and InFlightRequests are ignored)
func TestScoreQueueDepth_IgnoresBatchSizeAndInFlight(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 3, BatchSize: 0, InFlightRequests: 0},
		{ID: "b", QueueDepth: 3, BatchSize: 10, InFlightRequests: 0},
		{ID: "c", QueueDepth: 3, BatchSize: 0, InFlightRequests: 20},
		{ID: "d", QueueDepth: 3, BatchSize: 10, InFlightRequests: 20},
	}
	scores := scoreQueueDepth(nil, snapshots)
	// All have same QueueDepth → all score 1.0 (uniform case)
	assert.Equal(t, 1.0, scores["a"])
	assert.Equal(t, 1.0, scores["b"])
	assert.Equal(t, 1.0, scores["c"])
	assert.Equal(t, 1.0, scores["d"])
}
```

**Also add a test for BC-1 with non-uniform QueueDepth** (append after the new test above):

```go
// TestScoreQueueDepth_OnlyQueueDepthAffectsScore verifies BC-1 with non-uniform QueueDepth:
// GIVEN snapshots with different QueueDepth AND different BatchSize/InFlightRequests
// WHEN scoreQueueDepth is called
// THEN scores depend only on QueueDepth (min-max over QueueDepth)
func TestScoreQueueDepth_OnlyQueueDepthAffectsScore(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 10, BatchSize: 0, InFlightRequests: 0},   // highest QueueDepth
		{ID: "b", QueueDepth: 0, BatchSize: 100, InFlightRequests: 50}, // lowest QueueDepth, huge batch+inflight
	}
	scores := scoreQueueDepth(nil, snapshots)
	// "b" has QueueDepth=0 (min) → scores 1.0, despite having large BatchSize and InFlightRequests
	assert.Equal(t, 1.0, scores["b"], "lowest QueueDepth should score 1.0 regardless of BatchSize/InFlightRequests")
	// "a" has QueueDepth=10 (max) → scores 0.0, despite having zero BatchSize and InFlightRequests
	assert.Equal(t, 0.0, scores["a"], "highest QueueDepth should score 0.0 regardless of BatchSize/InFlightRequests")
}
```

**Verify:** `go test ./sim/... -run TestScoreQueueDepth -v`
- `TestScoreQueueDepth_IgnoresBatchSizeAndInFlight` — should **FAIL** (currently uses `EffectiveLoad()`, so different `BatchSize`/`InFlightRequests` give different effective loads, meaning they won't all be uniform)
- `TestScoreQueueDepth_OnlyQueueDepthAffectsScore` — should **FAIL** (instance "b" with `EffectiveLoad()=150` will score 0.0, not 1.0)
- `TestScoreQueueDepth_MinMaxNormalization` — should still **PASS** (its snapshots have `BatchSize=0, InFlightRequests=0`)
- `TestScoreQueueDepth_UniformLoad_AllScoreOne` — should still **PASS** (its snapshots have equal `EffectiveLoad()`)

**Lint:** `golangci-lint run ./sim/...`

### Task 2: Fix scoreQueueDepth to use QueueDepth only (BC-1, BC-2, BC-3)

**Files:** modify `sim/routing_scorers.go`, modify `sim/routing.go`, modify `docs/guide/routing.md`

**What to change — 3 edits in `scoreQueueDepth` + 3 stale comment fixes across other files:**

1. **Line 127-135 (doc comment):** Replace the entire comment block with:

```go
// scoreQueueDepth computes per-instance queue depth scores using min-max normalization.
// Lower queue depth → higher score. All-equal depths → all score 1.0.
// Matches GIE's queue-scorer semantics: reads QueueDepth only (WaitingQueueSize).
//
// Signal freshness (R17, INV-7):
//
//	Reads: QueueDepth (Periodic when interval>0, else Immediate).
```

2. **Line 140:** Change `load := snap.EffectiveLoad()` → `load := snap.QueueDepth`

3. **Line 152:** Change `load := snap.EffectiveLoad()` → `load := snap.QueueDepth`

4. **`sim/routing.go:160` (stale comment in `WeightedScoring` docstring):** Change `queue-depth (min-max normalization of EffectiveLoad),` → `queue-depth (min-max normalization of QueueDepth),`

5. **`sim/routing_scorers.go:183` (stale comment in `scoreLoadBalance` docstring):** Change `Reads: EffectiveLoad() — same as scoreQueueDepth (synchronous + Periodic composite).` → `Reads: EffectiveLoad() = QueueDepth + BatchSize + InFlightRequests (synchronous + Periodic composite).` (Remove the "same as scoreQueueDepth" reference since they now read different signals.)

6. **`docs/guide/routing.md:34` (stale table entry):** Change `| \`queue-depth\` | Effective load: \`QueueDepth + BatchSize + InFlightRequests\` (min-max normalized) | queue-scorer |` → `| \`queue-depth\` | Queue depth: \`QueueDepth\` only (min-max normalized) | queue-scorer |`

7. **`docs/guide/routing.md:80` (stale signal freshness claim):** Change `queue-depth's Immediate signal corrects stale KV signals` → `queue-depth's load signal complements stale KV signals` (queue-depth now reads Periodic QueueDepth, not Immediate InFlightRequests).

That's it. The min-max formula, variable names, and return type are unchanged.

**Verify:** `go test ./sim/... -run TestScoreQueueDepth -v`
- All 4 `TestScoreQueueDepth_*` tests should **PASS**

**Also run full test suite:** `go test ./...`
- All tests must pass. Existing tests that call `scoreQueueDepth` with `BatchSize=0, InFlightRequests=0` are unaffected.

**Lint:** `golangci-lint run ./sim/...`

### Task 3: Verify EffectiveLoad callers are unaffected (BC-4)

This is a verification-only task — no code changes.

**Check:** Grep for all `EffectiveLoad()` callers and confirm none are in `scoreQueueDepth`:

```bash
grep -n 'EffectiveLoad()' sim/routing.go sim/routing_scorers.go sim/admission.go sim/cluster/counterfactual.go
```

Expected remaining callers (unchanged):
- `sim/routing.go:124,126,134` — `LeastLoaded` policy
- `sim/routing.go:250,254` — `WeightedScoring` tie-breaking (via `selectBestCandidate`)
- `sim/routing_scorers.go:187` — `scoreLoadBalance`
- `sim/admission.go:128` — admission policy
- `sim/cluster/counterfactual.go:51` — counterfactual scoring

`scoreQueueDepth` should have ZERO `EffectiveLoad()` calls.

**Verify:** `go test ./... -count=1` (full suite, no caching)

## Sanity Checklist

- [x] **R1 (silent continue):** No new error paths — only changing what signal is read
- [x] **R2 (determinism):** No map iteration order changes — same loop structure
- [x] **R4 (construction sites):** No struct fields added — only changing method calls
- [x] **R8 (exported mutable maps):** No new maps
- [x] **R13 (behavioral contracts):** Tests assert observable scores, not internal implementation
- [x] **INV-1 (request conservation):** Scoring is stateless query — no effect on request lifecycle
- [x] **INV-7 (signal freshness):** Comment updated to reflect `QueueDepth` only (no longer includes synchronous `InFlightRequests` compensation)
- [x] No new CLI flags, no new interfaces, no new types
- [x] `EffectiveLoad()` preserved for other callers (load-balance, LeastLoaded, admission, counterfactual)
