# Iteration 2: Candidate Strategies

## Critical Findings from Iteration 1

- P1: Min-max normalization makes weights irrelevant at N=2
- P2: Need magnitude-aware decisions, not normalized scoring
- **P3 (NEW): Cache-aware routing is COUNTERPRODUCTIVE** — even with oracle cache (0s delay), routing to cached instance creates load imbalance that costs more than the cache hit saves. Pure queue-depth routing beats ppc:5,qd:1 by 2-8% on E2E P99.

This means the problem space is fundamentally different than assumed. The frontier isn't "better cache utilization" — it's "smarter load distribution under stale signals."

## Revised Problem Understanding

With 5s stale queue/batch signals and only InFlightRequests as synchronous:
- Multiple routing decisions happen between refreshes
- Each decision increments InFlightRequests synchronously, preventing pile-on within one refresh cycle
- But the stale QueueDepth/BatchSize cause systematic bias: the instance that was less loaded 5s ago keeps getting routed to even if it's now more loaded

The winning algorithm must: **handle stale signal better than static weighted scoring.**

---

## Candidate D: Power-of-Two-Choices with Stale-Signal Awareness

**Mechanism:** Instead of scoring all instances, randomly sample 2 candidates, then pick the one with lower EffectiveLoad. This is the Po2C algorithm which is theoretically optimal for load balancing under limited information.

**Self-critique:** With only 2 instances, Po2C is identical to least-loaded. Only helps at N>2. Does not use cache signal at all.

**Rejected** for N=2.

---

## Candidate E: InFlightRequests-Dominant Load Balancing

**Mechanism:** Weight the synchronous signal (InFlightRequests) much more heavily than the stale signals (QueueDepth, BatchSize). The stale signals provide a baseline, but the fresh signal drives the decision.

**Algorithm:**
```
for each instance i:
    freshLoad_i = InFlightRequests_i  // synchronous, always accurate
    staleLoad_i = QueueDepth_i + BatchSize_i  // 5s stale
    
    // Weighted: fresh signal 3x more important than stale signal
    effectiveScore_i = 3*freshLoad_i + staleLoad_i

select instance with MINIMUM effectiveScore
```

**Self-critique:** 
- InFlightRequests already included in EffectiveLoad(). This just changes the relative weighting.
- The weight "3" is another tunable. But it has a clear rationale: InFlightRequests is always accurate, so it should dominate.
- With 2 instances and min-max, this is STILL equivalent to regular load balancing (binary ranking). Need raw scores, not normalized.

**Key insight: use raw scores, not min-max normalized scores.**

---

## Candidate F: Raw-Score Load-Aware Routing with Cache Tiebreaker

**Mechanism:** Route to the instance with lowest *raw* EffectiveLoad (no normalization). When loads are equal (tie), use cache signal as tiebreaker. When loads differ by more than a threshold, ignore cache entirely.

**Algorithm:**
```
for each instance i:
    load_i = EffectiveLoad(i)  // raw, not normalized

// Find instance with minimum load
minLoad = min(load_i)
maxLoad = max(load_i)

if maxLoad - minLoad > loadThreshold:
    // Large load difference: route to least loaded (ignore cache)
    pick argmin(load_i)
else:
    // Loads are similar: use cache as tiebreaker
    for each instance i in {instances with load <= minLoad + loadThreshold}:
        cacheScore_i = cacheQueryFn(i, req.InputTokens)
    pick argmax(cacheScore_i) among eligible instances
```

**Self-critique:**
- loadThreshold is a tunable. But it can be derived from the system: e.g., 1 (meaning: loads within 1 request of each other are "tied")
- With N=2, this becomes: if loads differ by >1, pick least loaded. If loads equal or differ by 1, pick cached instance.
- This is simple and interpretable. But it doesn't use raw scores to break the min-max equivalence problem.

---

## Candidate G: Exponential-Decay Load Scoring

**Mechanism:** Instead of min-max normalization (binary with 2 instances), use exponential decay scoring that preserves absolute load information. `score_i = exp(-β * load_i)` naturally handles any N and preserves magnitude.

When combined with cache benefit, the composite score becomes:
```
score_i = exp(-β * load_i) * (1 + γ * cacheHits_i / requestBlocks)
```

Where:
- `exp(-β * load_i)` penalizes high load exponentially (with β controlling sensitivity)
- `(1 + γ * cacheRatio)` boosts instances with cache hits multiplicatively
- γ controls how much cache benefit is worth in load-equivalent terms

**Algorithm:**
```
β = 0.1  // load sensitivity (tunable)
γ = 0.0  // cache multiplier (start with 0 = load-only; tune later)

for each instance i:
    load_i = EffectiveLoad(i)
    score_i = exp(-β * load_i)
    
    // Optional cache boost (only if γ > 0)
    if cacheFn != nil && γ > 0:
        hits = cacheFn(i, req.InputTokens)
        totalBlocks = ceil(len(req.InputTokens) / blockSize)
        cacheRatio = hits / totalBlocks  // 0 to 1
        score_i *= (1 + γ * cacheRatio)
    
select argmax(score_i)
```

**Self-critique:**
- The exponential decay preserves absolute load differences (load=5 vs load=10 produces 0.61 vs 0.37 difference, not binary 1.0 vs 0.0)
- β=0.1 means: each additional request reduces the score by ~10%. At load=20, score ≈ 0.14. At load=0, score = 1.0.
- Starting with γ=0 tests pure load-balancing via exponential decay
- Can later add cache as multiplicative boost — multiplicative means cache only matters when loads are close (exp difference small)

---

## Selection

**Winner: Candidate G (Exponential-Decay Load Scoring)**

Rationale:
1. Solves P1 directly: exponential scoring preserves absolute magnitude even with N=2
2. Consistent with P3: starts with γ=0 (no cache), can add cache later only if proven useful
3. Multiplicative cache boost (future) means cache only wins when loads are similar — naturally prevents the load imbalance from P3
4. β controls stale-signal sensitivity: low β is conservative (smooths noise), high β is aggressive (amplifies small differences)
5. Clean ablation path: γ=0 is load-only, β=0 is random, β→∞ is deterministic least-loaded
