# Iteration 1: Candidate Strategies

## Context

We need an algorithm that replaces the static `Σ clamp(s_i) × w_i` block in `WeightedScoring.Route()`. 

**Key observations from baselines:**
1. Prefix cache awareness is critical (GLIA loses 45% on P99 without it)
2. Under load, both baselines converge — the routing signal quality at saturation is the frontier
3. GLIA's TTFT is catastrophic — its KV projection causes pile-on
4. The kv-utilization signal (3:2:2's third component) is not demonstrably helpful
5. Signals are stale: snapshots at 5s, prefix cache at 2s. Synchronous signal = InFlightRequests only.

**Prior principles:** RP-1 (orthogonal signals), RP-6 (kv-util counterproductive), S6 (scheduling zero-sum).

---

## Candidate A: Regime-Adaptive Weighted Scoring

**Mechanism:** Detect the current operating regime (low-load vs high-load) from available signals, and switch weight profiles accordingly. Under low load, emphasize cache affinity. Under high load, emphasize load balance.

**Algorithm:**
```
loadPressure = mean(EffectiveLoad across instances) / threshold
if loadPressure < 0.5:
    weights = {precise-prefix-cache: 0.7, queue-depth: 0.3}
elif loadPressure < 1.0:
    weights = {precise-prefix-cache: 0.5, queue-depth: 0.5}
else:
    weights = {precise-prefix-cache: 0.3, queue-depth: 0.7}
apply standard weighted scoring with dynamic weights
```

**Self-critique:**
- The `threshold` parameter is ad-hoc — what counts as "high load"?
- Step function creates discontinuities — requests near the boundary oscillate
- The load signal is stale (5s), so regime detection is always delayed
- Only 2 signals — ignores potentially useful information from FreeKVBlocks, CacheHitRate
- "Low load" regime might never trigger if rate is always high

---

## Candidate B: Rank-Based Score Fusion with Staleness Discounting

**Mechanism:** Instead of weighted averages of raw scores, use **rank-based fusion**. Each scorer ranks instances from best to worst. The composite score is a weighted sum of ranks. Additionally, discount stale signals proportionally to their age relative to the decision rate.

**Algorithm:**
```
for each scorer:
    compute raw scores → rank instances (best=N, worst=1)
scorers used: precise-prefix-cache, queue-depth (from EffectiveLoad)
composite_rank[instance] = 0.6 * cache_rank + 0.4 * load_rank
select argmax(composite_rank), random tie-break
```

**Self-critique:**
- Rank fusion loses magnitude information — a 10% load difference and a 100% load difference both become rank 1 vs rank 2
- With only 2 instances, ranks are binary (1 or 2), making this equivalent to: "if both scorers agree, pick that one; if they disagree, pick the cache winner" — very simplistic
- Staleness discounting adds complexity but the discount factor is another tunable
- Doesn't really solve the "automatic" problem — still fixed weight ratio (0.6:0.4)

---

## Candidate C: Marginal-Gain Scoring

**Mechanism:** Instead of scoring absolute instance state, score the **marginal benefit** of sending a request to each instance. The marginal benefit combines: (1) cache savings — how many prefill tokens are saved by cache hits on this instance vs the worst instance, and (2) load cost — the projected queuing delay based on effective load.

**Algorithm:**
```
for each instance i:
    // Cache benefit: tokens saved by cache hit (in time units)
    cachedBlocks_i = cacheQueryFn(instance_i, req.InputTokens)
    cachedTokens_i = cachedBlocks_i * blockSize
    prefillSaved_i = cachedTokens_i  // proportional to prefill time saved

    // Load cost: projected wait time (proportional to queue position)
    loadCost_i = EffectiveLoad(i)  // synchronous component via InFlightRequests

    // Marginal score: benefit minus cost
    // Normalize both to [0,1] via min-max across instances
    normCache_i = minmax(prefillSaved_i)
    normLoad_i = minmax(loadCost_i)  // inverted: lower load = higher score

    score_i = normCache_i - α * normLoad_i

select argmax(score_i)
```

Where `α` controls the cost-benefit tradeoff. The key insight: this naturally handles the regime adaptation problem:
- When one instance has much more cache → cache term dominates (large spread) → route there
- When cache is equal across instances → cache term cancels out → pure load balance
- When load is very asymmetric → load cost dominates → avoid overloaded instance
- `α` is a single tunable parameter, not per-scorer weights

**Self-critique:**
- Still has one tunable parameter (α). But it has a clear physical interpretation: "how many tokens of cache savings justify sending to an instance with 1 more request in queue"
- Min-max normalization with 2 instances reduces to: best gets 1.0, worst gets 0.0. With >2 instances, it spreads properly.
- The cache signal is 2s stale and the queue/batch signals are 5s stale. Only InFlightRequests is fresh. This means the cache savings estimate is slightly delayed, but since caches are relatively stable, 2s delay is tolerable.
- Doesn't use KV-utilization (consistent with RP-6)

---

## Selection

**Winner: Candidate C (Marginal-Gain Scoring)**

Rationale:
1. Single tunable parameter with physical interpretation vs multiple ad-hoc thresholds
2. Naturally regime-adaptive without explicit regime detection
3. Directly models the cost-benefit tradeoff of routing decisions
4. Consistent with prior principles (RP-1: orthogonal signals, RP-6: no kv-util)
5. The "cache benefit vs load cost" framing makes ablation clean: remove cache → pure load balance, remove load → pure cache affinity

Candidate A was rejected for: ad-hoc regime thresholds, step function discontinuities, delayed regime detection on stale signals.
Candidate B was rejected for: loss of magnitude information, degenerates to trivial binary ranking with 2 instances.
