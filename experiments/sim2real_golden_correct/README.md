# sim2real_golden_correct: Adaptive Regime-Detection Router

## Quick Start

```bash
cd experiments/sim2real_golden_correct
bash compare.sh                          # runs all 36 experiments (~5 min)
python3 analyze.py results/              # prints comparison tables
python3 analyze.py results/ --figures    # tables + box plots (needs matplotlib)
```

## Results Summary

Cluster: 4x H100 instances, Qwen3-32B, TP=1, 50ms metric refresh, 2s cache signal delay.

| Workload | Adaptive vs 2:1:1 | Adaptive vs 3:2:2 | Glia vs 2:1:1 |
|----------|-------------------|--------------------|----------------|
| FM-3: Burst | +11% E2E mean | +4% E2E mean | +3% E2E mean |
| FM-5: Short Output | **+40.6% E2E mean** | **+50.6% E2E mean** | -1390% (catastrophic) |
| FM-8: High Rate | **+42.6% E2E mean** | **+43.9% E2E mean** | -736% (catastrophic) |

Positive % = our algorithm is better (lower latency).

---

## The Algorithm: Adaptive Regime Detection

The core idea: **don't use the same scorer weights for every request.** Instead, detect which "regime" the cluster is in right now and pick weights that match.

### How It Works (30 lines of logic)

Every time a request arrives, the router:

1. **Runs all 5 scorers** on every instance (same scorers the baseline uses, plus extras)
2. **Checks two signals** to detect the current regime:
   - `cacheSpread` = max(prefix-cache-score) - min(prefix-cache-score) across instances
   - `avgKVUtil` = average KV cache utilization across instances
3. **Picks regime-specific weights:**

```
if cacheSpread > 0.1:
    REGIME 1 — Cache-affinity
    Weights: prefix-cache:4, load-aware:1, active-requests:1, running-requests:0, kv-util:0
    Why: One instance has significantly more cached prefix data. Route there,
         with a small load-balancing nudge to avoid total pile-on.

elif avgKVUtil > 0.7:
    REGIME 2 — Memory-aware
    Weights: prefix-cache:0, load-aware:1, active-requests:2, running-requests:1, kv-util:1
    Why: KV caches are filling up. Forget about prefix affinity, spread load
         evenly and factor in memory pressure.

else:
    REGIME 3 — Load-balance (default)
    Weights: prefix-cache:0, load-aware:1, active-requests:2, running-requests:1, kv-util:0
    Why: No strong cache signal, no memory pressure. Just balance load using
         three complementary signals.
```

4. **Computes weighted score** per instance, picks the highest.

### Why It Beats the Baseline

The llm-d default (2:1:1) uses static weights: `prefix-cache:2, queue-depth:1, kv-util:1`. This means prefix affinity ALWAYS gets 50% of the routing decision. When 7 prefix groups compete for 4 instances, the prefix scorer creates imbalanced routing (3-4 groups pile onto the same instance) and the queue-depth signal (only 25% weight) can't correct it fast enough.

The adaptive router detects that in these scenarios, cache spread is low (many groups, scores are similar), so it **disables prefix affinity entirely** and routes purely on load. This eliminates the pile-on.

### The 5 Scorers

| Scorer | What it measures | Signal type |
|--------|-----------------|-------------|
| `precise-prefix-cache` | How many prefix tokens are cached on each instance | Periodic (2s stale) |
| `load-aware` | `1/(1+QueueDepth+BatchSize+InFlightRequests)` | Mixed (InFlightRequests is synchronous) |
| `active-requests` | `1/(1+InFlightRequests)` — requests dispatched but not done | Synchronous |
| `running-requests` | `1/(1+BatchSize)` — requests currently in a batch | Periodic |
| `kv-utilization` | `1 - KVUtilization` — prefer instances with free KV memory | Periodic |

Key insight: `active-requests` and `load-aware` include `InFlightRequests`, which updates **synchronously** (no staleness). This gives the router real-time feedback even when periodic metrics are stale.

---

## The Workloads

### FM-3: Burst Absorption (comparable, +11% E2E)

**What it models:** A chatbot service where one popular system prompt gets bursty traffic.

- 1000 requests at 8 req/s across 4 instances
- 50% of traffic shares one prefix (16K tokens) and arrives in extreme bursts (gamma CV=8)
- 30% steady non-prefix traffic, 20% steady different-prefix traffic
- Output: medium length (~64 tokens)

**Why adaptive helps a little:** During bursts, the prefix scorer sends all bursty requests to the cached instance. The adaptive router keeps prefix affinity here (cacheSpread is high — one instance has the cache, others don't), but the extra load-aware signals help spread during the most extreme spikes.

**Real-world analogy:** Customer support chatbot with a long system prompt. Traffic spikes during business hours.

### FM-5: Short Output Classification (big win, +40-50% E2E)

**What it models:** A document classification or extraction service — many tenants, short answers.

- 2000 requests at 60 req/s across 4 instances
- 7 prefix groups (different document collections), each with 16K-token prefix
- Very short output: mean 8 tokens (yes/no, category label, extracted field)
- Smooth Poisson arrivals

**Why adaptive wins big:** With 7 groups across 4 instances, the prefix scorer gives similar scores to many instances (low cache spread). The baseline still gives prefix 50% weight, creating imbalanced routing. Adaptive detects low cache spread and switches to pure load-balance, spreading requests evenly. Short output means E2E is almost entirely TTFT, so TTFT improvements translate directly to E2E wins.

**Real-world analogy:** Multi-tenant RAG service doing document classification, entity extraction, or routing decisions. Each tenant has their own document collection prefix.

### FM-8: Short Output at Very High Rate (big win, +43% E2E)

**What it models:** Same as FM-5 but under higher load — stress test.

- 2000 requests at 80 req/s across 4 instances (20 req/s per instance average)
- Same 7 prefix groups, same short output
- Higher rate means more contention and faster queue buildup

**Why adaptive wins big:** Same mechanism as FM-5, amplified. At 80 req/s, when the baseline piles 3-4 groups onto one instance, that instance sees 40-60 req/s while others idle. The queue builds up fast. Adaptive's load-balance regime prevents this entirely.

**Real-world analogy:** High-throughput classification pipeline processing document batches.

---

## The Baselines

### Baseline 2:1:1 (llm-d production default)

```yaml
scorers:
  - precise-prefix-cache: 2    # 50% of routing weight
  - queue-depth: 1              # 25%
  - kv-utilization: 1           # 25%
```

Three scorers with static weights. Prefix cache always dominates. This works well when prefix groups map cleanly to instances (few groups, many instances). Breaks down when groups outnumber instances.

### Baseline 3:2:2 (heavier static weights)

```yaml
scorers:
  - precise-prefix-cache: 3    # 43% of routing weight
  - queue-depth: 2              # 29%
  - kv-utilization: 2           # 29%
```

Same three scorers, slightly less prefix-heavy. Actually **worse** than 2:1:1 on FM-5 by 20% — proving that static weight tuning alone doesn't solve the problem. The issue isn't the weight ratio; it's that prefix affinity shouldn't apply at all when cache spread is low.

### Glia HRA (KV headroom projection)

A completely different approach: no scorer pipeline. Instead, it projects how much KV cache each instance would use if the request were placed there, then picks the instance with the most headroom.

Works well for KV-pressure scenarios but **catastrophically fails on short-output workloads** (FM-5: 29882ms vs 1194ms adaptive). Reason: it ignores prefix cache affinity entirely AND doesn't have strong load-balancing signals, so under high request rates it makes poor placement decisions.

---

## Sim2Real: Running This on Real llm-d

### What Needs to Change for Real Deployment

The simulator uses the same scorer names and normalization as llm-d's GAIE (Gateway API Inference Extension). The algorithm translates to real llm-d with minimal changes.

### Approach: Custom SchedulerProfile (no GAIE fork needed)

GAIE's `SchedulerProfile` is an interface. llm-d can inject a custom implementation via `runner.WithSchedulerConfig()`. This means the adaptive logic lives entirely in llm-d — zero GAIE changes.

**1. Create one new file** in llm-d: `pkg/plugins/profile/adaptive_profile.go`

```go
type AdaptiveProfile struct {
    filters []Filter
    scorers []*WeightedScorer  // [0]=prefix-cache, [1]=load-aware, [2]=kv-util
    picker  Picker
}

func (p *AdaptiveProfile) Run(ctx context.Context, req *LLMRequest,
    cycleState *CycleState, endpoints []Endpoint) (*ProfileRunResult, error) {

    // 1. Run filters (same as baseline)
    eps := p.runFilters(ctx, cycleState, req, endpoints)

    // 2. Run all scorers, collect per-endpoint scores
    allScores := make([]map[Endpoint]float64, len(p.scorers))
    for i, s := range p.scorers {
        allScores[i] = s.Score(ctx, cycleState, req, eps)
    }

    // 3. Regime detection
    cacheSpread := maxScore(allScores[0]) - minScore(allScores[0])
    avgKVUtil := avgEndpointKVUtil(eps)

    var weights []float64
    switch {
    case cacheSpread > 0.1:
        weights = []float64{4, 1, 0}  // cache-affinity
    case avgKVUtil > 0.7:
        weights = []float64{0, 1, 1}  // memory-aware
    default:
        weights = []float64{0, 1, 0}  // load-balance
    }

    // 4. Weighted sum with regime weights
    // 5. Pick max-score endpoint
}
```

**2. Wire it in** `cmd/epp/main.go`:

```go
runner.NewRunner().
    WithSchedulerConfig(buildAdaptiveConfig()).
    Run(ctx)
```

No YAML config changes for scheduling. The profile is built in code.

### Mapping Simulation Scorers to Real llm-d Scorers

| Sim scorer | Real llm-d scorer | Notes |
|-----------|-------------------|-------|
| `precise-prefix-cache` | `precise-prefix-cache-scorer` | Same: queries instance KV cache hash maps |
| `load-aware` | `load-aware-scorer` | Same: `1/(1+EffectiveLoad)` |
| `active-requests` | Custom or use load-aware | InFlightRequests available via endpoint metrics |
| `running-requests` | Custom or use queue-depth | BatchSize available via endpoint metrics |
| `kv-utilization` | `kv-cache-utilization-scorer` | Same: `1 - KVCacheUsagePercent/100` |

For the simplest sim2real translation, use 3 scorers (prefix-cache, load-aware, kv-util) instead of 5. The simulation's `active-requests` and `running-requests` are subcomponents of `load-aware` — using `load-aware` alone captures most of the signal.

### Simplified 3-Scorer Version for Real Deployment

```
if cacheSpread > 0.1:
    prefix-cache:4, load-aware:1, kv-util:0    (cache-affinity)
elif avgKVUtil > 0.7:
    prefix-cache:0, load-aware:1, kv-util:1    (memory-aware)
else:
    prefix-cache:0, load-aware:1, kv-util:0    (load-balance)
```

This is the same algorithm with the 5 load scorers collapsed into `load-aware`. It should perform similarly because `load-aware` already includes queue depth, batch size, and in-flight requests in its `EffectiveLoad()` calculation.

### Real Workload Equivalents

| Sim workload | What to test on real cluster |
|-------------|------------------------------|
| FM-3 (burst) | Send bursty traffic to one prefix group. Use a load generator with gamma-distributed inter-arrival times |
| FM-5 (short output) | Multi-tenant RAG with 5-10 document collections, classification queries (short output). Rate: ~15 req/s/instance |
| FM-8 (high rate) | Same as FM-5 but push to ~20 req/s/instance |

### Expected Real-World Improvement

The simulation models 50ms metric refresh (matching GAIE's `RefreshMetricsInterval`) and 2s cache signal delay (matching llm-d's `defaultSpeculativeTTL`). These are production defaults.

Conservative estimate: **20-40% E2E improvement** on multi-tenant short-output workloads. The simulation shows 40-50% but real systems have additional variance (network jitter, GC pauses, vLLM scheduling differences) that may reduce the gap.

The burst workload (FM-3) improvement is smaller in simulation (+11%) and may be within noise in production. Focus sim2real validation on short-output multi-tenant patterns first.

---

## File Structure

```
sim2real_golden_correct/
  compare.sh                    # Main benchmark script (run this)
  analyze.py                    # Analysis + comparison tables
  routers/
    router_adaptive.go # The adaptive algorithm (replaces sim/routing.go)
    router_glia.go              # Glia HRA for comparison
    policy_baseline_211.yaml    # llm-d default: ppc:2, qd:1, kvu:1
    policy_baseline_322.yaml    # Heavier static: ppc:3, qd:2, kvu:2
    policy_adaptive.yaml # 5 scorers (weights ignored, regime overrides)
    policy_glia.yaml            # Dummy scorer (glia bypasses pipeline)
  workloads/
    workload_fm3_burst.yaml     # Burst absorption
    workload_fm5_short_output.yaml # Short output classification (big win)
    workload_fm8_short_output_highrate.yaml # High-rate variant (big win)
  results/                      # Auto-generated by compare.sh
```
