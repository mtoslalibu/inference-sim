# Problem: Evolving a Better Routing Strategy for llm-d

## What we're trying to do

The default llm-d routing strategy uses three scorers with fixed weights:

```
precise-prefix-cache: 3, queue-depth: 2, kv-utilization: 2
```

Each scorer scores instances on [0,1]. Scores are combined as a weighted sum, then argmax
picks the winner. This works, but the weights were hand-picked and the scorer set is fixed.

We want to use BLIS simulation to **discover a routing configuration that beats the baseline**
on E2E latency and TTFT for the same workloads.

## Scorers in scope

Six scorers with verified llm-d/GAIE parity:

| Scorer | What it does | Signal |
|---|---|---|
| `precise-prefix-cache` | Higher score for instances with more cached prefix blocks | KV cache block state |
| `no-hit-lru` | For cold requests, prefers least-recently-used instances. Warm requests get 0.5 (neutral). | Cache block state + LRU history |
| `queue-depth` | Min-max normalization: shorter queue = higher score | WaitingQueueSize |
| `kv-utilization` | Score = 1 - KVUtilization. More free cache = higher score | KVCacheUsagePercent |
| `active-requests` | Fewer in-flight requests = higher score (router-tracked, synchronous) | InFlightRequests |
| `load-aware` | Linear ramp: empty queue = 0.5, at threshold (128) = 0.0 | WaitingQueueSize |

## BLIS setup

```bash
./blis run \
  --model Qwen/Qwen3-14B \
  --hardware H100 \
  --tp 1 \
  --latency-model trained-physics \
  --num-instances 4 \
  --block-size-in-tokens 64 \
  --gpu-memory-utilization 0.95 \
  --total-kv-blocks 4719 \
  --max-num-scheduled-tokens 2048 \
  --max-num-running-reqs 256 \
  --max-model-len 40960 \
  --snapshot-refresh-interval 50000 \
  --cache-signal-delay 50000 \
  --routing-policy weighted \
  --routing-scorers "precise-prefix-cache:3,queue-depth:2,kv-utilization:2" \
  --workload-spec <workload.yaml>
```

Calibrated to real vLLM on 4x H100-SXM-80GB (Qwen3-14B, TP=1, block_size=64,
gpu_memory_utilization=0.95, total_kv_blocks=4719 pinned to real deployment value).

## Baseline

`--routing-scorers "precise-prefix-cache:3,queue-depth:2,kv-utilization:2"`

## Why fixed weights are the problem

The baseline uses fixed weights `3:2:2`. These can't be right at all load levels:

- **At low load**: all queues are short, all KV utilization is low. The queue-depth and
  kv-utilization scorers return ~1.0 for every instance (no differentiation). Their weight
  just dilutes the prefix-cache scorer's good decisions.
- **At high load**: prefix-cache scorer sends requests to cached-but-overloaded instances.
  The queue-depth scorer tries to push back, but with only weight 2 vs prefix's 3, it loses.
  Result: hot instances get hotter.

The right priority depends on cluster state, but the weights are baked in at deploy time.

## Direction: single adaptive composite scorer

Instead of three scorers with fixed weights, we build **one scorer** that reads all signals
and decides how to blend them based on current cluster load. The non-linearity lives inside
the scorer -- no framework changes needed.

Conceptual structure:

```
Pass 1: compute cluster-level load pressure from all endpoints
  avgQD = mean(queue_depths)
  avgKV = mean(kv_utilizations)
  loadPressure = f(avgQD, avgKV)    <-- shape to discover

Pass 2: score each endpoint
  prefixScore = cached_blocks / total_blocks     (per-instance)
  loadScore   = g(queue_depth, kv_utilization)   (per-instance)
  score       = blend(prefixScore, loadScore, loadPressure)
```

At low load pressure: prefix score dominates (maximize cache reuse).
At high load pressure: load score dominates (avoid overloaded instances).

The transition shape (`f`, `g`, `blend`) is what we evolve through iteration -- exactly like
the admission experiment evolved the polynomial power from linear to quintic.

### Why this works for sim2real

In llm-d, a scorer receives `[]Endpoint` with full per-endpoint metrics:
- `endpoint.GetMetrics().WaitingQueueSize`
- `endpoint.GetMetrics().KVCacheUsagePercent`
- `endpoint.Get(prefix.PrefixCacheMatchInfoKey)`

Computing cluster-level aggregates is a simple loop over the same data. No new signals,
no framework changes, no new APIs. Deploy as one plugin with one `pluginRef`.

### What we evolve (iteration by iteration)

1. The **load pressure function** `f` -- what signals drive it, what curve shape
2. The **blend function** -- how prefix vs load scores mix as pressure changes
3. The **per-instance load score** `g` -- which load signals matter, how to combine them

### Parameter-free goal

Like quintic admission (zero manual thresholds), the algorithm should self-tune. The curve
shape creates natural behavior at all load levels without per-workload knobs.

## Methodology: two-phase discovery

### Phase 1: Find where the baseline cracks

The baseline is competent. To beat it, we first need workloads that expose where fixed
weights structurally hurt. We probe four realistic scenarios:

| # | Scenario | Why 3:2:2 should struggle | Workload shape |
|---|---|---|---|
| W1 | **High prefix reuse + high load** | Prefix scorer concentrates traffic on cached-but-overloaded instances. Queue-depth (weight 2) can't override prefix (weight 3). | Shared system prompt (e.g., 512 common tokens), high QPS (1.2-1.5x capacity). |
| W2 | **Cold burst after warm steady-state** | Cache is warm from steady traffic. Burst of new-prefix requests arrives. Prefix scorer is useless (nothing cached for these), but still has weight 3 diluting load signals. | Steady-state phase (shared prefix, moderate load), then burst phase (unique prefixes, 2x rate). |
| W3 | **KV pressure** | When KV cache is nearly full, prefix scorer routes TO full instances (they have cached blocks). But those instances are about to preempt, destroying the very cache we routed for. | Large prompts (1.5-2K tokens), high rate, filling KV cache to >90%. |
| W4 | **Load asymmetry (bursty)** | Min-max normalization of queue-depth is relative. If 3 instances have QD=10 and one has QD=12, the spread normalizes to full [0,1]. Prefix cache then easily overrides this small signal. | Poisson arrivals with high variance, moderate load. |

For each workload, run the baseline and record E2E, TTFT, p99, and per-instance load
distribution. Workloads where the baseline shows poor tail latency or load imbalance
become the evolution workload suite for Phase 2.

### Phase 2: Evolve the routing strategy

Iterate on the composite scorer algorithm. Each iteration:

1. Implement the candidate blend function in BLIS
2. Run against all Phase 1 workloads where baseline cracked
3. Measure: E2E improvement, TTFT improvement, load distribution uniformity
4. Record findings in ledger, advance

Stop when improvement plateaus (diminishing returns between iterations, like iter8 vs iter9
in the admission experiment).

## Success criteria

- E2E latency (mean): >= 20% improvement over baseline
- TTFT (mean): >= 20% improvement over baseline
- Stretch goal: >= 35% on both
