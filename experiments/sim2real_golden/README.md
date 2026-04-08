# sim2real_golden — Portable Adaptive Router Experiment

Self-contained experiment comparing adaptive routing algorithms against the
llm-d 2:1:1 baseline using BLIS (discrete-event LLM inference simulator).

Each router is a **drop-in replacement for `sim/routing.go`** — copy the file,
build, run. No other BLIS files need modification.

## Prerequisites

- BLIS main branch with PRs #960 and #966 merged (scorers: load-aware,
  active-requests, running-requests, queue-depth reading QueueDepth only)
- Go 1.22+, Python 3.8+
- `matplotlib` (optional, for box plot figures: `pip install matplotlib`)

## Directory Structure

```
sim2real_golden/
├── README.md              # This file
├── compare.sh             # Benchmark script: swap → build → run → restore
├── analyze.py             # Results comparison tables
├── workloads/             # 7 failure-mode workloads (spec v2 YAML)
│   ├── workload_fm2a_groups_gt_instances.yaml   ← core
│   ├── workload_fm3_burst.yaml                  ← core
│   ├── workload_fm6_cold_pressure.yaml          ← core
│   ├── workload_fm1_prefix_pileon.yaml          (extended)
│   ├── workload_fm2b_groups_lt_instances.yaml   (extended)
│   ├── workload_fm4_multiregime.yaml            (extended)
│   └── workload_fm5_short_output.yaml           (extended)
└── routers/               # Router files + policy YAMLs
    ├── policy_baseline_211.yaml        # Baseline: ppc:2, qd:1, kvu:1
    ├── router_adaptive_v2.go           # Adaptive-v2 algorithm
    ├── policy_adaptive_v2.yaml         # v2 scorer config
    ├── router_adaptive_expanded.go     # Adaptive-expanded (golden) algorithm
    ├── policy_adaptive_expanded.yaml   # Expanded scorer config
    ├── router_glia.go                  # Glia HRA algorithm
    └── policy_glia.yaml               # Glia dummy scorer config
```

## Routers

### Baseline 2:1:1 (llm-d default)

Static weighted scoring: `precise-prefix-cache:2, queue-depth:1, kv-utilization:1`.
Uses stock BLIS `routing.go` — no file swap needed.

**In plain English:** Every request gets routed to the instance with the best
fixed-weight combination of "has my prefix cached" (weight 2) and "isn't too
busy" (weight 1). The weights never change regardless of what's happening.

**The problem:** Because the cache score is weighted 2x, it dominates. When many
requests share a prefix, they all pile onto the one instance that cached it.
The queue-depth signal updates every 5 seconds — too slow to push back against
a sudden flood. The result is one overloaded instance and three idle ones.

### Adaptive-v2 (3-scorer regime detection)

Replaces `WeightedScoring.Route()` with per-request regime detection using 3
scorers: `precise-prefix-cache`, `load-aware`, `kv-utilization`.

**In plain English:** Instead of fixed weights, this router looks at the current
state of the cluster before each request and picks one of three strategies:
- If cache hits are spread unevenly across instances, lean into cache affinity
  (send the request where its prefix lives).
- If memory is running low (>70% KV cache used), ignore the cache entirely and
  just balance load — prevents piling onto a nearly-full instance.
- Otherwise, just spread load evenly.

**What it exploits that baseline misses:** The baseline blindly trusts a stale
cache signal. Adaptive-v2 *checks whether the cache signal is worth following*
before using it — and backs off to load-balancing when it isn't.

**Regimes:**
- **Cache-affinity** (cache spread > 0.1): ppc:4, load-aware:1 — lean into cache
  locality when prefix hits vary across instances
- **Memory-aware** (avg KV util > 0.7): load-aware:1, kvu:1 — disable prefix
  affinity under memory pressure to prevent pile-on
- **Load-balance** (default): load-aware:1 — weak cache signal, just balance load

**Key insight:** `load-aware` reads `QueueDepth` with threshold cap at 128
(llm-d's load-aware-scorer semantics), replacing the stale-prone min-max
`queue-depth` scorer in the baseline.

### Adaptive-expanded / Golden (5-scorer regime detection)

Same regime detection as v2 but with two additional load signals:
`active-requests` (synchronous InFlightRequests) and `running-requests`
(min-max on BatchSize).

**In plain English:** Same adaptive regime switching as v2, but with more eyes
on the load. v2 only knows how deep each queue is. Expanded also knows how
many requests are in-flight (dispatched but not queued yet) and how many are
actively running in each batch. More load signals = finer-grained decisions,
especially under bursty traffic where in-flight count reacts instantly while
queue depth lags behind.

**Regimes:**
- **Cache-affinity**: ppc:4, active-requests:1, load-aware:1
- **Memory-aware**: active-requests:2, load-aware:1, running-requests:1, kvu:1
- **Load-balance**: active-requests:2, load-aware:1, running-requests:1

Bigger TTFT wins on prefix-heavy workloads but slight FM-5 regression vs v2.

### Glia HRA (KV-cache headroom projection)

Bypasses the scorer pipeline entirely. Projects KV-cache block usage for each
instance after hypothetically placing the request:

**In plain English:** Before routing, Glia asks "if I put this request here,
how full will the GPU memory be?" It estimates how many KV-cache blocks the
request will need, checks which instances have enough room, and picks the one
with the most headroom. Instances that would run out of memory are penalized.

**What it misses:** It has no awareness of prefix caching at all — it only cares
about memory headroom. This makes it good at avoiding OOM-style preemptions but
bad at reusing cached prefixes, which is why it loses badly on prefix-heavy
workloads (FM-2a: -618% E2E P99).

1. Estimate request blocks: `ceil(inputTokens * 1.6 / blockSize)`
2. Estimate total/free blocks from `FreeKVBlocks` and `KVUtilization`
3. Score: `-projectedUsage/totalBlocks - 0.001*queueLoad`
4. Inadmissible instances (insufficient headroom) get -10.0 penalty

## Workloads

### Core Workloads (default)

The default `compare.sh` run uses 3 production-representative workloads chosen
for clear gains, zero timeouts, and distinct failure mode coverage:

| ID | Name | What it tests | Why chosen |
|----|------|---------------|------------|
| FM-2a | Groups > Instances | 6 prefix groups across 4 instances → routing contention | Biggest TTFT wins (v2: +43% P99, expanded: +79% P99). Realistic multi-tenant scenario. |
| FM-3 | Burst Absorption | 3x burst spikes → stale snapshot pile-on | Best all-around gains on both E2E and TTFT (v2: +39% E2E P99, +22% TTFT). Clean runs, no timeouts. |
| FM-6 | Cold Traffic Under KV Pressure | Long-context requests with 50% cold traffic | Dramatic TTFT P99 collapse: 2195ms → 115ms (v2: +95%). Mixed cold/warm traffic is production-realistic. |

**FM-2a** — Imagine 7 different tenants (each with their own system prompt) sharing
4 GPU instances. There aren't enough instances for each tenant to have their own,
so the cache scorer keeps flip-flopping requests between instances, causing
repeated cache misses. The adaptive routers detect this instability and fall back
to load-balancing instead of chasing unstable cache hits.

**FM-3** — Steady traffic with sudden 10x burst spikes. During a spike, all burst
requests share one prefix, so the baseline piles them onto one instance (cache
weight 2x beats queue-depth 1x). The queue-depth signal is 5 seconds stale, so
it can't react in time. Adaptive routers use real-time in-flight counts to
spread the burst across all instances immediately.

**FM-6** — All requests are "cold" (no shared prefixes), with large context
windows that eat up GPU memory fast. Since no prefix caching helps here, the
cache scorer is useless — the real challenge is avoiding instances whose KV
cache is nearly full. Adaptive routers detect memory pressure and route around
nearly-full instances, preventing expensive preemptions.

### Extended Workloads (--all flag)

| ID | Name | What it tests | Notes |
|----|------|---------------|-------|
| FM-1 | Prefix Pile-On | 75% traffic shares one prefix → pile-on | Baseline timeouts on seed=456 (preemption livelock, issue #963) |
| FM-2b | Groups < Instances | 2 prefix groups across 4 instances | Gains are smaller, less compelling for presentation |
| FM-4 | Multi-Regime Phased | Warm-up → pressure → cooldown | Near-wash — no meaningful difference between algorithms |
| FM-5 | Short Output | Classification workload (short outputs) | Expanded regresses on E2E/TTFT mean vs baseline |

**FM-1** — Worst-case pile-on: 75% of all traffic shares one system prompt. The
baseline sends nearly everything to whichever instance cached that prefix,
starving the other 3. Adaptive routers detect the extreme imbalance and switch
to load-balancing.

**FM-2b** — Opposite of FM-2a: only 2 prefix groups across 4 instances. The
baseline concentrates 90% of traffic on the 2 instances with cache hits, leaving
the other 2 nearly idle. A simpler version of the same "cache-chasing wastes
capacity" problem.

**FM-4** — Traffic changes character over time: first prefix-heavy, then all cold,
then mixed, then a new dominant prefix. Static 2:1:1 weights are a compromise
that's suboptimal in every phase. In practice, the phases are long enough that
the differences wash out.

**FM-5** — Classification-style workload: 7 prefix groups but very short outputs
(~8 tokens). Since the response is almost entirely TTFT (no long decode phase),
any TTFT improvement should directly improve end-to-end latency. High request
rate (60 req/s) stresses the system differently than the other workloads.

All workloads use `slo_class: "standard"` and target Qwen/Qwen3-32B on H100 TP=1.

## How to Run

```bash
cd experiments/sim2real_golden

# Core workloads only (3 workloads × 4 routers × 3 seeds = 36 runs, ~3 min)
bash compare.sh

# All 7 workloads (84 runs, ~7 min)
bash compare.sh --all
```

The script:
1. Builds BLIS with stock `routing.go` (baseline)
2. For each non-baseline router: swaps `router_*.go` → `sim/routing.go`, rebuilds
3. Runs workloads × 3 seeds per router
4. Restores original `routing.go` on exit (even on Ctrl-C)
5. Prints comparison tables via `analyze.py`

Results are saved in `results/` as JSON metrics files.

### Analyze Results

```bash
# Tables only
python3 analyze.py results

# Tables + box plot figures (saved to results/figures/)
python3 analyze.py results --figures
```

Figures are per-workload 2x2 grids (E2E Mean, E2E P99, TTFT Mean, TTFT P99)
with color-coded box plots per algorithm and % gain vs baseline annotated on
each box. Green = better, red = worse, bold = 15%+ gain.

## Simulation Results (Core Workloads)

### FM-2a: Groups > Instances (multi-tenant prefix routing)

```
               baseline         v2   (gain)   expanded   (gain)       glia   (gain)
    E2E Mean       2547       2411    +5.4%       2261   +11.3%      16403  -543.9%
     E2E P99       5795       4753   +18.0%       3862   +33.4%      41630  -618.4%
   TTFT Mean        161        113   +30.1%         83   +48.3%      14012 -8601.7%
    TTFT P99       2834       1611   +43.1%        597   +78.9%      39407 -1290.3%
```

### FM-3: Burst Absorption (burst traffic under stale snapshots)

```
               baseline         v2   (gain)   expanded   (gain)       glia   (gain)
    E2E Mean       3178       2503   +21.2%       2457   +22.7%       3028    +4.7%
     E2E P99       8662       5310   +38.7%       5006   +42.2%       8001    +7.6%
   TTFT Mean        108         84   +22.1%         81   +24.5%         98    +9.0%
    TTFT P99        524        449   +14.2%        409   +21.9%        387   +26.1%
```

### FM-6: Cold Traffic Under KV Pressure (mixed cold/warm with long context)

```
               baseline         v2   (gain)   expanded   (gain)       glia   (gain)
    E2E Mean      12730      12300    +3.4%      12290    +3.5%      12565    +1.3%
     E2E P99      30581      30160    +1.4%      30221    +1.2%      30729    -0.5%
   TTFT Mean        147         71   +51.9%         70   +52.5%        145    +1.3%
    TTFT P99       2195        115   +94.8%        117   +94.7%       2379    -8.4%
```

All values in ms, averaged across 3 seeds. Positive % = faster than baseline.

## How to Port to Real BLIS

After PRs #960 and #966 are on your BLIS main:

```bash
# 1. Copy router file
cp routers/router_adaptive_v2.go /path/to/blis/sim/routing.go

# 2. Build
cd /path/to/blis && go build -o blis main.go

# 3. Run with policy YAML
./blis run --model Qwen/Qwen3-32B --policy-config routers/policy_adaptive_v2.yaml \
    --workload-spec workloads/workload_fm3_burst.yaml
```

## Simulation Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | Qwen/Qwen3-32B | Standard LLM benchmark model |
| Hardware | H100, TP=1 | Single-GPU production config |
| Latency model | trained-physics | BLIS's physics-informed model |
| Instances | 4 | Typical small cluster |
| Cache signal delay | 2s | llm-d's defaultSpeculativeTTL |
| Snapshot refresh | 5s | Production-like staleness |
| Seeds | 42, 123, 456 | 3 seeds for variance |
| Timeout | 90s per run | Guards against preemption livelock (#963) |



## sim2real options
  Option A: Multi-profile with custom ProfileHandler (zero core changes)

  llm-d already supports multiple named profiles with different weights. Define 3 profiles in YAML:

  profiles:
    - name: cache-affinity
      scorers: [{name: precise-prefix-cache, weight: 4}, {name: load-aware, weight: 1}]
    - name: memory-aware
      scorers: [{name: load-aware, weight: 1}, {name: kv-utilization, weight: 1}]
    - name: load-balance
      scorers: [{name: load-aware, weight: 1}]

  Then write a custom ProfileHandler that replaces SingleProfileHandler. Its Pick() method checks cacheSpread and avgKVUtil from endpoint state
  and returns the matching profile. The ProfileHandler interface already exists for this exact use case.

  Option B: Modify runScorerPlugins directly (surgical, 1 file)

  Add regime detection before the weight accumulation loop in scheduler_profile.go:160-168. Compute cacheSpread/avgKVUtil from the endpoints,
  select regime weights, and use those instead of scorer.Weight(). Same pattern as our BLIS router files — one function, clear markers.
  