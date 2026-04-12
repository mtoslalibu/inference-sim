# Admission Control: llm-d/GAIE vs BLIS — First Discussion

## Context

Following the sim2real_golden playbook (where we used BLIS to explore adaptive routing and produced a transfer bundle for llm-d), we now want to do the same for **admission control**. This document maps how admission works in production (llm-d + GAIE) vs how BLIS models it, and identifies gaps worth exploring.

---

## 1. How Admission Control Works in llm-d / GAIE

### The Short Version

GAIE has two admission controllers. The **legacy** one is simple: check saturation, and if the system is saturated AND the request is sheddable (priority < 0), reject it with HTTP 429. Non-sheddable requests always pass through. The **flow control** one is more sophisticated: it enqueues requests into a priority-aware queue, gates dispatch on saturation, and can evict in-flight sheddable requests if things get worse.

### Where It Happens

The entry point is `pkg/epp/requestcontrol/admission.go` in GAIE. The RequestControl Director calls `Admit()` on whichever controller is configured.

**Legacy path** (simpler):
1. Request arrives at EPP (Endpoint Picker)
2. `LegacyAdmissionController.Admit()` calls `rejectIfSheddableAndSaturated()`
3. If `saturation >= 1.0` AND `priority < 0` → reject with `ResourceExhausted` (HTTP 429)
4. Otherwise → proceed to scheduler (filter → score → pick)
5. If scheduler finds zero endpoints after filtering → HTTP 503

**Flow control path** (newer, more capable):
1. Request arrives at EPP
2. `FlowControlAdmissionController.Admit()` calls `FlowController.EnqueueAndWait()`
3. Request enters a priority queue with TTL deadline
4. Dispatch gated by saturation — if saturated, request waits
5. Possible outcomes: dispatched, rejected (queue full), evicted (TTL expired, client disconnected, or post-dispatch eviction of sheddable requests under pressure)

### Saturation Detection — Two Flavors

Both live in `pkg/epp/framework/plugins/flowcontrol/saturationdetector/`.

**Concurrency-based** (`concurrency/detector.go`):
- Signal: total in-flight requests across all pods
- Formula: `saturation = totalInflight / (numPods * maxConcurrency)`
- Defaults: `maxConcurrency=100`, `headroom=0.0`
- Also supports token-based mode (`totalTokens / (numPods * maxTokenConcurrency)`)

**Utilization-based** (`utilization/detector.go`):
- Signals: queue depth + KV cache utilization per pod
- Formula: per-pod score = `max(queueDepth/threshold, kvUtil/threshold)`, then average across pods
- Defaults: `queueDepthThreshold=5`, `kvCacheUtilThreshold=0.8`
- Treats stale metrics (>200ms old) as 100% saturated

### Post-Dispatch Eviction

Even after dispatch, GAIE can yank sheddable requests back:
- `pkg/epp/flowcontrol/eviction/plugin.go` tracks in-flight requests
- When saturation spikes, evicts lowest-priority, newest-dispatched requests first (minimize KV waste)
- Only sheddable (priority < 0) requests are eviction candidates

### Key Rejection Points

| Stage | Condition | Error |
|-------|-----------|-------|
| Admission (legacy) | Saturated + sheddable | 429 ResourceExhausted |
| Admission (flow control) | Queue at capacity | 429 ResourceExhausted |
| Queue wait | TTL expired | 503 ServiceUnavailable |
| Queue wait | Client disconnected | 503 ServiceUnavailable |
| Scheduler | All endpoints filtered out | 503 ServiceUnavailable |
| Post-dispatch | Eviction under pressure | Abort to model server |

### What llm-d Adds on Top

llm-d itself (`tmp/llm-d-inference-scheduler`) mostly defers to GAIE for admission. Its main contribution is the **disaggregated P/D profile handler** (`pkg/plugins/profile/disagg_profile_handler.go`) which orchestrates prefill/decode endpoint selection. If no decode workers are available after filtering, it returns "failed to find available decode workers" → 503. The scorers (active-request, load-aware, prefix-cache) influence routing but don't hard-reject — they just deprioritize loaded pods.

---

## 2. How BLIS Models Admission Control

### The Short Version

By default, BLIS runs with **AlwaysAdmit** — every request is admitted, no questions asked. But there are several admission mechanisms available that mirror production behavior.

### What Mechanisms Exist

**Admission Policies** (`sim/admission.go`):

| Policy | Behavior | Flag |
|--------|----------|------|
| **AlwaysAdmit** (default) | Admits everything | `--admission-policy always-admit` |
| **TokenBucket** | Rate-limits by input token count | `--admission-policy token-bucket` |
| **TierShedAdmission** | Sheds low-priority requests under overload | `--admission-policy tier-shed` |
| **RejectAll** | Rejects everything (test only) | `--admission-policy reject-all` |

**Gateway Queue + Flow Control** (`sim/cluster/gateway_queue.go`, `sim/saturation.go`):
- Priority-ordered queue between admission and routing
- Saturation-gated dispatch: holds requests when cluster is saturated
- Enabled with `--flow-control` flag (disabled by default)

**Saturation Detectors** (`sim/saturation.go`):

| Detector | Signal | Formula |
|----------|--------|---------|
| **NeverSaturated** (default) | None | Always 0.0 |
| **UtilizationDetector** | Queue depth + KV util | `avg(max(qd/thresh, kv/thresh))` per instance |
| **ConcurrencyDetector** | In-flight requests | `totalInflight / (instances * maxConcurrency)` |

### Default Pipeline (No Flags)

```
Request → AlwaysAdmit → Route to instance → Instance queue → Execute
```

No gateway queue, no saturation gating, no shedding. Requests flow straight through.

### Pipeline With Flow Control Enabled

```
Request → AdmissionPolicy → Gateway Queue → [saturation gate] → Route → Instance queue → Execute
```

Requests wait in the gateway queue until saturation drops below 1.0. If the queue is at capacity, the lowest-priority request is shed.

---

## 3. What's Identical Between BLIS and llm-d/GAIE

| Feature | BLIS | GAIE | Match? |
|---------|------|------|--------|
| Utilization-based saturation | `UtilizationDetector` with qd + kv thresholds | `utilization/detector.go` with same signals | Yes — same formula |
| Concurrency-based saturation | `ConcurrencyDetector` with maxConcurrency | `concurrency/detector.go` with maxConcurrency | Yes — same formula |
| Priority-ordered gateway queue | Priority heap with shed-on-overflow | Flow controller with priority bands | Structurally similar |
| Saturation-gated dispatch | Hold when `saturation >= 1.0` | Hold when `saturation >= usageLimit` | Close — BLIS uses hard 1.0 threshold |
| Sheddable concept | SLO tiers with priority 0-4 | `IsSheddable(priority < 0)` | Conceptually same, different numbering |
| Queue capacity shedding | Shed lowest-priority when full | Reject when at capacity | Similar |

---

## 4. Gaps — What BLIS Does NOT Model

### Gap 1: Post-Dispatch Eviction
GAIE can **evict in-flight sheddable requests** after they've been dispatched to a model server (abort the request to free capacity). BLIS has no equivalent — once a request is dispatched to an instance, it runs to completion or times out. This matters under sudden load spikes: production can reclaim GPU capacity mid-request, BLIS cannot.

### Gap 2: Headroom Parameter
GAIE's saturation detectors have a `headroom` parameter (0.0-1.0) that allows burst capacity above ideal limits. The filter uses `capacity * (1 + headroom)` while the saturation signal uses strict capacity. BLIS has no headroom concept — it's binary (saturated or not at threshold 1.0).

### Gap 3: Usage Limit Policy (Per-Priority Ceilings)
GAIE's flow controller has `UsageLimitPolicy` that computes per-priority-band saturation ceilings. This means different priority levels can be gated at different saturation thresholds (e.g., sheddable gated at 0.7, standard at 0.9, critical at 1.0). BLIS has a single global saturation threshold.

### Gap 4: TTL-Based Queue Eviction
GAIE requests have a TTL — if they sit in the queue too long, they're evicted with HTTP 503. BLIS's gateway queue has no timeout mechanism. Requests wait indefinitely (or until simulation horizon).

### Gap 5: Flow Isolation (Sharding)
GAIE's flow controller distributes requests across shards by flow ID + priority, with per-shard capacity tracking. This provides isolation between flows. BLIS has a single global gateway queue.

### Gap 6: Metrics Staleness Handling
GAIE's utilization detector treats metrics older than 200ms as "100% saturated" (safety fallback). BLIS's snapshot refresh interval models staleness for routing signals, but the saturation detector always uses whatever signals are available — no staleness-aware fallback.

### Gap 7: Token-Based Concurrency Mode
GAIE's concurrency detector supports token-based counting (estimated tokens in flight, not just request count). BLIS's concurrency detector only counts requests.

---

## 5. What Questions This Opens

1. **Post-dispatch eviction**: How much does it matter? Under what workloads does the ability to cancel in-flight sheddable requests significantly change outcomes? This feels like the biggest gap — can we quantify the benefit in simulation?

2. **Per-priority saturation thresholds**: Does tiered gating (critical at 1.0, sheddable at 0.7) outperform global gating? Could BLIS explore optimal threshold curves per priority class?

3. **TTL-based queue eviction**: What's the right TTL? Too short = unnecessary shedding. Too long = stale responses nobody wants. Can simulation find sweet spots per workload pattern?

4. **Headroom tuning**: What's the optimal headroom for burst absorption without cascading overload? This is a knob production teams struggle with — simulation could map the Pareto frontier.

5. **Are the current BLIS mechanisms sufficient for the sim2real transfer story?** The gateway queue + utilization detector already match the core GAIE flow. The gaps are refinements. Should we close them before exploring new admission strategies, or explore first and close gaps as needed?

---

## 6. Starting Point for Experiments

The sim2real_golden playbook worked because we had:
- Clear failure modes (7 workload patterns)
- Drop-in router replacements
- Clean comparison harness (compare.sh + analyze.py)
- Transfer documentation

For admission evolution, we'd need:
- **Failure modes**: Workloads where admission matters (overload, mixed-priority, burst + recovery)
- **Drop-in admission replacements**: New admission policies in `sim/admission.go`
- **Metrics**: Beyond latency — shed rate, fairness (Jain index across tenants), goodput (completed requests / total), priority inversion rate
- **Transfer path**: Map BLIS admission innovations back to GAIE's admission controller or flow controller plugin system
