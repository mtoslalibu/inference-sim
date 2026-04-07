# Discovered Adaptive Routing Algorithms

Both algorithms use the same three-regime detection framework. They differ in which load scorers they use. All scorers are GIE/llm-d parity.

## Baseline (for reference)

**Policy:** `weighted` with `ppc:2, qd:1, kvu:1`

Static fixed weights. llm-d production default. After PR #960, queue-depth reads QueueDepth only (5s stale, no synchronous feedback).

**Limitation 1 — Scorer conflict:** When a prefix is cached on one instance, `ppc` says "route there" but `kvu` says "route away" (cached instance has higher KV utilization). These fight each other with static weights.

**Limitation 2 — Stale equalization:** Under 5s snapshot refresh, all instances report identical QueueDepth within a window. Min-max normalization gives everyone score 1.0, eliminating load differentiation.

---

## Adaptive-v2 (3-scorer, zero-regression)

**Policy:** `adaptive-v2`

**Scorers (3):**
- `precise-prefix-cache` (ppc) — which instance has this request's prefix cached? Periodic signal via cache-signal-delay.
- `load-aware` (la) — llm-d #958. Threshold-capped linear scoring on QueueDepth. Score = 0.5 when idle, score = 0 when queue >= 128. Hard overload protection that works even when all instances report identical stale QueueDepth.
- `kv-utilization` (kvu) — 1 - KVUtilization. Periodic signal.

**Regime detection (per-request):**

| Regime | Condition | Weights | Rationale |
|--------|-----------|---------|-----------|
| Cache-affinity | ppc spread > 0.1 | ppc:4, la:1 | Route to cached instance; kvu deliberately excluded to avoid fighting ppc |
| Memory-aware | spread <= 0.1, avg KV util > 0.7 | la:1, kvu:1 | No prefix to exploit, KV tight — protect against exhaustion |
| Load-balance | spread <= 0.1, avg KV util <= 0.7 | la:1 | No prefix, memory spacious — pure load distribution |

**Key insight:** Drop kvu when cache matters. The static baseline can't do this.

**Results vs baseline:**
- Wins on **6/7 workloads**, never regresses more than -8% on any single metric
- TTFT: +52% FM-1, +43% FM-2a, +95% FM-6
- E2E: +39% P99 FM-3, +17% mean FM-2b

**Tradeoff:** Conservative. No synchronous load signal means TTFT wins are limited by stale snapshot equalization within each 5s window.

---

## Adaptive-Golden (5-scorer, synchronous signal)

**Policy:** `adaptive-golden`

**Scorers (5):**
- `precise-prefix-cache` (ppc) — same as v2
- `load-aware` (la) — same as v2 (threshold-capped QueueDepth)
- `active-requests` (ar) — llm-d #957. Synchronous InFlightRequests counter. Formula: idle = 1.0, busy = (maxCount - count) / maxCount. **Zero staleness** — updates instantly at gateway on every dispatch/completion. This is the only signal that differentiates instances within a snapshot refresh window.
- `running-requests` (rr) — GIE #956. Min-max on BatchSize (in-batch request count). Captures execution-phase load that queue depth misses.
- `kv-utilization` (kvu) — same as v2

**Regime detection (per-request):**

| Regime | Condition | Weights | Rationale |
|--------|-----------|---------|-----------|
| Cache-affinity | ppc spread > 0.1 | ppc:4, ar:1, la:1 | Strong prefix routing with synchronous + periodic load guards |
| Memory-aware | spread <= 0.1, avg KV util > 0.7 | ar:2, la:1, rr:1, kvu:1 | Multi-signal load balancing with KV protection |
| Load-balance | spread <= 0.1, avg KV util <= 0.7 | ar:2, la:1, rr:1 | All load signals, synchronous ar weighted highest |

**Key insight:** Same regime detection as v2, plus synchronous `ar` signal that provides real-time load feedback between stale snapshot refreshes. `ar` weight 2x in load regimes because it's the freshest signal.

**Results vs baseline:**
- Wins on **5/7 workloads**
- TTFT: +85% FM-1, +79% FM-2a, +95% FM-6 (bigger than v2 due to synchronous signal)
- E2E: +42% P99 FM-3, +31% P99 FM-2b

**Results vs v2:**
- FM-1 TTFT: +69% mean, +65% P99 (synchronous signal eliminates pile-on within snapshot windows)
- FM-2a TTFT: +26% mean, +63% P99
- FM-5: **regresses** -21% E2E mean, -30% TTFT mean (ar in cache-affinity regime competes with ppc for short-output requests)

**Tradeoff:** Higher ceiling than v2 on TTFT-sensitive workloads, but FM-5 short-output regression. The synchronous signal is most valuable when many requests arrive within a single 5s snapshot window.

---

## Which to use?

- **v2** if you want zero regressions and simplicity (3 scorers, all periodic)
- **Golden** if TTFT is the priority metric and FM-5-style short-output classification is not the dominant workload pattern
