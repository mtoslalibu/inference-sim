---
description: Validate a sim2real translation bundle against its simulation bundle. Checks workloads, configs, signals, policies, and runtime health.
---

## User Input

```text
$ARGUMENTS
```

Parse user input if provided (e.g., `--sim <path> --real <path> --workload <name>`).

## Step 0: Auto-Detect and Confirm Paths

Before running ANY checks, **auto-detect** paths by scanning the working directory and context, then **confirm with the user** using `AskUserQuestion`.

### Auto-Detection Strategy

1. **Sim bundle**: Look for directories containing `README.md` + `algorithm/` + `workloads/` under `experiments/`. Common patterns: `sim2real_bundle*`, `sim2real_*bundle*`.
2. **Real bundle**: Look for directories containing `generated/` + (`baseline/` or `treatment/`) under `experiments/`. Common patterns: `bad_and_good*`, `*admission*`, `*deployment*`.
3. **BLIS codebase**: The current working directory (check for `sim/` dir and `go.mod`).
4. **GAIE codebase**: Look for `gateway-api-inference-extension` under `tmp/`, `../`, or nearby.
5. **llm-d scheduler**: Look for `llm-d-inference-scheduler` under `tmp/`, `../`, or nearby.
6. **Workloads**: List subdirectories under `<real>/baseline/` and `<real>/treatment/`.

Use `Glob` and `Bash` (ls) to find candidates. Then present findings to the user:

```
I found the following paths. Please confirm or correct:

Sim bundle:      <detected or "not found">
Real bundle:     <detected or "not found">
BLIS codebase:   <detected or "not found">
GAIE codebase:   <detected or "not found">
llm-d scheduler: <detected or "not found">
Workload(s):     <detected or "all">
```

Use `AskUserQuestion` to let them confirm or provide corrections. Do NOT proceed until confirmed.

## Goal

Perform a comprehensive parity check between a BLIS simulation bundle and its real llm-d deployment results. Produce a structured **evidence-based** report.

## Operating Constraints

**STRICTLY READ-ONLY**: Do **not** modify any files. Output a structured report.

## Evidence Requirements

For EVERY check, you MUST show:
1. **What you checked** — plain English, one sentence
2. **Expected** — the value from simulation (with source: which file, which field)
3. **Actual** — the value from real deployment (with source: which file, how computed)
4. **Verdict** — PASS / WARN / FAIL
5. **Why** — if WARN or FAIL, explain in plain English what this means and what to do about it

For numerical checks, always show a comparison table. For code/config checks, show the relevant snippets side by side.

## Reference Paths (resolved after Step 0)

- **Simulation codebase** (`BLIS`): confirmed by user
- **GAIE codebase** (`GAIE`): confirmed by user
- **llm-d scheduler** (`LLMD`): confirmed by user
- **Sim bundle** (`SIM`): `<sim>/README.md`, `<sim>/config.md`, `<sim>/algorithm/`, `<sim>/workloads/`, `<sim>/results/`
- **Real bundle** (`REAL`): `<real>/generated/` (Go plugins + YAML configs), `<real>/baseline/<workload>/` and `<real>/treatment/<workload>/` (each with `trace_header.yaml`, `trace_data.csv`, `server_logs/`, `epp_stream_done`)

## Checklist

Run ALL checks below.

---

### 1. WORKLOAD PARITY

For each workload in the real bundle, find the matching sim workload YAML in `<sim>/workloads/`.

**1a. Arrival Rate (QPS)**
- **Intended QPS**: compute from `trace_data.csv` `arrival_time_us` column: `num_requests / (max - min arrival)`. This is the rate the load generator *scheduled* requests at.
- **Actual QPS**: compute from `trace_data.csv` `send_time_us` column: `num_requests / (max - min send_time)`. This is the rate requests were *actually sent* to the server. If actual QPS << intended QPS, the load generator hit a concurrency ceiling and requests queued client-side.
- From sim: `aggregate_rate` in workload YAML
- Tolerance: intended QPS within 5% of sim spec. If actual QPS is more than 5% below intended QPS, emit WARN — this means the load generator couldn't keep up with the schedule (likely concurrency ceiling or server backpressure), so the server saw less load than intended and results are not comparable to simulation.
- Show: table with sim spec rate, intended QPS, actual QPS — all three columns so the reader can spot concurrency bottlenecks

**1b. SLO Class Distribution**
- From sim: each client's `rate_fraction`
- From real: count `slo_class` column in `trace_data.csv`, compute percentages
- Tolerance: within 2pp per class
- Show: table with each class — expected %, actual %, difference

**1c. Input Token Distribution**
- From sim: `input_distribution` params (mean, std_dev, min, max)
- From real: compute mean, stdev, min, max of `input_tokens` column
- Tolerance: mean within 5%, stdev within 10%
- Show: table with expected vs actual for mean, stdev, min, max

**1d. Output Token Distribution**
- From sim: `output_distribution` params
- From real: compute from `output_tokens` column
- Same tolerances as input
- Show: same table format as 1c

**1e. Prefix Tokens**
- From sim: check if any client has `prefix_group` or prefix config
- From real: check `prefix_length` column — should match (all zeros if no prefix in sim)
- Show: "All zero: yes/no" + if not zero, show distribution

**1f. Streaming**
- From sim: `streaming` field per client
- From real: `streaming` column in trace
- Show: expected vs actual streaming fraction

**1g. Burstiness / Arrival Pattern**
- From sim: `arrival.process` (poisson, constant, etc.)
- From real: compute coefficient of variation (CV) of inter-arrival times
  - Poisson: CV should be ~1.0 (0.8-1.2 acceptable)
  - Constant: CV should be ~0 (<0.1)
- Show: expected arrival process, measured CV, interpretation

---

### 2. CONFIG PARITY

**2a. vLLM Server Config**
Read `trace_header.yaml` AND the first vLLM server log from `server_logs/` to extract actual config.

Check these against `<sim>/config.md`:
- Model name (exact match)
- TP degree (`tensor_parallel_size`)
- GPU type (from log or header)
- `max_model_len`
- `gpu_memory_utilization`
- `enable_prefix_caching` (must match sim expectation)
- `max_num_seqs`
- `max_num_batched_tokens`
- `enable_chunked_prefill`

Show: table with each config param — expected (from config.md), actual (from server log)

**2b. Instance Count**
- From sim: `--num-instances` in run commands or config.md
- From real: count distinct server log files in `server_logs/`
- Show: expected count vs actual count

---

### 3. SIGNAL PARITY

Read the algorithm files from `<sim>/algorithm/` and the generated Go plugin from `<real>/generated/`.

**IMPORTANT: Code proof requirement.** For every claim about how GAIE/llm-d works, you MUST show the actual source code from the `GAIE` or `LLMD` codebase as proof. Do not make claims based on field names or assumptions — grep the codebase and show file:line and the relevant code snippet.

**3a. Signal Availability**
For each signal the algorithm uses, verify it exists in GAIE:
- Queue depth: grep for `WaitingQueueSize` in `GAIE` codebase
- KV utilization: grep for `KVCacheUsagePercent` in `GAIE` codebase
- Priority: grep for `Priority` in scheduling types in `GAIE` codebase
- Show: for each signal, the **exact GAIE source file:line** and the struct definition or accessor code snippet

**3b. Signal Transform — CODE PROOF REQUIRED**
Check the generated Go plugin applies correct transforms. To determine the correct transform:

1. **Find where GAIE populates the metric**: Grep for `KVCacheUsagePercent` assignments in `GAIE` codebase to see what value range it stores (0-1 or 0-100). Show the code.
2. **Find how GAIE's own saturation detector uses it**: Grep for the saturation detector in `GAIE` (e.g., `detector.go` or similar) to see if it divides by 100. Show the code.
3. **Find the Prometheus metric source**: Grep for the vLLM metric name (e.g., `kv_cache_usage_perc`) to see what vLLM emits. Show the code.
4. **Check test values**: Grep for test files that set `KVCacheUsagePercent` to see what range test data uses (0.5 vs 50.0). Show the code.

Then compare against the generated plugin's transform. Show side-by-side:
```
GAIE source (file:line):     <actual code>
GAIE detector (file:line):   <actual code>
GAIE test (file:line):       <actual code with test values>
Generated plugin (file:line): <actual code>
Sim algorithm (file:line):   <actual code>
```

Verdict: PASS if transforms are consistent, FAIL if not. Plain English explanation of what the bug means practically.

**3c. Saturation Formula — CODE PROOF REQUIRED**
1. **Find GAIE's saturation formula**: Grep for the saturation detector/calculator in `GAIE`. Show the full function.
2. **Find GAIE's empty-pool behavior**: What does GAIE return when there are no pods? Show the code.
3. Compare against sim algorithm and generated plugin. Show all three implementations side-by-side with file:line references.
4. Compare threshold values from YAML against GAIE defaults (grep for default threshold constants in `GAIE`).

Show:
```
GAIE detector (file:line):    <full saturation function>
Sim algorithm (file:line):    <full saturation function>
Generated plugin (file:line): <full saturation function>
```

**3d. Snapshot Delay — CODE PROOF REQUIRED**
- From sim: `--snapshot-refresh-interval` value from config.md
- From GAIE: grep for metric refresh/scrape/staleness configuration in `GAIE` and `LLMD` codebases. Look for:
  - `MetricsStalenessThreshold` or similar
  - Prometheus scrape interval defaults
  - Any signal propagation delay constants
- Show: the actual GAIE/llm-d code that controls signal freshness, with file:line
- Plain English: explain what this delay difference means for admission decisions

---

### 4. POLICY PARITY

**IMPORTANT: Code proof requirement.** For claims about GAIE routing/admission behavior, show the actual source code from the `GAIE` or `LLMD` codebase.

**4a. Routing Policy — CODE PROOF REQUIRED**
- From sim: `--routing-policy` in run commands or README
- From real: check `baseline_config.yaml` and `treatment_config.yaml` for picker plugins
- **From GAIE**: grep for each picker plugin type (e.g., `random-picker`, `fcfs-ordering-policy`, `max-score-picker`) in the `GAIE` and `LLMD` codebases. Show what each plugin does (file:line + brief code snippet or doc comment).
- Map: sim `round-robin` ~ real `random-picker` (both are non-affinity); sim `weighted-scoring` ~ real `max-score-picker` with scorers
- Show: sim policy name, real YAML plugin list, GAIE code proof of what each plugin does, whether they're equivalent

**4b. Admission Policy — Baseline**
- From sim: admission policy for baseline runs (e.g., `gaie-legacy`)
- From real: `baseline_config.yaml` should have NO admission plugin OR the matching GAIE control plugin
- **From GAIE**: if baseline has no admission plugin, grep for GAIE's default/built-in admission behavior. Does GAIE apply any admission by default? Show the code.
- Show: the relevant YAML section from baseline_config.yaml, plus GAIE default behavior proof

**4c. Admission Policy — Treatment**
- From sim: admission algorithm parameters (thresholds, ramps)
- From real: `treatment_config.yaml` plugin parameters must match exactly
- Check: parameter names map correctly to priority tiers (sheddable vs batch — this has been a source of bugs!)
- Show: side-by-side table of sim params vs YAML params

**4d. Priority Tier Mapping — CODE PROOF REQUIRED**
Verify the generated Go plugin dispatches thresholds to the correct priority values:
- critical = 100, standard = 0: always admit
- sheddable = -50: should get the MORE aggressive ramp (lower shedStart)
- batch = -10: should get the LESS aggressive ramp (higher shedStart)

**From GAIE**: grep for priority constants or InferenceModel priority definitions in `GAIE`/`LLMD`. Show where these priority values (100, 0, -10, -50) are defined or documented.

Cross-check against `<sim>/README.md` transfer instructions.
- Show: the Go switch/case block from the generated plugin, annotated with which tier gets which ramp. Show the GAIE priority definitions as proof. Highlight if the mapping is wrong.

---

### 5. RUNTIME ANALYSIS

For each workload run (baseline and treatment), analyze the actual execution:

**5a. vLLM Config Validation**
Parse each server log's startup lines. Verify `non-default args` match expected config.
Extract: model, TP size, max_model_len, gpu_memory_utilization, enable_prefix_caching, max_num_seqs, max_num_batched_tokens.
- Show: extracted values from log vs expected

**5b. Instance Health & Request Distribution**
- Count `/completions` requests per instance by grepping each server log file in `server_logs/` for `/completions` (or `/chat/completions` if chat API). This is the authoritative per-instance request count — `trace_data.csv` does not have an instance identifier column.
- Compute total served requests (sum across instances), per-instance percentage, and spread (max-min as % of average).
- Check for even distribution (for random/round-robin routing): no instance should get <10% or >40% of total (for 4 instances, expect ~25% each). Spread should be < 10%.
- Compare total served vs total sent (from trace_data.csv row count) — the difference is requests shed by admission before reaching any instance.
- Flag any instance that appears unhealthy (no requests, errors in logs)
- Show: table with instance name, `/completions` request count, percentage of total served, and a summary row with total served vs total sent and spread %

**5c. Prefix Hit Ratio**
If `enable_prefix_caching=True`, grep server logs for prefix cache hit metrics.
If `enable_prefix_caching=False`, confirm no unexpected prefix hits.
- Show: prefix caching status and any hit/miss metrics found

**5d. Request Status**
- Baseline: expect nearly 100% `status=ok` (no admission shedding)
- Treatment: check `status` column for rejection patterns. Compute shed rate per SLO class.
- Verify shed behavior matches algorithm expectations (sheddable shed most, batch shed less, critical/standard never shed)
- Show: table with SLO class, total requests, ok count, shed count, shed rate %

**5e. Error Analysis**
Grep server logs for `ERROR`, `WARNING`, `OOM`, `CUDA`, `timeout` patterns.
- Show: count of each pattern per log file, and any particularly concerning lines

**5f. KV Cache Utilization**
From server logs, look for KV cache size info:
- `GPU KV cache size: N tokens`
- `Maximum concurrency for X tokens per request: Y.Zx`
- Show: these values per instance

---

## Output Format

Structure the report as follows. Every section must have evidence tables/snippets, not just verdicts.

```markdown
# Sim2Real Parity Report
**Real bundle**: <path>
**Sim reference**: <path>
**Date**: <date>

## Summary
| Category | PASS | WARN | FAIL |
|----------|------|------|------|
| Workload | X | Y | Z |
| Config   | X | Y | Z |
| Signal   | X | Y | Z |
| Policy   | X | Y | Z |
| Runtime  | X | Y | Z |
| **Total** | **X** | **Y** | **Z** |

## 1. Workload Parity
### Workload: <name>

**1a. Arrival Rate** — PASS
Expected 210 req/s (from workloads/w3.yaml:aggregate_rate), measured 210.4 req/s (25200 requests over 119.8s). Difference: 0.2%.

| Metric | Sim Spec | Real Trace | Diff | Verdict |
|--------|----------|------------|------|---------|
| QPS | 210 | 210.4 | 0.2% | PASS |

[... continue for each check with tables and explanations ...]

## Action Items
1. [FAIL] <what to fix and why>
2. [WARN] <what to investigate>
```

## Parallelization

Use the Agent tool to parallelize independent checks:
- Agent 1: Workload parity (trace_data.csv analysis via python3)
- Agent 2: Config + Signal parity (YAML/Go file comparison)
- Agent 3: Runtime analysis (server log parsing)

Combine results into the final report.
