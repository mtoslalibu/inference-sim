---
description: Analyze sim2real experiment results. Given a results directory, produce per-workload latency comparison tables between baseline and treatment.
---

## User Input

```text
$ARGUMENTS
```

Parse user input for the results directory path. Example: `experiments/sim2real_admission_evolution/apr16_concurrency/jing-concurrency/results`

## Goal

Read all workload results under `<results>/baseline/` and `<results>/treatment/`, compute latency metrics from `trace_data.csv`, and print clean comparison tables.

## Operating Constraints

**STRICTLY READ-ONLY**: Do not modify any files. Only print analysis output.

## Step 1: Discover Workloads

List subdirectories under `<results>/baseline/` and `<results>/treatment/`. Match workloads that exist in both. Sort by name.

## Step 2: QPS Fidelity Check

For each workload, compute:
- **Intended QPS**: `(num_requests - 1) / ((max(arrival_time_us) - min(arrival_time_us)) / 1e6)`
- **Actual QPS**: `(num_requests - 1) / ((max(send_time_us) - min(send_time_us)) / 1e6)`

If actual QPS is more than 5% below intended QPS, flag it with a warning — the load generator hit a concurrency ceiling.

## Step 3: Per-Workload Tables

For each workload, produce ONE table comparing baseline vs treatment for **accepted requests only** (status=ok).

### Metrics

All metrics computed from `trace_data.csv` columns. Use `send_time_us` as the reference start time (absolute Unix epoch microseconds).

| Metric | Formula | Description |
|--------|---------|-------------|
| **E2E** (end-to-end latency) | `(last_chunk_time_us - send_time_us) / 1000` | Total time from request sent to last response token received, in ms |
| **TTFT** (time to first token) | `(first_chunk_time_us - send_time_us) / 1000` | Time from request sent to first streaming token received, in ms. Includes network RTT + server queue wait + prefill compute |
| **TPOT** (time per output token) | `(last_chunk_time_us - first_chunk_time_us) / (output_tokens - 1) / 1000` | Average inter-token latency during generation, in ms. Only for requests with output_tokens > 1 |

### Aggregations

For each metric, show: **mean**, **p50**, **p90**, **p99**

### Shed ratio

- Count requests where `status != 'ok'` as shed
- Show: `shed_count / total_count (percentage)`

### Table Format

Two tables per workload: one **aggregate** (all accepted requests) and one **per SLO tier**.

**Table 1: Aggregate**

```
### <workload_name> (<intended_qps> QPS intended, <actual_qps> QPS actual)

| Metric | Baseline | Treatment | Diff |
|--------|----------|-----------|------|
| **Shed** | X/Y (Z%) | X/Y (Z%) | +Npp |
| **E2E mean** | Xms | Xms | -N% |
| **E2E p50** | Xms | Xms | -N% |
| **E2E p90** | Xms | Xms | -N% |
| **E2E p99** | Xms | Xms | -N% |
| **TTFT mean** | Xms | Xms | -N% |
| **TTFT p50** | Xms | Xms | -N% |
| **TTFT p90** | Xms | Xms | -N% |
| **TTFT p99** | Xms | Xms | -N% |
| **TPOT mean** | Xms | Xms | -N% |
| **TPOT p50** | Xms | Xms | -N% |
| **TPOT p90** | Xms | Xms | -N% |
| **TPOT p99** | Xms | Xms | -N% |
```

**Table 2: Per SLO Tier**

For each SLO tier (critical, standard, sheddable, batch), show a row with key metrics. Only include tiers that have at least 10 accepted requests in both baseline and treatment.

```
#### Per SLO Tier

| Tier | Shed (B/T) | E2E mean (B) | E2E mean (T) | E2E Diff | TTFT p50 (B) | TTFT p50 (T) | TTFT Diff | TPOT mean (B) | TPOT mean (T) | TPOT Diff |
|------|------------|-------------|-------------|----------|-------------|-------------|-----------|-------------|-------------|-----------|
| critical | 0%/0% | Xms | Xms | -N% | Xms | Xms | -N% | Xms | Xms | -N% |
| standard | 0%/0% | Xms | Xms | -N% | Xms | Xms | -N% | Xms | Xms | -N% |
| sheddable | X%/Y% | Xms | Xms | -N% | Xms | Xms | -N% | Xms | Xms | -N% |
| batch | X%/Y% | Xms | Xms | -N% | Xms | Xms | -N% | Xms | Xms | -N% |
```

Diff column: percentage change from baseline to treatment. Negative = treatment is better (lower latency). For shed, show percentage point difference.

Format millisecond values: use integers for values >= 1000ms, one decimal for values < 1000ms.

## Step 4: Summary

After all tables, print a brief summary:

```
## Summary

| Workload | QPS | Shed (B/T) | E2E mean gain | TTFT p50 gain | TPOT mean gain |
|----------|-----|------------|---------------|---------------|----------------|
| ... | ... | ... | ... | ... | ... |
```

## Important: Output Method

**Use Bash (python3) to compute the numbers, then OUTPUT THE TABLES AS PLAIN TEXT in your response message.** Do NOT rely on Bash tool output being visible to the user — it is not. The tables must appear in your text response so the user can see them in their terminal.

Workflow:
1. Run python3 via Bash to compute all metrics and collect them into a structured result
2. Read the Bash output
3. Write the tables as markdown text in your response message to the user

## Notes

- Only analyze accepted requests (status=ok) for latency metrics
- If a workload exists in baseline but not treatment (or vice versa), skip it with a note
- Use python3 via Bash for CSV processing
- Do NOT use `arrival_time_us` for latency calculations — it is relative, not absolute. Always use `send_time_us` as the start reference
