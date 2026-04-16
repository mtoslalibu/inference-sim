---
description: Find optimal SLO thresholds that maximize attainment gap between baseline and treatment across all workloads.
---

## User Input

```text
$ARGUMENTS
```

Parse user input for the results directory path. Example: `experiments/sim2real_admission_evolution/apr16_concurrency/jing-concurrency/results`

## Goal

Find a **single E2E latency threshold for critical** and a **single E2E latency threshold for standard** that maximizes the SLO attainment gap (treatment - baseline) **averaged across all workloads**. Then print per-workload attainment tables at those thresholds.

## Operating Constraints

**STRICTLY READ-ONLY**: Do not modify any files. Only print analysis output.

## Step 1: Discover Workloads

List subdirectories under `<results>/baseline/` and `<results>/treatment/`. Match workloads that exist in both. Sort by name.

## Step 2: Compute Optimal Thresholds

Use python3 via Bash to:

1. For each workload, load `trace_data.csv` from both baseline and treatment.
2. For accepted requests only (`status=ok`), compute E2E latency: `(last_chunk_time_us - send_time_us) / 1000` (in ms).
3. Sweep thresholds from 1000ms to 30000ms in 100ms steps.
4. For each threshold, for each tier (critical, standard separately):
   - Compute attainment = `count(e2e <= threshold) / total_ok_requests_in_tier * 100`
   - Compute gap = `treatment_attainment - baseline_attainment`
5. For each tier, compute the **max gap across all workloads** at each threshold.
6. Pick the threshold that maximizes the max gap across workloads. This gives one threshold for critical, one for standard.

## Step 3: Output Tables

**Use Bash (python3) to compute the numbers, then OUTPUT THE TABLES AS PLAIN TEXT in your response message.** Do NOT rely on Bash tool output being visible to the user — it is not. The tables must appear in your text response.

### Table 1: Optimal Thresholds with Per-Workload Attainment

One table per tier. The threshold is chosen to maximize the best single-workload gap across all workloads. Show all workloads.

```
## Optimal SLO Thresholds (maximizing max attainment gap across workloads)

### Critical (SLO = Xms, best gap: +N.Npp)

| Workload | QPS | Baseline | Treatment | Gap |
|----------|-----|----------|-----------|-----|
| ... | ... | X.X% | X.X% | +N.Npp |

### Standard (SLO = Xms, best gap: +N.Npp)

| Workload | QPS | Baseline | Treatment | Gap |
|----------|-----|----------|-----------|-----|
| ... | ... | X.X% | X.X% | +N.Npp |
```

### Table 3: Sweep Context

For each tier, show 5 thresholds around the optimum (every 500ms) so the user can see sensitivity:

```
### Threshold Sensitivity

#### Critical
| Threshold | Mean Gap | W1 Gap | W2 Gap | ... |
|-----------|----------|--------|--------|-----|
| ... | ... | ... | ... | ... |

#### Standard
| Threshold | Mean Gap | W1 Gap | W2 Gap | ... |
|-----------|----------|--------|--------|-----|
| ... | ... | ... | ... | ... |
```

## Step 4: QPS Extraction

For each workload, compute intended QPS from trace: `(num_requests - 1) / ((max(arrival_time_us) - min(arrival_time_us)) / 1e6)`. Show this in the per-workload table.

## Notes

- Only analyze accepted requests (`status=ok`) for attainment
- Use `send_time_us` as start time (not `arrival_time_us`)
- The goal is ONE threshold per tier that works best across ALL workloads simultaneously
- Use python3 via Bash for CSV processing
