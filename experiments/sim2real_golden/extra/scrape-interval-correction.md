# Scrape Interval Correction for Routing Experiments

## The Problem

`compare.sh` uses `SNAPSHOT_REFRESH=5000000` (5s) with the comment "production-like staleness." This was matched to the wrong GAIE parameter.

## GAIE Has Three Timing Parameters

From `pkg/epp/server/options.go` in GAIE:

| Parameter | Default | What It Controls |
|-----------|---------|------------------|
| `RefreshMetricsInterval` | **50ms** | How often EPP scrapes each pod's vLLM `/metrics` endpoint. **This is the signal freshness scorers see.** |
| `MetricsStalenessThreshold` | **2s** | Safety guard: if last scrape > 2s old, treat pod as saturated (score 1.0). Only fires on pod crash / network failure. |
| `RefreshPrometheusMetricsInterval` | **5s** | How often EPP publishes its own aggregated gauges to EPP's own `/metrics` for Grafana. **Scorers never read this.** |

The 5s we used matches `RefreshPrometheusMetricsInterval` — EPP's self-monitoring interval. The actual pod metric scrape that feeds scorers is **50ms**.

## Data Flow Diagram

```
Pod vLLM /metrics  --[50ms]--> in-memory Metrics ---> scorers (queue-depth, kv-utilization, load-aware, etc.)
                                     |                 saturation detector
                                     |                 admission controller
                                [5s aggregate]
                                     v
                            EPP's own /metrics ---> Grafana dashboards (monitoring EPP itself)
```

Scorers read the in-memory `Metrics` struct which is refreshed every 50ms. The 5s interval only affects what Grafana sees about EPP health.

## What to Change

In `compare.sh`, line 29:

```bash
# OLD (wrong — matches EPP self-publishing, not pod scrape)
SNAPSHOT_REFRESH=5000000         # 5s — stale snapshots (production-like)

# CORRECT — matches GAIE's RefreshMetricsInterval default
SNAPSHOT_REFRESH=50000           # 50ms — GAIE pod scrape interval
```

## Impact on Results

With 5s refresh, scorers saw snapshots that were on average 2.5s stale. At high request rates, that means:
- `queue-depth` scorer was reacting to queue state from 2.5s ago
- `kv-utilization` scorer was seeing KV pressure from 2.5s ago
- `load-aware` scorer was using stale load signals

With 50ms refresh, signals are <50ms old — much closer to real-time.

**The 5s setting was not fine for routing.** It artificially amplified staleness far beyond what production llm-d experiences, making all routers worse than real production:
- At high request rates, thousands of requests pile onto one instance before any scorer notices
- The baseline's `queue-depth:1` was especially handicapped — its only load-balancing signal was blind for 5s
- The adaptive router's regime detection (cache spread, avg KV util) was also reading 2.5s-old averages
- Results are directionally valid (adaptive still beat baseline) but the absolute numbers and improvement magnitudes could change significantly with correct 50ms scraping

With correct 50ms scraping:
- `queue-depth` and `load-aware` scorers react to load changes 100x faster
- Pile-on effects are detected and corrected much sooner
- Baseline may perform better (its queue-depth signal actually works now), narrowing the gap with adaptive
- Or adaptive may still dominate — but we need to rerun to find out

The **cache signal delay** (`--cache-signal-delay 2000000`, 2s) is separate and correct — it models the ZMQ propagation delay for KV cache events in llm-d, not Prometheus scraping.

## Recommended Experiment Matrix

To understand the impact, run both settings:

```bash
# Run 1: Correct GAIE parity (50ms scrape)
SNAPSHOT_REFRESH=50000 bash compare.sh

# Run 2: Original setting for comparison (5s scrape)
SNAPSHOT_REFRESH=5000000 bash compare.sh
```

Compare results to see how much signal freshness affects routing quality. If the adaptive routers still win with fresh signals, the result is stronger (the win isn't just from being better at handling staleness).

## Settings Summary for Rerun

| Parameter | Value | Source |
|-----------|-------|--------|
| `--snapshot-refresh-interval` | `50000` (50ms) | GAIE `RefreshMetricsInterval` default |
| `--cache-signal-delay` | `2000000` (2s) | llm-d `defaultSpeculativeTTL` (unchanged, still correct) |
| Seeds | 42, 123, 456 | Same as before |
| Everything else | Unchanged | Same model, instances, workloads, routers |
