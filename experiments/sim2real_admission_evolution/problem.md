# SLO Attainment Admission Evolution

## Goal

Evolve an admission control algorithm (AdaptiveAdmission) that beats GAIE legacy baseline by **30% or more** in SLO attainment for **critical** and **standard** requests.

## Metric

**SLO attainment** = fraction of requests (completed + rejected) that complete under an E2E latency target. A rejected request is a guaranteed SLO miss (infinite latency).

Targets: E2E < 3s, 5s, 10s, 20s (reported per SLO class).

## Baseline: GAIE Legacy

Binary cliff gate: computes saturation = avg(max(queueDepth/5.0, kvUtilization/0.8)) per instance. When saturation >= 1.0, rejects all requests with priority < 0 (sheddable, batch). Never rejects critical or standard (priority >= 0).

**Key weakness**: GAIE only activates at saturation 1.0 (very late). By then, queues are deep and critical/standard requests already have high latency. Earlier shedding of low-priority traffic could free capacity sooner.

## Strategy

1. Never reject critical or standard (rejection = guaranteed SLO miss)
2. Shed sheddable much earlier than GAIE (saturation 0.3-0.5 vs 1.0) to free capacity
3. Earlier shedding = shorter queues for critical/standard = more requests meeting SLO target
4. The improvement comes from latency reduction, not from avoiding rejections (both algorithms protect critical/standard)

## Workloads

- **W1**: Sustained 1.5x overload, 45% sheddable, large requests (1024in/256out)
- **W2**: Burst pattern: 1x steady then 2x burst, 45% sheddable, large requests

## Success Criterion

Critical and standard SLO attainment improves by >= 30% (absolute or relative) across workloads and SLO targets.
