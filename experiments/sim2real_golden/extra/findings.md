1. FM-3: Burst Absorption — Best all-around showcase
- No timeouts, clean runs across all seeds
- v2: +21% E2E mean, +39% E2E P99, +22% TTFT mean
- expanded: +23% E2E mean, +42% E2E P99, +25% TTFT mean
- Shows gains on both E2E and TTFT — easy to explain

2. FM-2a: Groups > Instances — Biggest TTFT wins
- No timeouts
- v2: +18% E2E P99, +30% TTFT mean, +43% TTFT P99
- expanded: +33% E2E P99, +48% TTFT mean, +79% TTFT P99
- 6 prefix groups across 4 instances is a realistic multi-tenant scenario

3. FM-6: Cold Traffic Under KV Pressure — Dramatic TTFT P99 improvement
- No timeouts for v2/expanded/glia (only baseline seed=456 times out)
- v2: +95% TTFT P99 (2195ms → 115ms)
- expanded: +95% TTFT P99 (2195ms → 117ms)
- E2E gains are modest (+3%), but the TTFT tail collapse is striking
- Mixed cold/warm traffic is realistic for production


These 3 cover burst traffic, multi-tenant prefix routing, and cold/warm mix — all realistic production scenarios with clean reproducibility.
