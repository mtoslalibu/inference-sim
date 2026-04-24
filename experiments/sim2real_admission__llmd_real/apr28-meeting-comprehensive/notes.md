# GAIE-Legacy vs Binary Quintic (k=300) — Qwen3-14B, 4x H100 TP=1

## Setup

- **Baseline**: GAIE-legacy admission (binary cliff: sheddable rejected when saturation >= 1.0)
- **Treatment**: Binary quintic admission (droppable rejection probability `p = min(sat^5 * 300, 1.0)`)
- **Model**: Qwen/Qwen3-14B on 4x H100 TP=1, trained-physics latency model
- **Routing**: round-robin, snapshot-refresh-interval=50ms
- **Binary tiers**: protected (priority >= 0, always admit) vs droppable (priority < 0, quintic shedding)
- **Workloads**: 10 scenarios (5 shapes x 2 load levels)

## Workload Summary

| Workload         | Critical % | Sheddable % | Input Tokens | Output Tokens | Rate (under/mid) |
|------------------|-----------|-------------|--------------|---------------|-------------------|
| W1               | 50%       | 50%         | 1024         | 256           | 35 / 90           |
| W2               | 30%       | 70%         | 1024         | 256           | 35 / 90           |
| Chatbot          | 80%       | 20%         | 4096         | 1024          | 7 / 15            |
| Code Completion  | 30%       | 70%         | 2048         | 128           | 30 / 75           |
| Blindspot        | 10%       | 90%         | 4096         | 1024          | 7 / 15            |

## Latency Results (ms)

### Under-load Workloads

| Workload | Tier | E2E-Mean (B) | E2E-Mean (Q) | dMean | E2E-P99 (B) | E2E-P99 (Q) | dP99 |
|----------|------|-------------|-------------|-------|-------------|-------------|------|
| w1_under | critical | 4460 | 4399 | -1% | 6990 | 6939 | -1% |
| w1_under | sheddable | 4440 | 4378 | -1% | 7131 | 6974 | -2% |
| w2_under | critical | 4458 | 4379 | -2% | 7128 | 6979 | -2% |
| w2_under | sheddable | 4446 | 4375 | -2% | 7088 | 6974 | -2% |
| chatbot_under | critical | 40052 | 38395 | -4% | 63906 | 62334 | -2% |
| chatbot_under | sheddable | 36970 | 26624 | **-28%** | 61350 | 50853 | -17% |
| codecompletion_under | critical | 2173 | 2160 | -1% | 3528 | 3491 | -1% |
| codecompletion_under | sheddable | 2180 | 2165 | -1% | 3499 | 3468 | -1% |
| blindspot_under | critical | 35636 | 19433 | **-45%** | 55971 | 28451 | **-49%** |
| blindspot_under | sheddable | 36026 | 19911 | **-45%** | 59288 | 31708 | **-47%** |

### Mid-load Workloads

| Workload | Tier | E2E-Mean (B) | E2E-Mean (Q) | dMean | E2E-P99 (B) | E2E-P99 (Q) | dP99 |
|----------|------|-------------|-------------|-------|-------------|-------------|------|
| w1_mid | critical | 9749 | 5347 | **-45%** | 15573 | 8378 | -46% |
| w1_mid | sheddable | 9703 | 5356 | **-45%** | 15638 | 8456 | -46% |
| w2_mid | critical | 9746 | 5214 | **-47%** | 15637 | 8211 | -47% |
| w2_mid | sheddable | 9719 | 5186 | **-47%** | 15560 | 8224 | -47% |
| chatbot_mid | critical | 316640 | 311828 | -2% | 587785 | 582529 | -1% |
| chatbot_mid | sheddable | 42853 | 37917 | **-12%** | 72393 | 69893 | -3% |
| codecompletion_mid | critical | 4708 | 2749 | **-42%** | 7605 | 4380 | -42% |
| codecompletion_mid | sheddable | 4709 | 2757 | **-41%** | 7615 | 4371 | -43% |
| blindspot_mid | critical | 37496 | 20567 | **-45%** | 59018 | 31746 | -46% |
| blindspot_mid | sheddable | 36946 | 20757 | **-44%** | 58975 | 32870 | -44% |

## Shed Stats

| Workload | Tier | n (B) | Shed (B) | n (Q) | Shed (Q) | dShed |
|----------|------|-------|----------|-------|----------|-------|
| w1_under | critical | 3137 | 0 | 3137 | 0 | +0 |
| w1_under | sheddable | 3163 | 0 | 2964 | 199 | +199 |
| w1_mid | critical | 8122 | 0 | 8122 | 0 | +0 |
| w1_mid | sheddable | 6713 | 1365 | 852 | 7226 | +5861 |
| w2_under | critical | 1841 | 0 | 1841 | 0 | +0 |
| w2_under | sheddable | 4459 | 0 | 4186 | 273 | +273 |
| w2_mid | critical | 4860 | 0 | 4860 | 0 | +0 |
| w2_mid | sheddable | 9948 | 1392 | 3791 | 7549 | +6157 |
| chatbot_under | critical | 3080 | 0 | 3080 | 0 | +0 |
| chatbot_under | sheddable | 80 | 620 | 14 | 686 | +66 |
| chatbot_mid | critical | 6478 | 0 | 6478 | 0 | +0 |
| chatbot_mid | sheddable | 45 | 1577 | 14 | 1608 | +31 |
| codecompletion_under | critical | 4953 | 0 | 4953 | 0 | +0 |
| codecompletion_under | sheddable | 11247 | 0 | 10962 | 285 | +285 |
| codecompletion_mid | critical | 12130 | 0 | 12130 | 0 | +0 |
| codecompletion_mid | sheddable | 28318 | 52 | 13745 | 14625 | +14573 |
| blindspot_under | critical | 383 | 0 | 383 | 0 | +0 |
| blindspot_under | sheddable | 2616 | 781 | 1264 | 2133 | +1352 |
| blindspot_mid | critical | 848 | 0 | 848 | 0 | +0 |
| blindspot_mid | sheddable | 2197 | 5055 | 867 | 6385 | +1330 |

## Key Observations

1. **Protected tier (critical) never shed** in either algorithm — both correctly protect priority >= 0 traffic
2. **Under-load**: Quintic starts gentle shedding (sat^5 dead zone) while GAIE-legacy sheds nothing — small latency improvements (1-4%) except blindspot (-45%) and chatbot sheddable (-28%)
3. **Mid-load**: Quintic aggressively sheds droppable traffic, yielding **40-47% E2E latency reduction** for both tiers in W1, W2, codecompletion, and blindspot
4. **Chatbot mid**: Extreme TTFT for critical tier (~270s) in both — system is saturated by long-context requests. Quintic provides marginal improvement (-2%)
5. **Blindspot** (10% critical, 90% sheddable): Large improvement even under-load because quintic proactively sheds heavy low-priority traffic before saturation cliff
6. **Trade-off**: Quintic sheds significantly more requests but keeps admitted requests fast. GAIE-legacy holds requests longer, causing queue buildup
