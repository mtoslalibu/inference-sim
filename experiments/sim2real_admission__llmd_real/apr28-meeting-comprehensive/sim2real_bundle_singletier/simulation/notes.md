# Admission Algorithm Comparison — Qwen3-14B, 4x H100 TP=1

---

# Part 1: Real Experiment Results (apr20, vLLM + llm-d gateway)

Source: `experiments/sim2real_admission_evolution/apr20_paramfree/jing-paramfree-first/results`

## Setup (Real)

- **Baseline**: GAIE-legacy admission (binary cliff: sheddable rejected at saturation >= 1.0)
- **Treatment**: Quintic admission (parameter-free, deployed in llm-d gateway)
- **Infrastructure**: Qwen/Qwen3-14B on 4x H100 TP=1, vLLM serving, llm-d gateway
- **SLO tiers**: 4-tier (critical, standard, batch, sheddable)
- **Workloads**: W1 (balanced) and W2 (batch-heavy) at 2 load levels (under/mid)

## Real Latency Results (ms)

### W1 — Balanced (critical 20%, standard 30%, batch 5%, sheddable 45%)

| Workload | Tier | E2E-Mean (B) | E2E-Mean (T) | dMean | E2E-P99 (B) | E2E-P99 (T) | dP99 |
|----------|------|-------------|-------------|-------|-------------|-------------|------|
| w1_under | critical | 6302 | 5522 | -12% | 13984 | 11739 | -16% |
| w1_under | standard | 6219 | 5451 | -12% | 13163 | 10884 | -17% |
| w1_under | batch | 6211 | 5513 | -11% | 12455 | 12483 | +0% |
| w1_under | sheddable | 6232 | 5464 | -12% | 12444 | 10835 | -13% |
| w1_mid | critical | 16154 | 12088 | **-25%** | 31495 | 30031 | -5% |
| w1_mid | standard | 16144 | 12126 | **-25%** | 31009 | 29853 | -4% |
| w1_mid | batch | 17085 | 23422 | +37% | 34413 | 31122 | -10% |
| w1_mid | sheddable | 17843 | 24148 | +35% | 33001 | 36647 | +11% |
### W2 — Batch-heavy (critical 10%, standard 20%, batch 65%, sheddable 5%)

| Workload | Tier | E2E-Mean (B) | E2E-Mean (T) | dMean | E2E-P99 (B) | E2E-P99 (T) | dP99 |
|----------|------|-------------|-------------|-------|-------------|-------------|------|
| w2_under | critical | 6083 | 5561 | -9% | 14121 | 11447 | -19% |
| w2_under | standard | 5878 | 5401 | -8% | 12095 | 10297 | -15% |
| w2_under | batch | 5878 | 5408 | -8% | 12679 | 11107 | -12% |
| w2_under | sheddable | 5927 | 5412 | -9% | 12933 | 11150 | -14% |
| w2_mid | critical | 15404 | 7258 | **-53%** | 28933 | 19312 | **-33%** |
| w2_mid | standard | 15171 | 7112 | **-53%** | 28818 | 18006 | **-38%** |
| w2_mid | batch | 15488 | 7910 | **-49%** | 29186 | 20797 | -29% |
| w2_mid | sheddable | 14968 | 12690 | -15% | 26738 | 20791 | -22% |
## Real Shed Stats

| Workload | Tier | Total (B) | n (B) | Shed (B) | Total (T) | n (T) | Shed (T) | dShed |
|----------|------|-----------|-------|----------|-----------|-------|----------|-------|
| w1_under | critical | 1230 | 1230 | 0 | 1230 | 1230 | 0 | +0 |
| w1_under | standard | 1840 | 1840 | 0 | 1840 | 1840 | 0 | +0 |
| w1_under | batch | 307 | 301 | 6 | 307 | 256 | 0 | -6 |
| w1_under | sheddable | 2923 | 2882 | 41 | 2923 | 2265 | 0 | -41 |
| w1_mid | critical | 3197 | 3197 | 0 | 3197 | 3197 | 0 | +0 |
| w1_mid | standard | 4863 | 4863 | 0 | 4863 | 4863 | 0 | +0 |
| w1_mid | batch | 774 | 74 | 700 | 774 | 27 | 0 | -700 |
| w1_mid | sheddable | 7366 | 719 | 6647 | 7366 | 230 | 0 | -6647 |
| w2_under | critical | 612 | 612 | 0 | 612 | 612 | 0 | +0 |
| w2_under | standard | 1287 | 1287 | 0 | 1287 | 1287 | 0 | +0 |
| w2_under | batch | 4069 | 4018 | 51 | 4069 | 3565 | 0 | -51 |
| w2_under | sheddable | 332 | 330 | 2 | 332 | 275 | 0 | -2 |
| w2_mid | critical | 1618 | 1618 | 0 | 1618 | 1618 | 0 | +0 |
| w2_mid | standard | 3302 | 3302 | 0 | 3302 | 3302 | 0 | +0 |
| w2_mid | batch | 10480 | 3838 | 6642 | 10480 | 2051 | 0 | -6642 |
| w2_mid | sheddable | 800 | 307 | 493 | 800 | 22 | 0 | -493 |

## Real Experiment Observations

1. **Protected tiers (critical, standard) never shed** in either — both preserve high-priority traffic
2. **W2 mid shows strongest gains**: -53% E2E for critical/standard tiers — quintic's early shedding prevents queue buildup
3. **W1 mid batch/sheddable latency increases** (+35-37%): fewer requests complete but those that do take longer — surviving low-priority requests compete with protected traffic
4. **Treatment eliminates GAIE shedding** (Shed=0 in nearly all treatment rows) — quintic rejects earlier at gateway, requests never reach GAIE's saturation detector
5. **Completed requests (n) drop in treatment** for batch/sheddable — proactive quintic rejection means fewer low-priority requests enter the system

---

# Part 2: BLIS Simulation Results (Binary Quintic k=300)

## Setup

- **Baseline**: GAIE-legacy admission (binary cliff: sheddable rejected when saturation >= 1.0)
- **Treatment**: Binary quintic admission (droppable rejection probability `p = min(sat^5 * 300, 1.0)`)
- **Model**: Qwen/Qwen3-14B on 4x H100 TP=1, trained-physics latency model
- **Routing**: round-robin, snapshot-refresh-interval=50ms
- **Binary tiers**: protected (priority >= 0, always admit) vs droppable (priority < 0, quintic shedding)
- **Workloads**: 10 scenarios (5 shapes x 2 load levels)
- **QPS calibration**: W1/W2 use real-validated rates. New workloads (chatbot, codecompletion, blindspot) adjusted down ~27-29% (under) / ~33-36% (mid) based on sim-vs-real latency gap (sim underestimates E2E by ~1.35x under-load, ~1.6x mid-load)

## Workload Summary

| Workload         | Critical % | Sheddable % | Input Tokens | Output Tokens | Rate (under/mid) |
|------------------|-----------|-------------|--------------|---------------|-------------------|
| W1               | 50%       | 50%         | 1024         | 256           | 35 / 90           |
| W2               | 30%       | 70%         | 1024         | 256           | 35 / 90           |
| Chatbot          | 80%       | 20%         | 4096         | 1024          | 5 / 10            |
| Code Completion  | 30%       | 70%         | 2048         | 128           | 40 / 95           |
| Blindspot        | 10%       | 90%         | 4096         | 1024          | 5 / 10            |

## Latency Results (ms)

### Under-load Workloads

| Workload | Tier | E2E-Mean (B) | E2E-Mean (Q) | dMean | E2E-P99 (B) | E2E-P99 (Q) | dP99 |
|----------|------|-------------|-------------|-------|-------------|-------------|------|
| w1_under | critical | 4460 | 4399 | -1% | 6990 | 6939 | -1% |
| w1_under | sheddable | 4440 | 4378 | -1% | 7131 | 6974 | -2% |
| w2_under | critical | 4458 | 4379 | -2% | 7128 | 6979 | -2% |
| w2_under | sheddable | 4446 | 4375 | -2% | 7088 | 6974 | -2% |
| chatbot_under | critical | 30908 | 24715 | **-20%** | 49976 | 39457 | **-21%** |
| chatbot_under | sheddable | 30654 | 22895 | **-25%** | 51627 | 39905 | **-23%** |
| codecompletion_under | critical | 2469 | 2380 | -4% | 3983 | 3851 | -3% |
| codecompletion_under | sheddable | 2477 | 2384 | -4% | 3995 | 3817 | -4% |
| blindspot_under | critical | 31098 | 19198 | **-38%** | 48580 | 28358 | **-42%** |
| blindspot_under | sheddable | 31701 | 19642 | **-38%** | 54277 | 31524 | **-42%** |

### Mid-load Workloads

| Workload | Tier | E2E-Mean (B) | E2E-Mean (Q) | dMean | E2E-P99 (B) | E2E-P99 (Q) | dP99 |
|----------|------|-------------|-------------|-------|-------------|-------------|------|
| w1_mid | critical | 9749 | 5347 | **-45%** | 15573 | 8378 | -46% |
| w1_mid | sheddable | 9703 | 5356 | **-45%** | 15638 | 8456 | -46% |
| w2_mid | critical | 9746 | 5214 | **-47%** | 15637 | 8211 | -47% |
| w2_mid | sheddable | 9719 | 5186 | **-47%** | 15560 | 8224 | -47% |
| chatbot_mid | critical | 185166 | 180746 | -2% | 327693 | 324324 | -1% |
| chatbot_mid | sheddable | 41203 | 32606 | **-21%** | 69627 | 63382 | -9% |
| codecompletion_mid | critical | 5578 | 2801 | **-50%** | 8839 | 4441 | **-50%** |
| codecompletion_mid | sheddable | 5580 | 2818 | **-49%** | 8855 | 4461 | -50% |
| blindspot_mid | critical | 37303 | 20279 | **-46%** | 57809 | 30792 | **-47%** |
| blindspot_mid | sheddable | 36867 | 20045 | **-46%** | 59254 | 32260 | -46% |

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
| chatbot_under | sheddable | 700 | 0 | 15 | 685 | +685 |
| chatbot_mid | critical | 6478 | 0 | 6478 | 0 | +0 |
| chatbot_mid | sheddable | 46 | 1576 | 14 | 1608 | +32 |
| codecompletion_under | critical | 4953 | 0 | 4953 | 0 | +0 |
| codecompletion_under | sheddable | 11247 | 0 | 10207 | 1040 | +1040 |
| codecompletion_mid | critical | 12130 | 0 | 12130 | 0 | +0 |
| codecompletion_mid | sheddable | 22296 | 6074 | 8813 | 19557 | +13483 |
| blindspot_under | critical | 383 | 0 | 383 | 0 | +0 |
| blindspot_under | sheddable | 3388 | 9 | 1836 | 1561 | +1552 |
| blindspot_mid | critical | 848 | 0 | 848 | 0 | +0 |
| blindspot_mid | sheddable | 3679 | 3573 | 1682 | 5570 | +1997 |

## Key Observations

1. **Protected tier (critical) never shed** in either algorithm — both correctly protect priority >= 0 traffic
2. **W1/W2 mid** (real-validated rates): **-45% to -47% E2E** — quintic aggressively sheds droppable traffic under overload, consistent with real experiment gains (-25% to -53%)
3. **Chatbot under**: -20% critical, -25% sheddable — quintic proactively sheds long-context low-priority traffic even below saturation cliff
4. **Chatbot mid**: Saturated (185s E2E for critical, 142s TTFT) — system capacity ceiling for 80% long-context critical traffic. Quintic helps sheddable (-21%) but can't help critical (-2%)
5. **Code completion mid** (rate=95): **-50% E2E** — at proper load, quintic delivers same magnitude gains as W1/W2
6. **Blindspot** (10% critical, 90% sheddable): Largest gains — **-38% under, -46% mid** — quintic proactively sheds heavy low-priority traffic, dramatically improving both tiers
7. **Trade-off**: Quintic sheds more requests but keeps admitted requests fast
