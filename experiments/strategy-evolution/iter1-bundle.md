# Iteration 1: Hypothesis Bundle — Marginal-Gain Scoring

## H-main: Marginal-gain scoring outperforms both baselines

**Prediction:** With α=1.0 (equal weight to cache benefit and load cost), marginal-gain scoring will:
- W1 (prefix-heavy): ≥20% E2E P99 improvement over GLIA, within 5% of BLIS 3:2:2
- W2 (load stress): ≥10% E2E mean improvement over both baselines
- Mechanism: automatic regime adaptation via score-magnitude cancellation

**Diagnostic if fails:** If W1 fails, cache signal staleness (2s) is too coarse for prefix-heavy workloads. If W2 fails, α=1.0 over-values cache when it should focus on load balance.

## H-ablation-cache: Cache component contribution

**Prediction:** Removing cache scoring (setting all cache scores to 0 → pure load balance) will:
- W1: Degrade E2E P99 by ≥30% (prefix cache is critical)
- W2: Improve or be neutral on E2E metrics (cache adds noise under load stress without prefix sharing)

**Diagnostic if fails:** If W1 doesn't degrade, the cache signal at 2s staleness isn't actually helping routing decisions. If W2 degrades, the cache signal has indirect load-balancing effects.

## H-ablation-load: Load component contribution  

**Prediction:** Removing load scoring (setting α=0 → pure cache routing) will:
- W1: Be comparable to full algorithm (cache dominates in prefix-heavy workload)
- W2: Degrade E2E P99 by ≥20% (load balance is critical without prefix sharing)

**Diagnostic if fails:** If W1 degrades, load balance matters even with prefix sharing (pile-on on cached instance). If W2 doesn't degrade, our load signal isn't carrying weight.

## H-control-negative: Effect vanishes on uniform workload

**Prediction:** On a workload with no prefix sharing and uniform request sizes at low rate, marginal-gain scoring should produce results indistinguishable from round-robin (both signals have zero spread, ties broken randomly).

## H-alpha-sensitivity: α parameter sensitivity

**Prediction:** α in [0.5, 2.0] produces stable results (within 10% of best). Below 0.5, W2 degrades (over-commit to cache). Above 2.0, W1 degrades (over-commit to load).

---

## Implementation Plan

1. Create `routing_marginal_gain.go` with the MarginalGainScoring algorithm
2. Wire it as a new routing policy variant (or replace the WeightedScoring.Route() body)
3. Run H-main across both workloads × 3 seeds
4. Run H-ablation-cache (α=0 effectively, or zero out cache scores) 
5. Run H-ablation-load (α=0)
6. Compare predictions to outcomes
