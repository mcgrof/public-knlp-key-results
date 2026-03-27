# k(D, ε) Lower Bound for Mixed-Precision KV Quantization

## Problem Setup

Given:
- D transformer layers
- k layers protected at INT8 (low noise σ_8)
- (D - k) layers at INT4 (high noise σ_4, with σ_4 >> σ_8)
- Error tolerance ε on relative PPL increase

Find: the minimum k such that quality is preserved.

## Error Budget Formulation

From the accumulation model, the total error is:

$$S(k) = \sum_{\ell \in \text{INT4}} \alpha_\ell \, \sigma_4
       + \sum_{\ell \in \text{INT8}} \alpha_\ell \, \sigma_8$$

The quality constraint requires:

$$S(k) \leq B(\varepsilon)$$

where $B(\varepsilon)$ is the maximum tolerable weighted error sum,
calibrated from the PASS/FAIL boundary in k-sweep experiments.

## Conservative (Worst-Case) Bound

Let $\alpha_{\max}$ be the maximum importance weight among
unprotected layers, and $\alpha_{\min}$ the minimum among
protected layers. Then:

$$(D - k) \, \alpha_{\max} \, \sigma_4 + k \, \alpha_{\min} \, \sigma_8
  \leq B(\varepsilon)$$

Solving for k:

$$k \geq \frac{D \, \alpha_{\max} \, \sigma_4 - B(\varepsilon)}
             {\alpha_{\max} \, \sigma_4 - \alpha_{\min} \, \sigma_8}$$

This bound is conservative because it assumes the worst-case
assignment of importance weights to layers. In practice, we choose
to protect the most sensitive layers, which is strictly better.

## Sorted-Tail (Greedy) Bound

Sort layers by importance weight: $\alpha_{(1)} \geq \alpha_{(2)}
\geq \cdots \geq \alpha_{(D)}$.

Protect the top-k layers (those with largest $\alpha$). The error is:

$$S(k) = \sum_{i=k+1}^{D} \alpha_{(i)} \, \sigma_4
       + \sum_{i=1}^{k} \alpha_{(i)} \, \sigma_8$$

The minimum k satisfying $S(k) \leq B(\varepsilon)$ is found by
computing $S(k)$ for increasing k until the bound is met:

$$k^* = \min \left\{ k : \sum_{i=k+1}^{D} \alpha_{(i)} \, \sigma_4
        + \sum_{i=1}^{k} \alpha_{(i)} \, \sigma_8 \leq B(\varepsilon)
        \right\}$$

This is optimal for the linearized model because greedy protection
of the highest-$\alpha$ layers minimizes total error at each k.

## The O(1) Condition

**Theorem (informal):** If the sorted importance weights $\alpha_{(i)}$
decay sufficiently fast, then $k^*$ remains O(1) as D grows.

Specifically, suppose:
1. There exist $C$ "sink" layers with $\alpha_{(i)} \gg \bar\alpha$
   for $i \leq C$, where $C$ is a model-architecture constant (e.g.,
   the attention sink at layer 0 and a small number of secondary
   sensitive layers).
2. The remaining layers have bounded aggregate sensitivity:
   $\sum_{i=C+1}^{D} \alpha_{(i)} \leq A_{\text{tail}}$ where
   $A_{\text{tail}}$ grows sub-linearly with D (or is bounded).

Then protecting the $C$ sink layers suffices as D grows, giving
$k^* = C = O(1)$.

**Intuition:** In deep transformers, most layers have similar,
moderate sensitivity. Only a small constant number of "special"
layers (layer 0 attention sink, possibly one or two others) have
outsized sensitivity. As D increases, the per-layer contribution
$\alpha_{(i)} \sigma_4$ for typical layers is small enough that their
sum stays within budget even without protection.

## Empirical Evidence

From v23 oracle ablations:

### Qwen2.5-0.5B (D=24):
- Layer 0: +161.8% (extreme sink)
- Layer 8: +2.3%
- All others: <1.5% individually
- k*=4 at ε=3%, k/D=0.167

### Qwen2.5-1.5B (D=28):
- Layer 0: +824.6% (5x worse sink)
- Layer 15: +3.2%
- All others: <0.8%
- k*=2 at ε=3%, k/D=0.071

The tail sensitivity (sum of non-sink layers) is bounded despite D
growing from 24 to 28, consistent with the O(1) condition.

## Estimating B(ε)

The error budget B(ε) is calibrated from k-sweep data:

1. For each model, run k-sweep with oracle ranking at g=32.
2. Find the k at the PASS/FAIL boundary for each ε.
3. Compute $S(k)$ at that boundary using measured $\alpha_\ell$ and
   $\sigma_4$, $\sigma_8$.
4. Set $B(\varepsilon) = S(k^*)$ at the boundary.

## Estimating σ_4/σ_8 Ratio

From single-layer ablations:
- INT8 single-layer delta: typically <0.2% (essentially zero)
- INT4 single-layer delta: ranges from 0.1% to 824%

The ratio $\sigma_4 / \sigma_8$ can be estimated from the per-layer
quantization step sizes or from comparing INT4 vs INT8 single-layer
deltas directly.
