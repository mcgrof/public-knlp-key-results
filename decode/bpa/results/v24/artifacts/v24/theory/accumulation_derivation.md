# Accumulation Model for KV Quantization Error

## Setup

A transformer with D layers computes a residual recursion:

$$x_{\ell+1} = x_\ell + f_\ell(x_\ell)$$

where $f_\ell$ is the attention+FFN block at layer $\ell$.

## Quantization Perturbation

KV cache quantization introduces a perturbation $\delta_\ell$ at each
layer, modifying the computation:

$$\tilde{x}_{\ell+1} = \tilde{x}_\ell + f_\ell(\tilde{x}_\ell) + \delta_\ell(\tilde{x}_\ell)$$

Here $\delta_\ell$ captures the effect of using quantized K,V in
attention instead of the original fp16 values. The perturbation is
data-dependent but bounded by the quantization error.

## Error Recursion

Define the error $e_\ell = \tilde{x}_\ell - x_\ell$.

Then:

$$e_{\ell+1} = e_\ell + [f_\ell(\tilde{x}_\ell) - f_\ell(x_\ell)] + \delta_\ell$$

## Linearization (First-Order Approximation)

Assuming $e_\ell$ is small relative to $x_\ell$, linearize:

$$f_\ell(\tilde{x}_\ell) - f_\ell(x_\ell) \approx J_\ell \cdot e_\ell$$

where $J_\ell = \partial f_\ell / \partial x \big|_{x=x_\ell}$ is the
Jacobian of the $\ell$-th block. This gives:

$$e_{\ell+1} \approx (I + J_\ell) \, e_\ell + \delta_\ell$$

## Unrolling

Starting from $e_0 = 0$ (no quantization before the first layer):

$$e_D = \sum_{\ell=0}^{D-1} A_{\ell \to D} \, \delta_\ell$$

where the **amplification operator** is:

$$A_{\ell \to D} = \prod_{j=\ell+1}^{D-1} (I + J_j)$$

with the convention that $A_{D-1 \to D} = I$.

## Norm Bound

Taking norms:

$$\|e_D\| \leq \sum_{\ell=0}^{D-1} \|A_{\ell \to D}\| \cdot \|\delta_\ell\|$$

Define the **amplification weight** $w_\ell = \|A_{\ell \to D}\|$ and
the **noise magnitude** $\sigma_\ell = \|\delta_\ell\|$ (expected over
data). Then:

$$\|e_D\| \leq \sum_{\ell=0}^{D-1} w_\ell \, \sigma_\ell$$

## Loss Linkage

The final error $e_D$ in representation space translates to a change
in next-token log-likelihood. Under a local Lipschitz assumption on
the language model head:

$$\Delta\text{NLL} \approx c \cdot \|e_D\|$$

where $c$ is a model-specific constant (fitted empirically from
single-layer ablations). Combining:

$$\Delta\text{NLL} \leq \sum_{\ell=0}^{D-1} \alpha_\ell \, \sigma_\ell$$

where $\alpha_\ell = c \cdot w_\ell$ are the **importance weights**
that combine amplification and head sensitivity.

## Practical Interpretation

- $\alpha_\ell$ is large for layers whose quantization errors are
  strongly amplified by downstream layers. Empirically, layer 0
  (attention sink) has the largest $\alpha_0$ because all subsequent
  layers amplify its error.

- $\sigma_\ell$ depends on the quantization scheme: INT8 gives
  $\sigma_8 \ll \sigma_4$ (INT4). The ratio $\sigma_4 / \sigma_8$
  determines how much worse INT4 is per layer.

- The error is **additive** across layers (first-order), explaining
  why per-layer INT4 sensitivity tests (each <3%) can accumulate to
  >3% when all layers are INT4 simultaneously.

## Key Assumption: Linearity

The derivation assumes errors stay small enough for linearization.
This is valid when:

1. Protected (INT8) layers have negligible error ($\sigma_8 \approx 0$).
2. The number of INT4 layers is moderate (not all D).
3. Individual $\delta_\ell$ are small relative to $\|x_\ell\|$.

Empirical evidence from v15-v23 supports this: single-layer INT4
deltas are typically <3%, and the linear accumulation model explains
the observed k-floor behavior.

## Measuring the Parameters

Two estimation routes:

**Route A (Direct, oracle ablations):** For each layer $\ell$, apply
INT4 at that layer only (all others INT8). The measured $\Delta$PPL
gives $s_\ell \approx \alpha_\ell \sigma_4$ directly.

**Route B (Amplification traces):** Inject small Gaussian noise at
layer $\ell$'s KV cache. Measure $\|e_j\|$ at each subsequent layer
$j$. The growth rate gives $w_\ell$. Combined with separately
measured $\sigma_4$ and $\sigma_8$, this yields $\alpha_\ell$.
