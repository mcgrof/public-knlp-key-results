# Public KNLP Key Results

Public experiment artifacts backing the claims in:

- **FIM-Guided Model Approximation** (paper-fim)
- **Memory-Traffic Saturation in Autoregressive Transformer Decode** (paper-memory-decode)

## FIM Paper (GPT-2 W&B Exports)

W&B data exported from entity `mcgrof-citizen`.

### gpt2-ra-sdpa-ablation

Reciprocal attention ablation on GPT-2 124M, 4xB200, FineWebEdu, 1hr wall-clock.

| Run | Best Val PPL | Best Val Loss |
|-----|-------------|--------------|
| gpt2-baseline | 282.06 | 5.642 |
| gpt2-ra | 50.46 | 3.921 |

### gpt2-kvsplice-ablation-test

KVSplice ablation on GPT-2 124M + MLA, 4xB200, FineWebEdu, 1hr wall-clock (v2 runs).

| Run | Best Val PPL | HellaSwag (norm) |
|-----|-------------|-----------------|
| MLA baseline | 71.05 | 30% |
| MLA + KVSplice all | 83.85 | 24% |
| MLA + KVSplice FIM | 63.08 | 31% |

### gpt2-bitter8-nocompile-b200x4

Pruning variant comparison on GPT-2 124M, B200x4, FineWebEdu, 50% sparsity.

| Run | Best Val PPL |
|-----|-------------|
| magnitude (bitter1) | 44.15 |
| bitter7 (FIM-guided) | 37.28 |
| bitter8 | 40.94 |
| bitter9 | 40.93 |

### File Structure (W&B exports)

Each run directory contains:
- `config.json` -- training hyperparameters
- `summary.json` -- final metrics
- `history.json` -- full training curve (sampled)
- `metadata.json` -- run ID, state, creation date, W&B URL

## Decode Paper (`decode/`)

Benchmark results backing the memory-traffic saturation paper.

### decode/bpa/

Core BPA (Bandwidth-Performance Analysis) results: 876 JSON files across
26 experiment versions. Covers 14 model architectures (7B-class), KV
precision asymmetry, key/value sensitivity, ratio classifier evaluation,
fused INT4 decode benchmarks across W7900, A100, H100, B200.

### decode/bpa-h100/

H100-specific GPU validation results (L40S cross-validation).

### decode/bpa-large-model/ and decode/bpa-large-model-v2/

Large-model validation: Llama-3.1-70B and Qwen2.5-72B on WikiText-103.
Per-layer sensitivity, scale attenuation of Qwen key sensitivity.

### decode/bpa-multi-gpu/

Multi-GPU tensor-parallel configuration results.

### decode/h100-vllm-fused-int4/

H100 vLLM fused INT4 decode test campaigns (March 2026): smoke testing,
threshold sweeps, KV precision-split policy evaluation.

### decode/spev01--spev06/

Speculative decoding experiments: interaction between speculative decode
and fused KV quantization.

## Reproduction

Results can be reproduced using the code at https://github.com/mcgrof/knlp
with the corresponding defconfig files and benchmark scripts.
