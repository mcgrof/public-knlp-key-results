# Public KNLP Key Results

Public experiment artifacts backing the claims in:

- **FIM-Guided Model Approximation** (paper-fim)
- **Memory-Traffic Saturation in Autoregressive Transformer Decode** (paper-memory-decode)

All data exported from Weights & Biases (entity: `mcgrof-citizen`).

## Projects

### gpt2-ra-sdpa-ablation

Reciprocal attention ablation on GPT-2 124M, 4xB200, FineWebEdu, 1hr wall-clock.

| Run | Best Val PPL | Best Val Loss |
|-----|-------------|--------------|
| gpt2-baseline | 282.06 | 5.642 |
| gpt2-sdpa_gate | 223.67 | 5.410 |
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

## File Structure

Each run directory contains:
- `config.json` -- training hyperparameters
- `summary.json` -- final metrics
- `history.json` -- full training curve (sampled)
- `metadata.json` -- run ID, state, creation date, W&B URL

## Reproduction

Results can be reproduced using the code at https://github.com/linux-kdevops/knlp
with the corresponding defconfig files.
