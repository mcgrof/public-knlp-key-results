# BPA v3: KV Cache Memory, FLOPs, and Time Results

## Model
- GPT2_RGSA (124M params), FineWebEdu, 615 iters
- Config: n_layer=12, n_head=12, n_embd=768
- local_window=256, chunk_size=64, top_b=8
- Evaluation: FineWebEdu val.bin, bf16 KV accounting

## Results at L=512

| Variant | PPL | PPL vs Dense | Enabled Rate | Kept Tokens | KV Read (KB) | KV Savings | FLOPs% | ms/tok |
|---------|-----|-------------|--------------|------------|-------------|------------|--------|--------|
| V0: Dense (baseline) | 245.5+/-8.2 | 0.0% | 1.000 | 512 | 18432 | 0.0% | 100.0% | 0.256 |
| V1: Local-only | 245.5+/-8.4 | -0.0% | 0.000 | 256 | 9216 | 50.0% | 50.0% | 0.257 |
| V2: Learned gate | 247.7+/-8.4 | +0.9% | 0.453 | 237 | 13388 | 27.4% | 72.6% | 0.637 |
| V3: Random gate | 245.9+/-6.3 | +0.2% | 0.457 | 222 | 13424 | 27.2% | 72.8% | 0.268 |

### V2 Tokens/Query Distribution (L=512)
- Mean: 237
- P50:  237
- P95:  242
- P99:  245

## Results at L=1024

| Variant | PPL | PPL vs Dense | Enabled Rate | Kept Tokens | KV Read (KB) | KV Savings | FLOPs% | ms/tok |
|---------|-----|-------------|--------------|------------|-------------|------------|--------|--------|
| V0: Dense (baseline) | 244.8+/-9.8 | 0.0% | 1.000 | 1024 | 36864 | 0.0% | 100.0% | 0.374 |
| V1: Local-only | 275.9+/-11.7 | +12.7% | 0.000 | 256 | 9216 | 75.0% | 25.0% | 0.380 |
| V2: Learned gate | 272.4+/-11.4 | +11.3% | 0.289 | 366 | 17202 | 53.3% | 46.7% | 0.889 |
| V3: Random gate | 281.0+/-7.6 | +14.8% | 0.290 | 308 | 17235 | 53.2% | 46.8% | 0.365 |

### V2 Tokens/Query Distribution (L=1024)
- Mean: 366
- P50:  366
- P95:  385
- P99:  389

## KV Cache Accounting

### Write Cost
- KV write per token: 36,864 bytes (36 KB)
- Write cost is constant across all variants (all tokens are written)

### Peak KV Allocation
- L=512: 18,874,368 bytes (18.0 MB)
- L=1024: 37,748,736 bytes (36.0 MB)

Peak KV allocation is unchanged across variants because the current implementation uses dense cache layout. All tokens are stored in the KV cache regardless of gating. Only the effective read traffic and compute (FLOPs) are reduced by BPA gating.

## Wall Time Analysis

V2 (learned gate) is significantly slower than V0/V1/V3 because the gate feature extraction requires running local-only attention at each layer to compute features, then evaluating the MLP gate per position. This is an implementation artifact, not fundamental to BPA.

V0, V1, and V3 have similar wall time because they all use dense matrix operations (masked attention). The attention mask does not reduce computation with dense kernels — it only changes which positions contribute to the output. True wall-time savings require sparse attention kernels that skip masked-out positions.

## Verdict

### L=512
**GO**: KV read savings = 27.4% with PPL regression = 0.9%. Proceed to L=2048.

### L=1024
**CONDITIONAL**: KV read savings = 53.3% but PPL regression = 11.3% (>1.0% threshold).
V2 gate (PPL=272.4) is barely better than V1 local-only (PPL=275.9). The gate adds value only if it selectively enables far-context where needed.

### Overall

BPA reduces effective KV read traffic by 27% at L=512 and 53% at L=1024. At L=512, PPL regression is negligible (<1%). At L=1024, PPL regresses ~12% — far-context genuinely helps at longer sequences and the gate is not selective enough to preserve quality while skipping it.

Wall time savings do not materialize because the current implementation uses dense attention kernels. The attention mask changes which positions contribute but does not reduce computation. True time savings require sparse attention kernels.

Peak KV allocation is unchanged (dense cache layout). BPA reduces only the effective usage/traffic, not the allocation.
