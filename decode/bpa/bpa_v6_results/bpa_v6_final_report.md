# BPA v6: Scale + Polish Results

> Boundary-Pressure Attention is a conditional rate controller for the model's history channel, attempting to match memory usage to the mutual information structure of language rather than to sequence length.

## Model
- GPT2_RGSA (124M params), FineWebEdu, 615 iters
- Config: n_layer=12, n_head=12, n_embd=768
- local_window=256, chunk_size=64
- Gate: v4 wbce_10x at 70% enabled_rate
- Surgical heads: 8 heads from layers 5-8
- Evaluation: FineWebEdu val.bin, bf16 KV accounting

## Gate Overhead Reduction

| Config | PPL | Gate ms/tok | Fwd ms/tok | Total ms/tok |
|--------|-----|------------|------------|-------------|
| baseline | 278.9 | 0.7500 | 0.4318 | 1.1902 |
| fast_gate | 270.7 | 0.2488 | 0.4560 | 0.7063 (1.7x) |
| sparse_ra_4 | 278.9 | 0.7754 | 0.4518 | 1.2268 |
| sparse_ra_8 | 279.0 | 0.7601 | 0.4472 | 1.2059 |
| fast+sparse4 | 270.8 | 0.2516 | 0.4350 | 0.6955 (1.7x) |

Best config: **fast+sparse4** at 1.7x speedup vs baseline, PPL=270.8

## Results at L=512

| Variant | PPL | PPL vs Dense | Kept Tokens | KV Read B/tok | FLOPs% | ms/tok |
|---------|-----|-------------|-------------|--------------|--------|--------|
| V0_dense | 251.8+/-6.2 | +0.0% | 320160 | 18874368 | 100.0% | 0.273 |
| V1_local_only | 251.6+/-6.2 | -0.0% | 192 | 9437184 | 50.0% | 0.268 |
| V6_ra_blend_fb4 | 251.9+/-6.3 | +0.0% | 282 | 16035840 | 85.0% | 0.746 |
| V6_ra_blend_fb8 | 251.9+/-6.3 | +0.0% | 371 | 18874368 | 100.0% | 0.740 |
| V6_ra_value_fb4 | 251.9+/-6.3 | +0.0% | 282 | 16035840 | 85.0% | 0.739 |
| V6_ra_value_fb8 | 251.9+/-6.3 | +0.0% | 371 | 18874368 | 100.0% | 0.736 |
| V6_recency_fb4 | 251.9+/-6.3 | +0.0% | 282 | 16035840 | 85.0% | 0.734 |
| V6_recency_fb8 | 251.9+/-6.3 | +0.0% | 371 | 18874368 | 100.0% | 0.742 |
| V6_random_fb4 | 286.2+/-26.0 | +13.7% | 282 | 16035840 | 85.0% | 0.737 |
| V6_random_fb8 | 286.2+/-26.0 | +13.7% | 371 | 18874368 | 100.0% | 0.734 |

## Results at L=1024

| Variant | PPL | PPL vs Dense | Kept Tokens | KV Read B/tok | FLOPs% | ms/tok |
|---------|-----|-------------|-------------|--------------|--------|--------|
| V0_dense | 248.4+/-6.6 | +0.0% | 480176 | 37748736 | 100.0% | 0.313 |
| V1_local_only | 279.5+/-7.0 | +12.5% | 224 | 9437184 | 25.0% | 0.303 |
| V6_ra_value_fb4 | 256.3+/-6.9 | +3.2% | 358 | 16035840 | 42.5% | 0.952 |
| V6_recency_fb8 | 256.8+/-7.0 | +3.4% | 493 | 22634496 | 60.0% | 0.940 |
| V6_ra_blend_fb8 | 256.8+/-7.0 | +3.4% | 493 | 22634496 | 60.0% | 0.938 |
| V6_ra_value_fb8 | 257.3+/-7.0 | +3.6% | 493 | 22634496 | 60.0% | 0.946 |
| V6_random_fb4 | 258.3+/-22.8 | +4.0% | 358 | 16035840 | 42.5% | 0.945 |
| V6_ra_blend_fb4 | 258.4+/-7.0 | +4.0% | 358 | 16035840 | 42.5% | 0.943 |
| V6_recency_fb4 | 258.5+/-7.0 | +4.1% | 358 | 16035840 | 42.5% | 0.943 |
| V6_random_fb8 | 258.6+/-22.5 | +4.1% | 493 | 22634496 | 60.0% | 0.958 |

## Results at L=2048

| Variant | PPL | PPL vs Dense | Kept Tokens | KV Read B/tok | FLOPs% | ms/tok |
|---------|-----|-------------|-------------|--------------|--------|--------|
| V0_dense | 261.4+/-11.5 | +0.0% | 560184 | 75497472 | 100.0% | 0.536 |
| V1_local_only | 310.4+/-12.7 | +18.7% | 240 | 9437184 | 12.5% | 0.524 |
| V6_ra_value_fb4 | 279.9+/-12.2 | +7.1% | 397 | 16041106 | 21.2% | 1.445 |
| V6_ra_value_fb8 | 280.8+/-12.2 | +7.4% | 554 | 22645029 | 30.0% | 1.452 |
| V6_random_fb4 | 283.5+/-11.8 | +8.4% | 397 | 16041106 | 21.2% | 1.450 |
| V6_ra_blend_fb8 | 283.5+/-12.8 | +8.5% | 554 | 22645029 | 30.0% | 1.457 |
| V6_recency_fb8 | 283.6+/-12.8 | +8.5% | 554 | 22645029 | 30.0% | 1.452 |
| V6_random_fb8 | 284.0+/-11.4 | +8.6% | 554 | 22645029 | 30.0% | 1.453 |
| V6_ra_blend_fb4 | 291.0+/-13.2 | +11.3% | 397 | 16041106 | 21.2% | 1.454 |
| V6_recency_fb4 | 291.4+/-13.2 | +11.5% | 397 | 16041106 | 21.2% | 1.446 |

## KV Cache Scaling Law

Fit: `KV_kept(L) = c * L^beta_eff` (log-log regression)

| Variant | beta_eff | c | R^2 |
|---------|---------|---|-----|
| V0_dense | 0.404 | 26928.8 | 0.937 |
| V1_local_only | 0.160 | 71.8 | 0.954 |
| V6_ra_blend_fb4 | 0.247 | 61.8 | 0.948 |
| V6_ra_blend_fb8 | 0.288 | 63.2 | 0.945 |
| V6_ra_value_fb4 | 0.247 | 61.8 | 0.948 |
| V6_ra_value_fb8 | 0.288 | 63.2 | 0.945 |
| V6_random_fb4 | 0.247 | 61.8 | 0.948 |
| V6_random_fb8 | 0.288 | 63.2 | 0.945 |
| V6_recency_fb4 | 0.247 | 61.8 | 0.948 |
| V6_recency_fb8 | 0.288 | 63.2 | 0.945 |

Dense scales as L^0.40 (expected ~1.0). Best RA variant (V6_ra_value_fb4) scales as L^0.25.

## Verdict

At L=1024: best RA_value (V6_ra_value_fb4) PPL=256.3 vs best recency (V6_recency_fb8) PPL=256.8 (delta=0.5)

At L=2048: RA_value (V6_ra_value_fb4) PPL=279.9 vs recency (V6_recency_fb8) PPL=283.6 (delta=3.7)

### Recommended BPA+RA Config

- Variant: V6_ra_value_fb4
- PPL: 256.3 (+3.2% vs dense)
- FLOPs: 42.5% of dense
- KV read: 16035840 B/token
