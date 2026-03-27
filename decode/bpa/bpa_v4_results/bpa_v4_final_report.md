# BPA v4: High-Recall Gate + Sparse Cache Results

## Model
- GPT2_RGSA (124M params), FineWebEdu, 615 iters
- Config: n_layer=12, n_head=12, n_embd=768
- local_window=256, chunk_size=64, top_b=8
- Evaluation: FineWebEdu val.bin, bf16 KV accounting
- Seeds: {1,2,3}, 50 eval batches per run

## Changes from v3
- Gate trained with weighted BCE (pos_weight*10) for high recall
- Budget-calibrated thresholding (quantile-based)
- Sweep across 30%/50%/70%/90% enabled_rate budgets
- Sparse KV cache: compacted far-context storage

## Results at L=512

| Variant | PPL | PPL vs Dense | Enabled Rate | KV Savings | FLOPs% | ms/tok |
|---------|-----|-------------|-------------- |------------|--------|--------|
| V0_dense | 245.5+/-8.2 | 0.0% | 1.000 | 0.0% | 100.0% | 0.243 |
| V1_local_only | 245.5+/-8.4 | -0.0% | 0.000 | 50.0% | 50.0% | 0.320 |
| V2_budget_30 | 246.6+/-8.2 | +0.4% | 0.301 | 35.0% | 65.0% | 0.761 |
| V2_budget_50 | 246.5+/-8.2 | +0.4% | 0.500 | 25.0% | 75.0% | 0.774 |
| V2_budget_70 | 246.0+/-8.3 | +0.2% | 0.699 | 15.0% | 85.0% | 0.586 |
| V2_budget_90 | 245.5+/-8.2 | +0.0% | 0.898 | 5.1% | 94.9% | 0.586 |
| V3_v3gate | 247.7+/-8.4 | +0.9% | 0.453 | 27.4% | 72.6% | 0.584 |
| VR_random | 245.9+/-6.4 | +0.2% | 0.503 | 24.8% | 75.2% | 0.242 |

## Results at L=1024

| Variant | PPL | PPL vs Dense | Enabled Rate | KV Savings | FLOPs% | ms/tok |
|---------|-----|-------------|-------------- |------------|--------|--------|
| V0_dense | 244.8+/-9.8 | 0.0% | 1.000 | 0.0% | 100.0% | 0.337 |
| V1_local_only | 275.9+/-11.7 | +12.7% | 0.000 | 75.0% | 25.0% | 0.341 |
| V2_budget_30 | 270.9+/-11.3 | +10.6% | 0.301 | 52.4% | 47.6% | 0.839 |
| V2_budget_50 | 268.2+/-11.1 | +9.6% | 0.500 | 37.5% | 62.5% | 0.839 |
| V2_budget_70 | 255.9+/-10.3 | +4.5% | 0.699 | 22.6% | 77.4% | 0.838 |
| V2_budget_90 | 246.2+/-9.8 | +0.6% | 0.900 | 7.5% | 92.5% | 0.838 |
| V3_v3gate | 272.4+/-11.4 | +11.3% | 0.289 | 53.3% | 46.7% | 0.840 |
| VR_random | 277.7+/-7.5 | +13.4% | 0.500 | 37.5% | 62.5% | 0.338 |

## Sparse Cache Footprint Reduction

With compacted KV cache (local window dense + far-context on demand):

| L | Budget | Dense KB | Sparse KB | Reduction | PPL reg | Sanity |
|---|--------|---------|-----------|---------- |---------|--------|
| 512 | 30% | 36864 | 11988 | 67.5% | +0.6% | OK |
| 512 | 50% | 36864 | 13824 | 62.5% | +0.6% | OK |
| 512 | 70% | 36864 | 15660 | 57.5% | +0.3% | OK |
| 512 | 90% | 36864 | 17496 | 52.5% | -0.0% | OK |
| 1024 | 30% | 73728 | 17532 | 76.2% | +11.8% | OK |
| 1024 | 50% | 73728 | 23040 | 68.8% | +10.5% | OK |
| 1024 | 70% | 73728 | 28548 | 61.3% | +5.2% | OK |
| 1024 | 90% | 73728 | 34092 | 53.8% | +0.7% | OK |

Sanity checks: all-on gate reproduces dense attention PPL; all-off gate reproduces local-only PPL.

## Gate Quality

V4 gate (wbce_10x, 256-dim 3-layer MLP, AUC=0.90):
- Recall@30% budget: 0.75
- Recall@50% budget: 0.94
- Recall@60% budget: 0.98
- Recall@70% budget: 0.99

The gate achieves high recall at sufficient budget but the fundamental constraint is that far-context is genuinely important at L=1024 for most positions. No gate can skip far-context cheaply without quality loss.

## Wall Time Analysis

All gated variants (V2_*) are 2-3x slower than V0/V1 due to gate feature extraction overhead. VR_random runs at near-dense speed because it skips feature extraction.

Dense attention kernels do not skip masked-out positions, so attention masking provides no wall-time savings. True time savings require sparse attention kernels.

## Verdict

### Criterion A: Gate quality at L=1024

**NOT MET** simultaneously: no budget achieves both <=1% PPL regression AND >=25% KV savings at L=1024.

Best tradeoffs at L=1024:
- V2_budget_90: PPL +0.6%, KV savings 7.5%
- V2_budget_70: PPL +4.5%, KV savings 22.6%
- V2_budget_50: PPL +9.6%, KV savings 37.5%
- V2_budget_30: PPL +10.6%, KV savings 52.4%

### Criterion A at L=512

**MET**: V2_budget_30 at L=512: PPL +0.4%, KV savings 35.0%.
**MET**: V2_budget_50 at L=512: PPL +0.4%, KV savings 25.0%.

### Criterion B: Real footprint reduction

**MET**: At L=1024 with 90% budget, sparse cache reduces peak KV from 73728 KB to 34092 KB (53.8% reduction) with PPL regression +0.7%.

### Criterion C: Reporting artifacts

All required artifacts generated:
- bpa_v4_results/bpa_v4_final_report.md
- bpa_v4_results/raw_results.json
- bpa_v4_results/sparse_cache_results.json
- bpa_v4_results/plots/ (4 plots)

### Key finding: gate is WORSE than random at L=1024

At L=1024 with 50% enabled rate: V2 learned gate PPL=268.2 vs VR random PPL=277.7. 
The learned gate, trained on L=512 boundary_pressure labels, does not generalize to L=1024. It selects the wrong positions for far-context, performing worse than random selection. This is the core problem: features extracted from L=512 local attention patterns carry different information at L=1024 where far-context structure changes fundamentally.

### Next steps (v5)

1. Collect gate training data at L=1024 (not L=512)
2. Train gate end-to-end with the language model
3. Use position-dependent features that generalize across L
4. Implement sparse attention kernels for wall-time savings
5. Consider per-layer gating (different layers may need different budgets)
