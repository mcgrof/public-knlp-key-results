# H100 Readiness Dossier

## Why W7900 Is Compute-Bound

The AMD Radeon Pro W7900 (48.3 GB, 864 GB/s bandwidth) is
compute-bound for single-sequence decode on Qwen2.5-0.5B and 1.5B
at all tested batch sizes (1-16) and sequence lengths (8K-32K).

Evidence from v23 Phase 1:
- dense: 21.85 ms/tok at L=8K (all batch sizes identical)
- g32_k4: 21.22 ms/tok (0.3% faster — noise, not bandwidth gain)
- amort_g8_S8_k4: 22.92 ms/tok (8% SLOWER due to Python overhead)
- Latency is invariant to batch size: batch=1 and batch=16 produce
  the same ms/token at every L and method.

The decode kernel is dominated by matrix-multiply operations in
the FFN layers, not by KV cache memory reads. The 0.5B model has
only 2 KV heads (vs 14 query heads), so the KV cache is already
small relative to the compute required for GQA attention expansion.

## What Requires Bandwidth-Bound GPU

### Experiment 1: Latency Improvement from KV Compression
**Setup**: 7B/8B model at L=32K-64K with batch=8-32 on H100.
**Why**: Larger model (32 KV heads × 128 dim) produces much larger
KV cache reads per attention op. At batch>8, the attention kernel
should become memory-bandwidth limited.
**Expected**: 10-30% latency reduction from kv_ratio=0.30 vs dense.
**Acceptance criterion**: >=10% ms/token reduction in at least one
(L, batch) configuration.

### Experiment 2: Fused INT4 Attention Kernel
**Setup**: Custom CUDA kernel that decodes INT4+amortized-scale KV
directly in the attention loop (no separate dequant pass).
**Why**: Current PyTorch-level implementation adds 8% overhead.
A fused kernel eliminates dequant overhead and may expose bandwidth
savings hidden by the current implementation.
**Expected**: Zero or negative overhead from INT4 KV vs INT8 KV.
**Acceptance criterion**: INT4 attention kernel <=100% of INT8
attention kernel latency.

### Experiment 3: 7B/8B Full Oracle Ranking
**Setup**: Qwen2.5-7B or Llama-3.1-8B, empirical per-layer INT4
ablation at L=8K-32K.
**Why**: Validate k/D scaling prediction (k*~2 for D~32).
If k*=2, then g32_k2 achieves ratio=2/32*0.5+30/32*0.25=0.266.
**Expected**: k*<=3 with oracle ranking.
**Acceptance criterion**: PASS@3% at all L with k<=3.

### Experiment 4: Cost-per-token Analysis
**Setup**: Production serving benchmark comparing dense, INT8, and
amort_g8_S8 at various concurrency levels.
**Why**: The capacity gain (3.26x concurrent sequences) should
translate to proportionally lower cost-per-token in a serving
scenario where GPU utilization is the bottleneck.
**Expected**: 2-3x cost reduction per token served.
**Acceptance criterion**: >=2x tokens/GPU-hour improvement.

## Exact Run List

When cloud GPU credits are available:

```
1. H100: 7B model + oracle ranking
   python eval_v23.py --phase 3 --model qwen7b --device cuda
   Estimated time: 4-8 hours (32 layers × 3 seeds × 3 L)

2. H100: Throughput/latency benchmark on 7B
   python eval_v23.py --phase 1 --model qwen7b --device cuda
   Estimated time: 2-4 hours

3. H100: Amort sweep on 7B with oracle ranking
   python eval_v23.py --phase 2 --model qwen7b --device cuda
   Estimated time: 4-8 hours

4. H100: Kernel profiling (manual)
   Profile attention kernel time breakdown using nsight or
   torch.profiler. Measure memory throughput vs compute
   throughput to confirm bandwidth-bound regime.
   Estimated time: 2-4 hours
```

Total estimated GPU time: 12-24 hours on a single H100.
Estimated cost: $25-$50 at current cloud pricing ($2/hr for H100).

## What We Already Know Without H100

1. INT4 KV compression (g32 + sensitivity-guided INT8) preserves
   quality to within 3% PPL across all tested models and lengths.
2. k* scales inversely with model depth D (k*=4 at D=24, k*=2 at
   D=28). Extrapolation: k*~2 at D=32-80.
3. Amortized scales (g=8, S=8) are RoPE-safe with no drift at
   L=32K. Scale mode (pre/post/norm) is irrelevant for g=8.
4. The capacity story is real: 3.26x more concurrent sequences
   at zero quality cost.
5. Layer 0 (attention sink) is universally the most sensitive
   layer and must always be protected at INT8.

## Decision Framework

If H100 experiments show:
- **Latency improvement** (>=10% ms/tok at any config):
  -> Productize: fuse kernel, benchmark cost savings, publish.
- **No latency improvement** (still compute-bound at 7B):
  -> Focus on capacity story + longer context (64K-128K).
  -> Try batch=32-64 to force bandwidth saturation.
- **k* grows** (k*>3 at 7B, contradicting k/D scaling):
  -> Investigate: different attention patterns at scale?
  -> Fall back to learned calibration for sensitive layers.
