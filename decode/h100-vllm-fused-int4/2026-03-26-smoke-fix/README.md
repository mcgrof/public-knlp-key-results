# H100 vLLM fused INT4 smoke fix — 2026-03-26

## Summary

This artifact bundle captures the first clean H100 smoke-test confirmation
that the narrow fused INT4 decode correctness bug was fixed on the live H100
editable vLLM tree.

## Source repo state used for the fix

Prune repo:
- `/data/vllm`
- branch: `20250325-fused-quantization`
- final source commit: `d72dbf120`

Relevant commit chain after the earlier minimal build-profile checkpoint:
- `11c6a73bb` — feat: decode-boundary probe + VLLM_FUSED_INT4_DEBUG instrumentation
- `b7fb86d21` — feat: Phase 1+2 decode-boundary probe and enhanced debug instrumentation
- `ef2fa98ef` — fix: guard fused INT4 forward against None attn_metadata during warmup
- `46eea0729` — fix: make fused INT4 cache shape compatible with FP16 allocator
- `6c651e03f` — fix: reshape prefill fallback output to match caller buffer shape
- `d72dbf120` — fix: FP16 shadow buffer for short-sequence INT4 decode correctness

## What was actually fixed

The original narrow reproducer was on the H100 fused path:
- `prompt_len=2`
- `max_tokens=2`
- fused previously diverged on the second generated token

The final fix introduced an FP16 shadow-buffer fallback for short-sequence
INT4 decode correctness while preserving the fused Triton decode kernel for the
intended decode regime.

## H100 smoke result

Artifact:
- `smoke-decode-boundary-d72dbf120.json`

Outcome:
- token 1 match: 5/5
- token 2 match: 5/5
- all match: 5/5
- status: PASS

Key formerly failing case now matches baseline:
- `prompt_len=2`
- baseline: `and dirty`
- fused: `and dirty`

## H100 runtime/backend confirmation

During the smoke run:
- baseline selected `FLASH_ATTN`
- fused selected `FUSED_INT4`
- fused backend verification reported:
  - `decode_kernel=fused_int4_triton`
  - `fallback=fp16_shadow_sdpa`
  - `min_fused_seq_len=8`

## Important deployment note

The live H100 pod uses an editable vLLM install rooted at:
- `/root/vllm-fused-git`

The pod git history was not normalized to the prune commit graph; instead, the
final fixed file content from prune commit `d72dbf120` was synced into the
editable H100 tree so the runtime code path matches the fixed source.

The accompanying manifest records the packaging assumptions for the exported
editable tarball.
