# H100 vLLM fused INT4 broad correctness probe — 2026-03-26

## Summary

This artifact captures the broader post-fix H100 correctness probe run after the
short-sequence decode bug was fixed.

## Source state
- prune source commit: `d72dbf120`
- H100 editable tree content synced to that fixed source state
- model: `Qwen/Qwen2.5-7B-Instruct`
- probe: `benchmarks/fused_int4_broad_probe.py`

## Outcome
- total prompt lengths tested: 13
- exact token match passes: 11
- failures: 2
- failing prompt lengths: `32`, `64`

## Interpretation
The narrow decode integration bug is fixed.
The remaining divergence appears only in the longer prompt regimes tested by
this broad probe and is currently treated as an INT4 quality/regime limit rather
than the original short-sequence serving-path correctness bug.

## Key result
- `len_1` through `len_17` all match baseline in this probe set
- `len_32` and `len_64` diverge with fused output producing repeated
  `pérdida` tokens where baseline remains stable

## Related artifacts
- smoke-fix bundle:
  `../2026-03-26-smoke-fix/`
