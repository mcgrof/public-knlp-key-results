# H100 vLLM fused INT4 policy sweep — 2026-03-27

## Goal
Derive a usable model-specific fused INT4 policy for Qwen/Qwen2.5-7B-Instruct on
H100 instead of blindly applying a static map.

## What was tested
- quantization policies:
  - symmetric, GROUP_SIZE=32
  - symmetric, GROUP_SIZE=16
  - asymmetric, GROUP_SIZE=32
  - asymmetric, GROUP_SIZE=16
- min fused sequence length (MSL) values:
  - 16, 24, 32, 40, 48, 56, 64, 80, 96

## Main result
Reference baseline for correctness in this sweep:
- the ordinary non-fused FP16 decode path

Best current policy **within the fused family** for Qwen/Qwen2.5-7B-Instruct on
H100:
- `VLLM_FUSED_INT4_MIN_SEQ_LEN=48`
- keep the current symmetric INT4 path with `GROUP_SIZE=32`

## Why 48
- `MSL=48` achieved perfect text match on the tested sweep set
- `MSL=40` still failed at longer prompts
- `MSL=64` was safe but more conservative than needed

## Why this matters
Using `48` instead of `64` reduces the FP16 shadow / fallback window by:
- **16 decode positions per sequence**
- **25% less protected FP16 window** relative to 64

That means the engine enters the fused INT4 path earlier while preserving the
validated Qwen/H100 correctness envelope from this sweep.

## What did not help
- asymmetric quantization did not improve correctness
- GROUP_SIZE=16 did not improve correctness

## Policy interpretation
This result reinforces the deployment rule:
- do not use a blind static fused INT4 map
- calibrate the model/regime first
- then choose the model-specific threshold / mapping

For Qwen2.5-7B-Instruct on H100, the effective classifier-style policy outcome
from this sweep is `MSL=48`.
