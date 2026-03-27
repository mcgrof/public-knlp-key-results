# Qwen2-7B H100 follow-up sweep — 2026-03-27

## Why this artifact exists
This bundle captures the missing follow-up H100 verification for whether the
current vLLM fused INT4 backend behaves for `Qwen/Qwen2-7B-Instruct` the same
way it did for `Qwen/Qwen2.5-7B-Instruct`.

The goal was not yet to test true K/V split precision. The goal here was to
answer a narrower question:
- if we run the current vLLM backend on Qwen2-7B,
- does the classifier-style result transfer,
- and does the currently implemented zero-point-asymmetric INT4 mode help?

## Model tested
- `Qwen/Qwen2-7B-Instruct`

## Main result
The current vLLM result transfers qualitatively from Qwen2.5-7B to Qwen2-7B:
- Qwen2-7B is highly sensitive at low fused thresholds
- the lowest perfect MSL in this sweep was **48**
- the currently implemented zero-point-asymmetric INT4 mode did **not** produce
  a decisive win over symmetric INT4

## Fine MSL sweep
From `qwen2_msl_fine_sweep_20260327_141701.json`:
- `MSL=8`  -> `0/7`
- `MSL=16` -> `0/7`
- `MSL=24` -> `0/7`
- `MSL=32` -> `2/7`
- `MSL=40` -> `4/7`
- `MSL=48` -> `7/7`
- `MSL=56` -> `7/7`
- `MSL=64` -> `7/7`

## Asymmetric sweep
From `qwen2_asym_sweep_20260327_141701.json`:
- at `MSL=32`, all symmetric/asymmetric variants scored `2/7`
- at `MSL=40`, the best variant was `asym_g16_msl40` at `5/7`, but this was not
  enough to change the overall policy outcome
- at `MSL=48`, **all four variants** reached `7/7`

## Interpretation
This bundle supports the same reconciliation conclusion reached for Qwen2.5-7B:
- the current vLLM zero-point-asymmetric INT4 knob is not the same as a true
  K/V precision split policy such as `K_INT8/V_INT4` or `K_FP16/V_INT4`
- in the current backend, the strongest operational lever remains the fused
  entry threshold (`VLLM_FUSED_INT4_MIN_SEQ_LEN`)
- for both Qwen2-7B and Qwen2.5-7B, the current H100 sweep settled at an MSL of
  **48** for perfect match on the tested prompt set

## What this does and does not answer
### Answered
- Does Qwen2-7B behave broadly like Qwen2.5-7B in the current vLLM backend? **Yes.**
- Does current zero-point-asymmetric INT4 clearly beat symmetric INT4? **No.**

### Not answered
- Does a true paper-style K/V split policy (`K_INT8/V_INT4` or `K_FP16/V_INT4`)
  work better? **Not yet.** That remains the next real experiment.
