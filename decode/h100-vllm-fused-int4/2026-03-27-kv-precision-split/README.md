# H100 vLLM true K/V precision-split verification — partial results (2026-03-27)

## Why this artifact exists
This bundle captures the first real end-to-end H100 vLLM attempt to verify the
paper's Qwen classifier claim inside the fused serving path using explicit K/V
precision-split policies instead of the weaker zero-point-asymmetric INT4 proxy.

Specifically, this is the first bundle that stages/tests:
- `K_INT8 / V_INT4`
- `K_FP16 / V_INT4`

for Qwen-class 7B models inside the current vLLM fused INT4 lane.

## Source code state
Prune `vllm` branch:
- `20250325-fused-quantization`

Relevant code commits:
- `8d6d9cb90` — `feat: K/V precision-split policies for fused INT4 backend`
- `9dd514d8b` — `bench: add Qwen2-7B and Mistral H100 fused INT4 policy sweep scripts`

## Files in this bundle
- `kv_precision_split_manifest_20260327_151647.json`
- `qwen25_kv_precision_split_20260327_152043.json`
- `qwen25_kint8_retest_20260327_153203.json`
- `qwen2_kv_precision_split_20260327_151647.json`
- `qwen2_kint8_retest_20260327_153203.json`

## What is already real / established
The following statements are supported by concrete H100 JSON artifacts in this
bundle:

### 1. The true K/V split lane is no longer hypothetical
The backend work moved past theory and into an actual H100 testable code path.
The prune `vllm` tree now contains code and probes for explicit K/V precision
splits.

### 2. Both Qwen2.5-7B-Instruct and Qwen2-7B-Instruct were exercised
Artifacts exist for:
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2-7B-Instruct`

### 3. Reference baseline vs best current fused policy
Reference baseline for correctness:
- the ordinary non-fused FP16 decode path

Best current fused policy at the time of this partial bundle:
- `K_INT4 / V_INT4`
- `MSL=48`
- `7/7` exact matches on the tested prompt set against the FP16/non-fused reference

### 4. `K_FP16 / V_INT4` shows real promise for Qwen2
For `Qwen/Qwen2-7B-Instruct`, the current partial results show:
- `K_FP16 / V_INT4`, `MSL=1`  -> `6/7`
- `K_FP16 / V_INT4`, `MSL=8`  -> `6/7`
- `K_FP16 / V_INT4`, `MSL=24` -> `6/7`

This is meaningfully stronger than the old low-MSL all-INT4 behavior and is the
strongest early signal in this bundle that a true paper-style K/V split policy
may help.

## What is still partial / unresolved
### 1. The result is not yet a finished policy recommendation
The manifest explicitly marks both models as:
- `PARTIAL`

That is correct. The current JSONs are enough to prove the lane is live and that
there are non-trivial effects, but not enough to declare a final winner.

### 2. `K_INT8 / V_INT4` is not yet cleanly proven as the winning policy
For both Qwen models, `K_INT8 / V_INT4` at `MSL=48` reaches `7/7`, but that does
not beat the existing best fused policy `K_INT4 / V_INT4` + `MSL=48`.
Both are still being judged against the same non-fused FP16 reference path.

At lower MSL values, the current `K_INT8 / V_INT4` results are mixed / weaker
than hoped, so this bundle does not yet justify saying that `K_INT8 / V_INT4`
cleanly validates the paper claim in serving.

### 3. `K_FP16 / V_INT4` is more promising for Qwen2 than Qwen2.5, but not yet finished
Current partial signal:
- `Qwen2` shows `6/7` at low MSL with `K_FP16 / V_INT4`
- `Qwen2.5` is weaker in the current partial bundle:
  - `MSL=1`  -> `3/7`
  - `MSL=8`  -> `3/7`
  - `MSL=24` -> `4/7`

That model gap is exactly why another focused pass is needed.

### 4. We do not yet have a clean README-backed final conclusion for “Qwen2.5 vs Qwen2”
This bundle is the staging area for that conclusion, not the final statement.
The next pass should:
- clean up the retest matrix
- explain whether the difference is real or an artifact of the current sweep
- produce a final recommendation or a tighter blocker explanation

## Current best reading
### What this bundle already supports
- The paper's stronger K/V split question is now finally being tested inside
  vLLM, not just argued about.
- The current vLLM implementation can exercise true K/V split policies.
- `K_FP16 / V_INT4` looks more interesting than `K_INT8 / V_INT4` so far,
  especially for `Qwen2`.

### What it does not yet support
- It does **not yet** prove that the paper classifier result is fully validated
  end-to-end in vLLM.
- It does **not yet** prove that one split policy cleanly dominates for both
  `Qwen2.5-7B` and `Qwen2-7B`.

## Why one more pass is required
The remaining question is no longer “can vLLM stage true K/V split policies?”
That is already answered.

The remaining question is:
- why does `Qwen2` look materially better than `Qwen2.5` in the current partial
  `K_FP16 / V_INT4` results,
- and what is the clean final recommendation for each model?

That is the purpose of the next focused pass.
