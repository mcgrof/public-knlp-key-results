# H100 vLLM true K/V precision-split final retest — Qwen2 vs Qwen2.5

## Why this artifact exists
This bundle is the focused last-mile retest for the true K/V precision-split
lane. It resolves the earlier ambiguity about whether `K_FP16 / V_INT4` looked
better for `Qwen/Qwen2-7B-Instruct` than for `Qwen/Qwen2.5-7B-Instruct`, and it
extends the prompt set from 7 prompts to 12 prompts.

## Source code state
Prune `vllm` commits relevant to this retest:
- `8d6d9cb90` — `feat: K/V precision-split policies for fused INT4 backend`
- `9cca26180` — `bench: add K/V precision-split final retest script (12 prompts)`

## Files in this bundle
- `final_retest_manifest_20260327_163906.json`
- `qwen2_final_retest_20260327_163906.json`
- `qwen25_final_retest_20260327_164338.json`

## What was tested
For both models:
- `K_FP16 / V_INT4` at `MSL=8` (two repetitions for determinism)
- `K_INT8 / V_INT4` at `MSL=8`
- `K_FP16 / V_INT4` at `MSL=24`
- `K_INT8 / V_INT4` at `MSL=24`
- `K_FP16 / V_INT4` at `MSL=48`

Prompt set:
- 12 prompts total
- 7 original prompts
- 5 extra prompts added for the final retest

## Final scores
### Qwen/Qwen2-7B-Instruct
- `K_FP16 / V_INT4`, `MSL=8`, rep1 -> `8/12`
- `K_FP16 / V_INT4`, `MSL=8`, rep2 -> `8/12`
- `K_INT8 / V_INT4`, `MSL=8`      -> `5/12`
- `K_FP16 / V_INT4`, `MSL=24`     -> `9/12`
- `K_INT8 / V_INT4`, `MSL=24`     -> `7/12`
- `K_FP16 / V_INT4`, `MSL=48`     -> `12/12`

### Qwen/Qwen2.5-7B-Instruct
- `K_FP16 / V_INT4`, `MSL=8`, rep1 -> `8/12`
- `K_FP16 / V_INT4`, `MSL=8`, rep2 -> `8/12`
- `K_INT8 / V_INT4`, `MSL=8`       -> `8/12`
- `K_FP16 / V_INT4`, `MSL=24`      -> `9/12`
- `K_INT8 / V_INT4`, `MSL=24`      -> `10/12`
- `K_FP16 / V_INT4`, `MSL=48`      -> `12/12`

## What is real / established now
### Reference baseline vs fused-policy comparison
Reference baseline for correctness throughout this retest:
- the ordinary non-fused FP16 decode path

Current best fused policy used as the practical comparator:
- symmetric `K_INT4 / V_INT4`
- `VLLM_FUSED_INT4_MIN_SEQ_LEN=48`

The split-policy scores below should be read as: how well does this fused
variant match the non-fused FP16 reference, and does it beat the current best
fused policy enough to replace it?

### 1. The Qwen2 vs Qwen2.5 difference is real, but narrower than it first looked
The earlier partial bundle made `Qwen2` look much better than `Qwen2.5` under
`K_FP16 / V_INT4`. The 12-prompt retest shows both models land at the same
headline score for `K_FP16 / V_INT4` at `MSL=8`:
- `8/12` for Qwen2
- `8/12` for Qwen2.5

So the strongest apparent model gap from the partial bundle was mostly a
small-sample artifact.

### 2. Qwen2.5 currently responds better than Qwen2 to `K_INT8 / V_INT4`
At `MSL=24`:
- Qwen2      -> `7/12`
- Qwen2.5    -> `10/12`

At `MSL=8`:
- Qwen2      -> `5/12`
- Qwen2.5    -> `8/12`

So the current implementation gives a materially stronger `K_INT8 / V_INT4`
signal for Qwen2.5 than for Qwen2.

### 3. Neither split policy beats the current best fused policy cleanly
The current best fused policy remains:
- symmetric `K_INT4 / V_INT4`
- `VLLM_FUSED_INT4_MIN_SEQ_LEN=48`
- exact-match envelope already validated for both Qwen models against the
  non-fused FP16 reference baseline

The final retest shows:
- `K_FP16 / V_INT4` can improve low-MSL behavior relative to naive all-INT4
- `K_INT8 / V_INT4` can improve Qwen2.5 more than Qwen2
- but neither split policy yet gives a clean reason to replace the current
  best fused `MSL=48` policy as the default serving recommendation

### 4. Determinism for the key Qwen2 FP16-split case looks stable
For `Qwen2`, `K_FP16 / V_INT4`, `MSL=8`, the two repeated runs matched exactly:
- `8/12` both times
- same failing prompts: `med_32`, `short_12`, `long_80`, `long_112`

That means the behavior is not random noise from a flaky one-off run.

## Final practical recommendation
### Qwen/Qwen2.5-7B-Instruct
- true K/V split policies show real signal
- `K_INT8 / V_INT4` is more promising here than on Qwen2
- but the current safest practical policy is still:
  - **symmetric `K_INT4 / V_INT4` with `MSL=48`**

### Qwen/Qwen2-7B-Instruct
- `K_FP16 / V_INT4` improves low-MSL behavior more than `K_INT8 / V_INT4`
- but the current safest practical policy is still:
  - **symmetric `K_INT4 / V_INT4` with `MSL=48`**

## Bottom line
The true K/V split enhancement is now real and tested.
It does produce meaningful model-specific behavior.
However, with the current implementation and tested prompt set, the split
policies are not yet strong enough to replace the existing `MSL=48` baseline as
the default recommendation for Qwen serving on H100.
