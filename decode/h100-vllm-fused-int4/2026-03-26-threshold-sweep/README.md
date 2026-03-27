# H100 vLLM fused INT4 threshold sweep — 2026-03-26

## Goal
Test whether increasing `VLLM_FUSED_INT4_MIN_SEQ_LEN` reduces or eliminates the
post-fix long-context failures seen in the broad correctness probe.

## What value was bumped
The default fused threshold in the fixed H100 path was:
- `VLLM_FUSED_INT4_MIN_SEQ_LEN=8`

The threshold sweep tested:
- `32`
- `64`

The meaning of this threshold is:
- below the threshold, decode uses the FP16 shadow / fallback path
- at or above the threshold, decode switches to the fused INT4 path

## Results
### default min_seq_len = 8
- result: 11/13 PASS
- `len_32`: FAIL
- `len_64`: FAIL

### min_seq_len = 32
- result: 11/13 PASS
- `len_32`: FAIL
- `len_64`: FAIL
- interpretation: no meaningful improvement over the default threshold for
  these failing cases

### min_seq_len = 64
- result: 12/13 PASS
- `len_32`: PASS
- `len_64`: FAIL
- interpretation: protecting more early decode steps with the FP16 shadow
  window is enough to recover the `len_32` case, but not the `len_64` case

## What this means
The remaining divergence is at least partly a context-length / exposure issue.
Raising the FP16 shadow threshold helps, which supports the hypothesis that the
post-fix failures are driven by accumulated INT4 quality loss rather than the
original narrow short-sequence decode bug.

Just as important, this sweep shows that the problem is not solved by a single
blind fused-INT4 map. The threshold only moves the cliff; it does not eliminate
it for this model.

## Negative implications of bumping the threshold
Increasing `VLLM_FUSED_INT4_MIN_SEQ_LEN` has real costs:
- it keeps more decode steps on the FP16 shadow path
- it reduces the time spent on the fused INT4 path
- it gives back some of the memory-traffic and latency benefit we wanted from
  fused INT4 in the first place
- a larger shadow window increases temporary high-precision state pressure
- it can become a model-specific workaround rather than a principled deployment
  policy

In short: bumping the threshold is a useful diagnostic and sometimes a tactical
mitigation, but it is not the full policy answer.

## Was this overlooked because the ratio classifier was not used first?
Yes — that is the cleanest interpretation.

The testing so far proved that the fused backend and short-sequence correctness
fix are real, but it also proved that blindly applying a static fused INT4 map
to Qwen2.5-7B is the wrong deployment policy.

The ratio classifier exists to prevent exactly this mistake:
- classify the model and regime first
- determine whether fused INT4 is safe
- then choose the mapping / threshold / fallback policy for that model

## Recommended policy direction
Do not treat fused INT4 as a universal default.

Instead:
1. run the ratio classifier for the target model / inference regime
2. derive the ideal fused quantization mapping from that result
3. choose the model-specific threshold / fallback behavior from the classifier
4. only then benchmark or deploy the fused path

For Qwen-like sensitive models, the ratio classifier should drive whether we:
- use fused INT4 at all,
- widen the FP16 shadow window,
- or select a different K/V precision policy entirely.
