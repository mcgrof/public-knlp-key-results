# Qwen classifier vs. current H100 vLLM results — reconciliation note

## The question
Why does the fused-quantization paper say Qwen2.5-7B / Qwen2-7B are strong
key/value asymmetry cases, while the current H100 vLLM sweep says the best
policy for `Qwen/Qwen2.5-7B-Instruct` is still a **symmetric** INT4 path with
`VLLM_FUSED_INT4_MIN_SEQ_LEN=48` and that the tested "asymmetric" setting did
not win?

## Short answer
These are **not the same experiment** and **not the same policy knob**.
There is not a direct contradiction yet.

## What the paper classifier actually says
From `paper-fim/quantization.tex`:
- `Qwen2.5-7B` and `Qwen2-7B` are **classifier-positive** models
- they show a severe **key/value sensitivity asymmetry**
- the practical interpretation is: **keys need much higher precision than values**
- in the paper, the relevant asymmetric policy family is things like:
  - `K_FP16 / V_INT4`
  - `K_INT8 / V_INT4`
  - or, more generally, "do not quantize K as aggressively as V"

This is a **K-precision-floor statement**.
It is not merely a statement about zero-point quantization or group size.

## What the current vLLM H100 sweep actually tested
The H100 vLLM policy sweep for `Qwen/Qwen2.5-7B-Instruct` tested:
- symmetric INT4 vs asymmetric INT4
- `GROUP_SIZE=32` vs `GROUP_SIZE=16`
- different `VLLM_FUSED_INT4_MIN_SEQ_LEN` values

In the current vLLM backend, the tested "asymmetric" knob means:
- **min/max range + zero-point offset** for INT4 quantization
- still using the same basic INT4 bitwidth policy for both K and V

That is **not the same** as a true K/V precision split such as:
- `K_INT8 / V_INT4`
- `K_FP16 / V_INT4`

So the vLLM sweep did **not** directly test the paper's main asymmetric K/V
claim inside the fused backend.

## What "zero-point asymmetric INT4" actually means
This needs to be stated plainly because the overloaded word "asymmetric" is the
main source of confusion.

### Symmetric INT4
The quantizer assumes values are roughly centered around zero.
A scale factor is chosen from the absolute range, and packed INT4 values are
interpreted around a fixed signed center.

In practical terms:
- both positive and negative range are treated symmetrically
- dequantization is conceptually like: `(q - center) * scale`
- this is simple and common when the tensor distribution is close to zero-mean

### Zero-point asymmetric INT4
The quantizer uses a min/max range and stores a learned / computed **zero-point**
offset for each group.

In practical terms:
- the quantizer no longer assumes the group is centered near zero
- dequantization is conceptually like: `q * scale + zero_point`
- this can help when the value distribution is shifted or skewed
- in the current fused INT4 backend, this is controlled by:
  - `VLLM_FUSED_INT4_ASYMMETRIC=1`

### Why this is easy to confuse with the paper
The paper's stronger claim is about **K and V needing different precision
levels**. That is a tensor-level policy question.

The current vLLM `ASYMMETRIC=1` knob is only a **quantizer-formula question**:
- same INT4 bitwidth for K and V
- same overall serving path
- different quantizer parameterization (scale + zero-point instead of symmetric scale)

So:
- paper asymmetry = **K/V precision asymmetry**
- current vLLM asymmetry = **zero-point asymmetric INT4 quantization**

## What the vLLM result really means
For `Qwen/Qwen2.5-7B-Instruct` on H100, under the **currently implemented vLLM
fused INT4 backend**:
- correctness is judged against the ordinary **non-fused FP16 decode path**
- enabling INT4 min/max zero-point quantization (`ASYMMETRIC=1`) did not beat
  the symmetric INT4 path inside the fused family
- reducing `GROUP_SIZE` from 32 to 16 did not help either
- the strongest working policy **within the fused family** was:
  - symmetric INT4
  - `GROUP_SIZE=32`
  - `VLLM_FUSED_INT4_MIN_SEQ_LEN=48`

This means:
> the currently implemented vLLM asymmetric-quantizer knob is **not** enough to
recover Qwen's key-sensitivity problem.

It does **not** mean:
> the paper's K/V asymmetry finding was wrong.

## Why this is not a contradiction
The paper result and the current vLLM result are operating on different axes:

### Paper / classifier axis
- **Which tensor needs more precision?**
- Answer for Qwen: **K needs much more precision than V**

### Current vLLM H100 sweep axis
- **Given the current fused INT4 backend, do zero-point asymmetry, group size,
  and delayed fused entry improve exact-text correctness?**
- Answer for Qwen2.5-7B-Instruct: **yes, delayed fused entry helps**;
  the tested zero-point asymmetry does not win

So the most accurate interpretation is:
- the paper says **use asymmetric K/V precision policy** for Qwen-like models
- the current vLLM backend does **not yet expose that full K/V policy family**
- therefore the H100 vLLM sweep is only a **partial implementation test** of
  the paper's broader classifier conclusion

## What we did verify in vLLM
We did verify these concrete H100 facts:

### Qwen2.5-7B-Instruct
- best current working policy: `MSL=48`
- tested zero-point asymmetry did not beat symmetric
- `GROUP_SIZE=16` did not beat `32`

### Mistral-7B-Instruct-v0.3
- best current working policy: `MSL=56`
- this does **not** mean Mistral contradicts the classifier
- it means the current vLLM fused path's **MSL requirement** for exact-text
  stability is higher for this model under this implementation

## Important subtlety: MSL is not the classifier ratio
The paper's classifier ratio and the vLLM `MIN_SEQ_LEN` are measuring different
things:

### Ratio classifier
- detects whether the model has a precision cliff
- especially whether K requires much higher precision than V

### `VLLM_FUSED_INT4_MIN_SEQ_LEN`
- decides how long decode stays on the safer FP16 shadow / paged path before
  switching into the fused INT4 path

A model needing a higher MSL in the current vLLM backend does **not** imply it
has a higher K/V asymmetry ratio in the paper sense.

## How `paper-memory-decode` fits into this
`paper-memory-decode` mostly expresses the same model-specific sensitivity story
through **adaptive per-layer INT8 vs INT4** overlays, not primarily through a
K/V split.

That repo still agrees qualitatively with the paper-classifier story:
- Qwen is more sensitive and needs more protection
- Mistral is more tolerant

But it operationalizes the policy differently:
- per-layer INT8 overlays
- not a direct K_INT8 / V_INT4 split inside vLLM fused decode

## What would count as a real end-to-end verification of the classifier in vLLM?
A stronger verification would require implementing and testing one of these in
vLLM fused decode itself:
- `K_INT8 / V_INT4`
- `K_FP16 / V_INT4`
- another explicit model-specific K/V precision split chosen by the classifier

Only then could we say we directly tested the paper's asymmetric K/V claim in
the serving path.

## Is this a bug in our vLLM implementation, or a limitation of what we support?
### Best current answer: **mainly a limitation, not the core bug**
For this specific confusion about the word "asymmetric", the issue is best
understood as a **limitation of the currently exposed vLLM policy space**.

Why:
- the backend does implement a real zero-point-asymmetric INT4 mode
- the code allocates and uses separate zero-point tensors (`k_zeros`, `v_zeros`)
- the Triton cache-write and decode paths both carry an `IS_ASYMMETRIC` branch
- so the current `ASYMMETRIC=1` mode is not just a placeholder label

But the bigger limitation is:
- the backend does **not yet expose a true K/V precision split policy** like
  `K_INT8 / V_INT4` or `K_FP16 / V_INT4`
- therefore the paper's main Qwen classifier finding cannot yet be tested
  end-to-end in the serving path using the currently implemented knobs alone

### Where there may still be implementation bugs
This does **not** prove the whole fused backend is bug-free.
Earlier H100 work already showed there were real integration/correctness issues
that had to be fixed. But for the narrow question here — why paper asymmetry and
current vLLM asymmetry do not line up — the primary problem is that the two
systems are testing **different kinds of asymmetry**.

## Bottom line
There is **not yet** a direct contradiction between the paper classifier and the
current H100 vLLM results.

What we have today is:
1. the paper says Qwen-class 7B models need **asymmetric K/V precision policy**
2. the current vLLM H100 backend was only tested with:
   - symmetric vs zero-point-asymmetric INT4
   - group-size changes
   - fused-entry threshold changes
3. that narrower vLLM test found the best current operational policy to be:
   - symmetric INT4
   - `GROUP_SIZE=32`
   - `VLLM_FUSED_INT4_MIN_SEQ_LEN=48`
4. the paper's stronger K/V asymmetry claim is therefore still **only partially
   instantiated** in the current vLLM backend

## Practical next step
If we want to verify the paper's classifier result *inside vLLM*, the next real
experiment is not another zero-point asymmetry sweep. It is a true K/V split
policy test such as:
- `K_INT8 / V_INT4`
- or `K_FP16 / V_INT4`

That would finally test whether the paper's Qwen asymmetry result survives all
the way through the live fused serving path.
