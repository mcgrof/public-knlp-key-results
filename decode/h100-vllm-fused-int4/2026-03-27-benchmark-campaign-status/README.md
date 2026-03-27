# H100 fused-vLLM benchmark campaign status — 2026-03-27

This note records the current campaign state now that the H100 fused-vLLM lane
is functional again. The immediate goal is no longer bring-up, correctness
triage, or policy debugging. The new goal is to execute the designed inference
benchmark suite from `knlp/docs/fused_kv_benchmark_runbook.md` and drive the
campaign toward full completion.

## What is already done
The H100 lane is no longer blocked on basic fused-vLLM functionality.
The following pre-benchmark engineering work is already complete:

- fused vLLM bring-up on H100
- short-sequence correctness bug fix
- broad correctness probes
- threshold sweeps
- Qwen policy sweep
- Mistral policy sweep
- true K/V precision-split implementation and retest work

Those results live in sibling directories under
`h100-vllm-fused-int4/` and should be treated as enabling evidence, not as the
main benchmark campaign itself.

## What the official inference benchmark suite is
The designed inference suite is documented in:
- `knlp/docs/fused_kv_benchmark_runbook.md`
- `knlp/docs/benchmarks/README.md`
- `knlp/docs/benchmarks/quickstart.md`
- `knlp/docs/benchmarks/reproducibility.md`
- `knlp/docs/benchmarks/smoke-test.md`

The runbook phases are:
1. Startup time
2. lm-eval (standard)
3. Latency sanity
4. NIAH
5. Full latency sweep
6. Throughput
7. GuideLLM sweep
8. Serving rate sweep
9. RULER
10. LongBench
11. InfiniteBench

## Current completion state
### Strict campaign accounting
Completed as a proper archived paired FP16-vs-FUSED benchmark campaign on H100:
- **0 / 11 phases**

Remaining:
- **11 / 11 phases**

### Why strict progress is still 0/11
The work completed so far was necessary H100 enablement and validation, but it
was not yet the official runbook campaign with the required paired artifacts,
backend manifests, and archived benchmark outputs.

## Immediate goal
The new goal with the fixed H100 vLLM is:
- start the real paired FP16-vs-FUSED inference benchmark campaign
- first complete the minimal respectable subset
- then drive to full 11/11 completion

## Minimal respectable subset status
Runbook Section 6 defines the minimal respectable subset as:
- lm-eval
- latency
- throughput
- NIAH

Current status on H100 as a proper campaign:
- completed: **0 / 4**
- remaining: **4 / 4**

## What is blocking full 11/11 right now
### No core fused-vLLM blocker
The fused-vLLM lane itself is no longer the gating issue.

### Real operational blockers for 11/11
The H100 host still lacks the long-context benchmark harness repos / data that
full 11/11 requires. As checked on 2026-03-27, these paths are missing on the
H100 host:
- `/root/xKV`
- `/root/LongBench`
- `/root/InfiniteBench`
- `/root/RULER`
- `/root/NIAH`

This means:
- phases 1, 2, 3, 5, 6, 7, 8 can be started once the exact benchmark config is
  chosen and output layout is prepared
- phases 4, 9, 10, 11 still need harness/data materialization on the H100 host
  (or an equivalent mounted path) before they can run

## What is not blocked
Checked on the H100 host:
- `/root/vllm-fused-git` exists
- Python `vllm` import works
- `lm_eval` is installed

This means the GPU can be used immediately for the early paired campaign steps.

## Practical next campaign order
1. Startup time
2. lm-eval
3. Latency sanity
4. Throughput
5. GuideLLM
6. Serving rate sweep
7. Materialize long-context harnesses and data on H100
8. NIAH
9. RULER
10. LongBench
11. InfiniteBench

## Bottom line
The H100 fused-vLLM lane is ready for real benchmark execution.
The main thing still missing for full 11/11 is benchmark harness/data setup on
that host, not kernel bring-up.
