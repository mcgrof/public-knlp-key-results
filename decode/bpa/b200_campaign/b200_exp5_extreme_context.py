#!/usr/bin/env python3
"""Experiment 5: Extreme Context on B200.

Push context to 64K-512K at B=1, record OOM thresholds, latency, memory.
"""

import csv
import json
import time
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_extreme_context(model, tokenizer, device, context_len, n_steps=8, n_warmup=2):
    """Measure decode at extreme context length with B=1."""
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(1, vocab_size, (1, context_len), device=device)

    torch.cuda.reset_peak_memory_stats()

    # Prefill
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    prefill_mem_gb = torch.cuda.max_memory_allocated() / 1024**3

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Timed decode
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_steps):
        with torch.no_grad():
            out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    decode_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    latency_ms = elapsed / n_steps * 1000
    tps = n_steps / elapsed

    del past, out
    torch.cuda.empty_cache()

    return {
        "latency_ms": latency_ms,
        "tokens_per_sec": tps,
        "peak_memory_gb": decode_mem_gb,
        "prefill_memory_gb": prefill_mem_gb,
    }


def main():
    device = "cuda:0"
    model_name = "Qwen/Qwen2.5-7B"

    print("Loading %s..." % model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device,
        trust_remote_code=True
    )
    model.eval()

    # Qwen2.5-7B supports up to 128K context
    contexts = [65536, 131072, 196608, 262144, 393216, 524288]
    results = []

    print("\nExperiment 5: Extreme Context (B=1)")
    print("%-10s %10s %10s %10s %10s" % (
        "Context", "lat(ms)", "tok/s", "peak(GB)", "prefill(GB)"))
    print("-" * 54)

    for ctx in contexts:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            r = measure_extreme_context(model, tokenizer, device, ctx)
            row = {
                "context_len": ctx,
                "latency_ms": round(r["latency_ms"], 2),
                "tokens_per_sec": round(r["tokens_per_sec"], 2),
                "peak_memory_gb": round(r["peak_memory_gb"], 2),
                "prefill_memory_gb": round(r["prefill_memory_gb"], 2),
            }
            results.append(row)
            print("%-10d %10.2f %10.2f %10.2f %10.2f" % (
                ctx, r["latency_ms"], r["tokens_per_sec"],
                r["peak_memory_gb"], r["prefill_memory_gb"]))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("%-10d %10s %10s %10s %10s" % (ctx, "OOM", "-", "-", "-"))
                results.append({
                    "context_len": ctx,
                    "latency_ms": None,
                    "tokens_per_sec": None,
                    "peak_memory_gb": None,
                    "prefill_memory_gb": None,
                    "status": "OOM",
                })
                torch.cuda.empty_cache()
            else:
                raise

    # Save
    csv_path = "/mnt/tmpfs/results/extreme_context_limits.csv"
    fields = ["context_len", "latency_ms", "tokens_per_sec",
              "peak_memory_gb", "prefill_memory_gb"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            if r.get("latency_ms") is not None:
                w.writerow(r)
    print("\nSaved: %s" % csv_path)

    json_path = "/mnt/tmpfs/results/extreme_context_limits.json"
    with open(json_path, "w") as f:
        json.dump({"model": model_name, "batch_size": 1, "results": results}, f, indent=2)
    print("Saved: %s" % json_path)

    # Summarize OOM threshold
    oom_ctx = None
    max_ok = 0
    for r in results:
        if r.get("status") == "OOM":
            if oom_ctx is None:
                oom_ctx = r["context_len"]
        elif r.get("latency_ms") is not None:
            max_ok = r["context_len"]

    print("\nMax successful context: %dK" % (max_ok // 1024))
    if oom_ctx:
        print("First OOM: %dK" % (oom_ctx // 1024))

    print("\nExperiment 5 complete.")


if __name__ == "__main__":
    main()
