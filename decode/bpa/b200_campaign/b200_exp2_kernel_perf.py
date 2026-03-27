#!/usr/bin/env python3
"""Experiment 2: Kernel Performance - Decode throughput on B200.

Measures tokens/sec, latency/token, and memory bandwidth at various batch sizes.
Uses Qwen2.5-7B as the standard model.
"""

import csv
import json
import time
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_decode_throughput(model, tokenizer, device, batch_size, context_len=2048,
                              n_decode_steps=32, n_warmup=5):
    """Measure decode throughput for given batch size and context length."""
    # Create dummy input of context_len tokens
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(1, vocab_size, (batch_size, context_len), device=device)

    # Prefill
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Warmup decode steps
    for _ in range(n_warmup):
        with torch.no_grad():
            out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize()

    # Timed decode steps
    start = time.perf_counter()
    for _ in range(n_decode_steps):
        with torch.no_grad():
            out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = batch_size * n_decode_steps
    tokens_per_sec = total_tokens / elapsed
    latency_ms = elapsed / n_decode_steps * 1000

    # Estimate KV memory bandwidth
    # Each decode step reads entire KV cache: 2 * n_layers * 2 * B * T * d_head * n_heads * 2bytes
    # For Qwen2.5-7B: 28 layers, 28 KV heads, 128 head_dim
    n_layers = model.config.num_hidden_layers
    n_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    avg_ctx = context_len + n_decode_steps // 2  # average context during decode
    kv_bytes_per_step = 2 * n_layers * n_kv_heads * head_dim * avg_ctx * batch_size * 2  # fp16
    kv_bw_gb_s = kv_bytes_per_step * n_decode_steps / elapsed / 1e9

    del past, out
    torch.cuda.empty_cache()

    return {
        "tokens_per_sec": tokens_per_sec,
        "latency_ms": latency_ms,
        "kv_bandwidth_gb_s": kv_bw_gb_s,
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

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    context_len = 2048
    results = []

    print("\nExperiment 2: Decode throughput sweep")
    print("Model: %s, Context: %d" % (model_name, context_len))
    print("%-8s %12s %12s %12s" % ("Batch", "tok/s", "lat(ms)", "KV BW(GB/s)"))
    print("-" * 48)

    for bs in batch_sizes:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            r = measure_decode_throughput(model, tokenizer, device, bs, context_len)
            row = {
                "batch_size": bs,
                "context_len": context_len,
                "tokens_per_sec": round(r["tokens_per_sec"], 1),
                "latency_ms": round(r["latency_ms"], 3),
                "kv_bandwidth_gb_s": round(r["kv_bandwidth_gb_s"], 1),
            }
            results.append(row)
            print("%-8d %12.1f %12.3f %12.1f" % (
                bs, r["tokens_per_sec"], r["latency_ms"], r["kv_bandwidth_gb_s"]))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("%-8d %12s %12s %12s" % (bs, "OOM", "-", "-"))
                torch.cuda.empty_cache()
                break
            else:
                raise

    # Save CSV
    csv_path = "/mnt/tmpfs/results/kernel_perf.csv"
    if results:
        fields = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow(r)
        print("\nSaved: %s" % csv_path)

    # Save JSON
    json_path = "/mnt/tmpfs/results/kernel_perf.json"
    with open(json_path, "w") as f:
        json.dump({"model": model_name, "context_len": context_len, "results": results}, f, indent=2)
    print("Saved: %s" % json_path)

    # Generate plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bs_list = [r["batch_size"] for r in results]
        tps_list = [r["tokens_per_sec"] for r in results]
        bw_list = [r["kv_bandwidth_gb_s"] for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(bs_list, tps_list, "o-", color="steelblue", linewidth=2)
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Tokens/sec")
        ax1.set_title("B200 Decode Throughput (Qwen2.5-7B, T=2048)")
        ax1.set_xscale("log", base=2)
        ax1.grid(True, alpha=0.3)

        ax2.plot(bs_list, bw_list, "s-", color="firebrick", linewidth=2)
        ax2.axhline(y=6550, color="gray", linestyle="--", alpha=0.5, label="Peak BW (6550 GB/s)")
        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("KV Bandwidth (GB/s)")
        ax2.set_title("B200 KV Memory Bandwidth Utilization")
        ax2.set_xscale("log", base=2)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = "/mnt/tmpfs/figures/kernel_perf_b200.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print("Saved: %s" % fig_path)
        plt.close()
    except Exception as e:
        print("Plot failed: %s" % e)

    print("\nExperiment 2 complete.")


if __name__ == "__main__":
    main()
