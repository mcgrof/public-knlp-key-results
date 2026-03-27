#!/usr/bin/env python3
"""Experiment 4: Context Scaling on B200.

Sweep context lengths at multiple batch sizes.
"""

import csv
import json
import time
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_decode(model, tokenizer, device, batch_size, context_len, n_steps=16, n_warmup=3):
    """Measure decode latency and throughput."""
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(1, vocab_size, (batch_size, context_len), device=device)

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    for _ in range(n_warmup):
        with torch.no_grad():
            out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_steps):
        with torch.no_grad():
            out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tps = batch_size * n_steps / elapsed
    latency_ms = elapsed / n_steps * 1000

    # KV bandwidth estimate
    n_layers = model.config.num_hidden_layers
    n_kv_heads = getattr(model.config, "num_key_value_heads",
                         model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    kv_bytes = 2 * n_layers * n_kv_heads * head_dim * context_len * batch_size * 2
    kv_bw = kv_bytes * n_steps / elapsed / 1e9

    del past, out
    torch.cuda.empty_cache()
    return tps, latency_ms, kv_bw


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

    contexts = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    batches = [1, 8, 16, 32]
    results = []

    print("\nExperiment 4: Context Scaling")
    print("%-8s %-8s %12s %12s %12s" % ("Batch", "Context", "tok/s", "lat(ms)", "KV BW(GB/s)"))
    print("-" * 56)

    for bs in batches:
        for ctx in contexts:
            try:
                torch.cuda.empty_cache()
                gc.collect()
                tps, lat, bw = measure_decode(model, tokenizer, device, bs, ctx)
                row = {
                    "batch_size": bs,
                    "context_len": ctx,
                    "tokens_per_sec": round(tps, 1),
                    "latency_ms": round(lat, 3),
                    "kv_bandwidth_gb_s": round(bw, 1),
                }
                results.append(row)
                print("%-8d %-8d %12.1f %12.3f %12.1f" % (bs, ctx, tps, lat, bw))
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("%-8d %-8d %12s %12s %12s" % (bs, ctx, "OOM", "-", "-"))
                    torch.cuda.empty_cache()
                    break
                else:
                    raise

    # Save CSV
    csv_path = "/mnt/tmpfs/results/context_scaling_b200.csv"
    if results:
        fields = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow(r)
    print("\nSaved: %s" % csv_path)

    # JSON
    json_path = "/mnt/tmpfs/results/context_scaling_b200.json"
    with open(json_path, "w") as f:
        json.dump({"model": model_name, "results": results}, f, indent=2)
    print("Saved: %s" % json_path)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for bs in batches:
            data = [r for r in results if r["batch_size"] == bs]
            if not data:
                continue
            ctxs = [r["context_len"] for r in data]
            lats = [r["latency_ms"] for r in data]
            bws = [r["kv_bandwidth_gb_s"] for r in data]

            ax1.plot(ctxs, lats, "o-", linewidth=2, markersize=5, label="B=%d" % bs)
            ax2.plot(ctxs, bws, "s-", linewidth=2, markersize=5, label="B=%d" % bs)

        ax1.set_xlabel("Context Length")
        ax1.set_ylabel("Decode Latency (ms)")
        ax1.set_title("B200 Decode Latency vs Context")
        ax1.set_xscale("log", base=2)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.axhline(y=6550, color="gray", linestyle="--", alpha=0.5,
                     label="Peak BW (6550)")
        ax2.set_xlabel("Context Length")
        ax2.set_ylabel("KV Bandwidth (GB/s)")
        ax2.set_title("B200 KV Bandwidth vs Context")
        ax2.set_xscale("log", base=2)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = "/mnt/tmpfs/figures/latency_vs_context_b200.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print("Saved: %s" % fig_path)
        plt.close()
    except Exception as e:
        print("Plot failed: %s" % e)

    print("\nExperiment 4 complete.")


if __name__ == "__main__":
    main()
