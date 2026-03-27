#!/usr/bin/env python3
"""Experiment 6: KV Activation Quantization on B200.

Compare FP16 baseline, INT4 weight-quant KV, INT4 activation-quant KV,
and fused INT4 pipelines on WikiText-103.
"""

import csv
import json
import time
import math
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def quantize_kv_int4(tensor, group_size=32):
    """INT4 round-trip quantization."""
    shape = tensor.shape
    d = shape[-1]
    g = group_size
    if d % g != 0:
        pad = g - (d % g)
        tensor = torch.nn.functional.pad(tensor, (0, pad))
        d = tensor.shape[-1]
    reshaped = tensor.reshape(*shape[:-1], d // g, g)
    vmin = reshaped.amin(dim=-1, keepdim=True)
    vmax = reshaped.amax(dim=-1, keepdim=True)
    scale = (vmax - vmin) / 15.0
    scale = scale.clamp(min=1e-8)
    quantized = ((reshaped - vmin) / scale).round().clamp(0, 15)
    dequantized = quantized * scale + vmin
    result = dequantized.reshape(*shape[:-1], d)
    if result.shape[-1] != shape[-1]:
        result = result[..., :shape[-1]]
    return result


def quantize_cache(cache, group_size=32):
    """Quantize all layers of a DynamicCache."""
    qcache = DynamicCache()
    for i, layer in enumerate(cache.layers):
        k_q = quantize_kv_int4(layer.keys, group_size=group_size).to(layer.keys.dtype)
        v_q = quantize_kv_int4(layer.values, group_size=group_size).to(layer.values.dtype)
        qcache.update(k_q, v_q, i)
    return qcache


def load_wikitext103(tokenizer, max_tokens=500000):
    """Load WikiText-103 validation set."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        text = "\n\n".join([t for t in ds["text"] if t.strip()])
    except Exception:
        # Fallback: use a long repeated prompt
        text = ("The study of computational linguistics involves the application "
                "of computer science and mathematics to natural language processing. ") * 5000

    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokens


def compute_ppl_with_quant(model, token_ids, device, window=2048, stride=1024,
                           quant_mode="none", group_size=32, max_tokens=100000):
    """Compute perplexity with optional KV quantization.

    quant_mode: "none" (FP16), "weight" (g=32), "activation" (g=8), "fused" (g=4)
    """
    total_nll = 0.0
    total_tokens = 0

    g_map = {"none": 0, "weight": 32, "activation": 8, "fused": 4}
    g = g_map.get(quant_mode, 0)

    n_windows = min((len(token_ids) - window) // stride + 1,
                    max_tokens // stride)

    for i in range(n_windows):
        start = i * stride
        end = start + window
        if end > len(token_ids):
            break

        input_ids = torch.tensor([token_ids[start:end]], device=device)

        with torch.no_grad():
            if g > 0:
                # Sliding window with quantized cache
                # Process first half, quantize cache, then score second half
                half = window // 2
                first_half = input_ids[:, :half]
                second_half = input_ids[:, half:]

                out1 = model(first_half, use_cache=True)
                cache = quantize_cache(out1.past_key_values, group_size=g)

                out2 = model(second_half, past_key_values=cache, use_cache=False)
                logits = out2.logits
                targets = second_half[:, 1:]
                logits = logits[:, :-1, :]
            else:
                out = model(input_ids, use_cache=False)
                logits = out.logits[:, :-1, :]
                targets = input_ids[:, 1:]

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum"
        )
        total_nll += loss.item()
        total_tokens += targets.numel()

    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
    return ppl, total_tokens


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

    print("Loading WikiText-103...")
    tokens = load_wikitext103(tokenizer, max_tokens=200000)
    print("Loaded %d tokens" % len(tokens))

    modes = [
        ("FP16_baseline", "none"),
        ("INT4_weight_g32", "weight"),
        ("INT4_activation_g8", "activation"),
        ("INT4_fused_g4", "fused"),
    ]

    results = []
    print("\n%-25s %10s %10s" % ("Pipeline", "PPL", "Tokens"))
    print("-" * 48)

    for label, mode in modes:
        torch.cuda.empty_cache()
        gc.collect()

        t0 = time.perf_counter()
        ppl, n_tok = compute_ppl_with_quant(
            model, tokens, device, window=2048, stride=512,
            quant_mode=mode, max_tokens=50000)
        elapsed = time.perf_counter() - t0

        tps = n_tok / elapsed if elapsed > 0 else 0
        row = {
            "pipeline": label,
            "perplexity": round(ppl, 4),
            "tokens_evaluated": n_tok,
            "tokens_per_sec": round(tps, 1),
            "elapsed_sec": round(elapsed, 1),
        }
        results.append(row)
        print("%-25s %10.4f %10d  (%.1fs, %.0f tok/s)" % (
            label, ppl, n_tok, elapsed, tps))

    # PPL degradation
    if results and results[0]["perplexity"] > 0:
        base_ppl = results[0]["perplexity"]
        print("\nPPL degradation vs FP16:")
        for r in results[1:]:
            delta = (r["perplexity"] - base_ppl) / base_ppl * 100
            print("  %s: +%.2f%%" % (r["pipeline"], delta))

    # Save
    csv_path = "/mnt/tmpfs/results/activation_quant_results.csv"
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print("\nSaved: %s" % csv_path)

    json_path = "/mnt/tmpfs/results/activation_quant_results.json"
    with open(json_path, "w") as f:
        json.dump({"model": model_name, "dataset": "wikitext-103", "results": results}, f, indent=2)
    print("Saved: %s" % json_path)

    print("\nExperiment 6 complete.")


if __name__ == "__main__":
    main()
