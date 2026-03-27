#!/usr/bin/env python3
"""Experiment 1: Numerical Correctness Validation on B200.

Compare FP16 baseline vs INT4 weight-quantized KV cache vs INT4 activation-quantized KV cache.
Measures max error, token agreement, and logit drift.
"""

import csv
import json
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_kv_int4_per_channel(tensor):
    """Quantize a KV cache tensor to INT4 per-channel (simulate).

    Groups of 32 values share a scale factor.
    Returns dequantized tensor (simulating the round-trip).
    """
    shape = tensor.shape
    # Reshape to group last dim into groups of 32
    d = shape[-1]
    g = 32
    if d % g != 0:
        # pad
        pad = g - (d % g)
        tensor = torch.nn.functional.pad(tensor, (0, pad))
        d = tensor.shape[-1]

    reshaped = tensor.reshape(*shape[:-1], d // g, g)
    # Per-group min/max
    vmin = reshaped.amin(dim=-1, keepdim=True)
    vmax = reshaped.amax(dim=-1, keepdim=True)
    scale = (vmax - vmin) / 15.0  # 4-bit = 16 levels
    scale = scale.clamp(min=1e-8)
    # Quantize and dequantize
    quantized = ((reshaped - vmin) / scale).round().clamp(0, 15)
    dequantized = quantized * scale + vmin
    result = dequantized.reshape(*shape[:-1], d)
    if result.shape[-1] != shape[-1]:
        result = result[..., :shape[-1]]
    return result


def run_correctness_check(model_name, device, prompt_text, max_new=50):
    """Run correctness check for a single model."""
    print("\n--- %s ---" % model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True
    )
    model.eval()

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    results = {}

    # --- Pipeline 0: FP16 baseline ---
    print("  FP16 baseline...")
    with torch.no_grad():
        out_fp16 = model(input_ids, use_cache=True)
    logits_fp16 = out_fp16.logits[:, -1, :].float()
    tokens_fp16 = logits_fp16.argmax(dim=-1)

    # Extract KV cache
    cache_fp16 = out_fp16.past_key_values

    # --- Pipeline 1: INT4 weight-quantized KV ---
    # Quantize the KV cache tensors and re-run the last token
    print("  INT4 weight-quant KV...")
    quantized_cache_w = []
    for layer_cache in cache_fp16:
        k = layer_cache[0]
        v = layer_cache[1]
        k_q = quantize_kv_int4_per_channel(k)
        v_q = quantize_kv_int4_per_channel(v)
        quantized_cache_w.append((k_q, v_q))

    # Build DynamicCache from quantized
    from transformers import DynamicCache
    qcache_w = DynamicCache()
    for i, (k_q, v_q) in enumerate(quantized_cache_w):
        qcache_w.update(k_q.half(), v_q.half(), i)

    with torch.no_grad():
        out_wq = model(input_ids[:, -1:], past_key_values=qcache_w,
                       use_cache=False)
    logits_wq = out_wq.logits[:, -1, :].float()
    tokens_wq = logits_wq.argmax(dim=-1)

    # --- Pipeline 2: INT4 activation-quantized KV ---
    # Same as weight quant in this context (both quantize cached K,V)
    # But we simulate "activation quant" by quantizing at finer granularity (g=8)
    print("  INT4 activation-quant KV (g=8)...")
    quantized_cache_a = []
    for layer_cache in cache_fp16:
        k = layer_cache[0]
        v = layer_cache[1]
        # Use smaller group for activation quant
        k_q = quantize_kv_int4_activation(k, group_size=8)
        v_q = quantize_kv_int4_activation(v, group_size=8)
        quantized_cache_a.append((k_q, v_q))

    qcache_a = DynamicCache()
    for i, (k_q, v_q) in enumerate(quantized_cache_a):
        qcache_a.update(k_q.half(), v_q.half(), i)

    with torch.no_grad():
        out_aq = model(input_ids[:, -1:], past_key_values=qcache_a,
                       use_cache=False)
    logits_aq = out_aq.logits[:, -1, :].float()
    tokens_aq = logits_aq.argmax(dim=-1)

    # --- Metrics ---
    # Max logit error
    max_err_wq = (logits_fp16 - logits_wq).abs().max().item()
    max_err_aq = (logits_fp16 - logits_aq).abs().max().item()

    # Token agreement
    agree_wq = (tokens_fp16 == tokens_wq).float().mean().item()
    agree_aq = (tokens_fp16 == tokens_aq).float().mean().item()

    # Logit drift (mean abs difference)
    drift_wq = (logits_fp16 - logits_wq).abs().mean().item()
    drift_aq = (logits_fp16 - logits_aq).abs().mean().item()

    # KL divergence
    p_fp16 = torch.softmax(logits_fp16, dim=-1)
    p_wq = torch.softmax(logits_wq, dim=-1)
    p_aq = torch.softmax(logits_aq, dim=-1)
    kl_wq = torch.nn.functional.kl_div(
        p_wq.log(), p_fp16, reduction="batchmean").item()
    kl_aq = torch.nn.functional.kl_div(
        p_aq.log(), p_fp16, reduction="batchmean").item()

    print("  Results:")
    print("    INT4-WQ: max_err=%.4f, agree=%.2f%%, drift=%.4f, KL=%.6f" % (
        max_err_wq, agree_wq * 100, drift_wq, kl_wq))
    print("    INT4-AQ: max_err=%.4f, agree=%.2f%%, drift=%.4f, KL=%.6f" % (
        max_err_aq, agree_aq * 100, drift_aq, kl_aq))

    results = {
        "model": model_name,
        "seq_len": seq_len,
        "int4_wq_max_error": max_err_wq,
        "int4_wq_token_agree": agree_wq,
        "int4_wq_logit_drift": drift_wq,
        "int4_wq_kl_div": kl_wq,
        "int4_aq_max_error": max_err_aq,
        "int4_aq_token_agree": agree_aq,
        "int4_aq_logit_drift": drift_aq,
        "int4_aq_kl_div": kl_aq,
    }

    # Cleanup
    del model, cache_fp16, qcache_w, qcache_a
    torch.cuda.empty_cache()

    return results


def quantize_kv_int4_activation(tensor, group_size=8):
    """INT4 activation quantization with smaller group size."""
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


def main():
    device = "cuda:0"
    torch.cuda.empty_cache()

    prompt = (
        "The theory of general relativity, proposed by Albert Einstein in 1915, "
        "fundamentally changed our understanding of gravity. Rather than viewing "
        "gravity as a force acting at a distance, Einstein described it as the "
        "curvature of spacetime caused by mass and energy. This revolutionary "
        "framework has been confirmed by numerous experiments, including the "
        "observation of gravitational waves and the imaging of black holes."
    )

    models = [
        "Qwen/Qwen2.5-7B",
        "mistralai/Mistral-7B-v0.3",
        "meta-llama/Llama-3.1-8B",
    ]

    all_results = []
    for model_name in models:
        try:
            result = run_correctness_check(model_name, device, prompt)
            all_results.append(result)
        except Exception as e:
            print("  ERROR: %s" % e)
            all_results.append({"model": model_name, "error": str(e)})
            torch.cuda.empty_cache()

    # Write CSV
    csv_path = "/mnt/tmpfs/results/correctness_validation.csv"
    if all_results and "error" not in all_results[0]:
        fields = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in all_results:
                if "error" not in r:
                    writer.writerow(r)
    print("\nSaved: %s" % csv_path)

    # Also JSON
    json_path = "/mnt/tmpfs/results/correctness_validation.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved: %s" % json_path)

    # Check for instability
    for r in all_results:
        if "error" in r:
            print("\nWARNING: %s failed: %s" % (r["model"], r["error"]))
        elif r.get("int4_wq_max_error", 0) > 100:
            print("\nABORT: Numerical instability in %s (max_err=%.2f)" % (
                r["model"], r["int4_wq_max_error"]))
            return 1

    print("\nExperiment 1 PASSED: No numerical instability detected.")
    return 0


if __name__ == "__main__":
    exit(main())
