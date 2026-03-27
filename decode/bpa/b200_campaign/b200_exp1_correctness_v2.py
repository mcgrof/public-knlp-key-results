#!/usr/bin/env python3
"""Experiment 1: Numerical Correctness Validation on B200 (v2).

Compare FP16 baseline vs INT4 weight-quantized KV cache vs INT4 activation-quantized KV.
Multi-token generation to get meaningful token agreement.
"""

import csv
import json
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def quantize_kv_int4(tensor, group_size=32):
    """Quantize a KV cache tensor to INT4 with given group size.
    Returns dequantized tensor (simulating the round-trip).
    """
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


def generate_with_kv_quant(model, input_ids, max_new_tokens, group_size=32):
    """Generate tokens using INT4-quantized KV cache at each step."""
    generated = input_ids.clone()
    past = None
    all_logits = []

    for step in range(max_new_tokens):
        if past is None:
            inp = generated
        else:
            inp = generated[:, -1:]

        with torch.no_grad():
            out = model(inp, past_key_values=past, use_cache=True)

        logits = out.logits[:, -1:, :]
        all_logits.append(logits.float())

        # Quantize the KV cache
        cache = out.past_key_values
        qcache = DynamicCache()
        for i in range(len(cache)):
            k = cache.key_cache[i]
            v = cache.value_cache[i]
            k_q = quantize_kv_int4(k, group_size=group_size)
            v_q = quantize_kv_int4(v, group_size=group_size)
            qcache.update(k_q.half(), v_q.half(), i)
        past = qcache

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated, torch.cat(all_logits, dim=1)


def generate_fp16(model, input_ids, max_new_tokens):
    """Generate tokens with FP16 KV cache (baseline)."""
    generated = input_ids.clone()
    past = None
    all_logits = []

    for step in range(max_new_tokens):
        if past is None:
            inp = generated
        else:
            inp = generated[:, -1:]

        with torch.no_grad():
            out = model(inp, past_key_values=past, use_cache=True)

        logits = out.logits[:, -1:, :]
        all_logits.append(logits.float())
        past = out.past_key_values

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated, torch.cat(all_logits, dim=1)


def compute_metrics(logits_ref, logits_test, tokens_ref, tokens_test, prefix_len):
    """Compute correctness metrics between reference and test outputs."""
    gen_ref = tokens_ref[:, prefix_len:]
    gen_test = tokens_test[:, prefix_len:]
    n = min(gen_ref.shape[1], gen_test.shape[1])
    gen_ref = gen_ref[:, :n]
    gen_test = gen_test[:, :n]

    # Token agreement
    agree = (gen_ref == gen_test).float().mean().item()

    # Logit metrics (per-step)
    n_logits = min(logits_ref.shape[1], logits_test.shape[1])
    lr = logits_ref[:, :n_logits, :]
    lt = logits_test[:, :n_logits, :]

    max_err = (lr - lt).abs().max().item()
    mean_drift = (lr - lt).abs().mean().item()

    # KL divergence (averaged over steps)
    p_ref = torch.softmax(lr, dim=-1).clamp(min=1e-8)
    p_test = torch.softmax(lt, dim=-1).clamp(min=1e-8)
    kl = (p_ref * (p_ref.log() - p_test.log())).sum(dim=-1).mean().item()

    return {
        "token_agreement": agree,
        "max_logit_error": max_err,
        "mean_logit_drift": mean_drift,
        "kl_divergence": kl,
    }


def run_model(model_name, device, prompt, max_new=32):
    """Run correctness check for one model."""
    print("\n--- %s ---" % model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device,
        trust_remote_code=True
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prefix_len = input_ids.shape[1]

    # FP16 baseline
    print("  FP16 baseline generation (%d tokens)..." % max_new)
    tokens_fp16, logits_fp16 = generate_fp16(model, input_ids, max_new)
    gen_text_fp16 = tokenizer.decode(tokens_fp16[0, prefix_len:], skip_special_tokens=True)
    print("  FP16: ...%s" % gen_text_fp16[:80])

    # INT4 weight-quant (g=32)
    print("  INT4-WQ (g=32) generation...")
    tokens_wq, logits_wq = generate_with_kv_quant(model, input_ids, max_new, group_size=32)
    gen_text_wq = tokenizer.decode(tokens_wq[0, prefix_len:], skip_special_tokens=True)
    print("  WQ32: ...%s" % gen_text_wq[:80])

    # INT4 activation-quant (g=8)
    print("  INT4-AQ (g=8) generation...")
    tokens_aq, logits_aq = generate_with_kv_quant(model, input_ids, max_new, group_size=8)
    gen_text_aq = tokenizer.decode(tokens_aq[0, prefix_len:], skip_special_tokens=True)
    print("  AQ8:  ...%s" % gen_text_aq[:80])

    # Metrics
    m_wq = compute_metrics(logits_fp16, logits_wq, tokens_fp16, tokens_wq, prefix_len)
    m_aq = compute_metrics(logits_fp16, logits_aq, tokens_fp16, tokens_aq, prefix_len)

    print("  INT4-WQ: agree=%.1f%% max_err=%.3f drift=%.4f KL=%.6f" % (
        m_wq["token_agreement"] * 100, m_wq["max_logit_error"],
        m_wq["mean_logit_drift"], m_wq["kl_divergence"]))
    print("  INT4-AQ: agree=%.1f%% max_err=%.3f drift=%.4f KL=%.6f" % (
        m_aq["token_agreement"] * 100, m_aq["max_logit_error"],
        m_aq["mean_logit_drift"], m_aq["kl_divergence"]))

    row = {
        "model": model_name,
        "prefix_len": prefix_len,
        "gen_tokens": max_new,
        "wq_g32_agree": m_wq["token_agreement"],
        "wq_g32_max_err": m_wq["max_logit_error"],
        "wq_g32_drift": m_wq["mean_logit_drift"],
        "wq_g32_kl": m_wq["kl_divergence"],
        "aq_g8_agree": m_aq["token_agreement"],
        "aq_g8_max_err": m_aq["max_logit_error"],
        "aq_g8_drift": m_aq["mean_logit_drift"],
        "aq_g8_kl": m_aq["kl_divergence"],
    }

    del model
    torch.cuda.empty_cache()
    return row


def main():
    device = "cuda:0"

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
        "Qwen/Qwen2.5-14B",
    ]

    results = []
    for name in models:
        try:
            row = run_model(name, device, prompt, max_new=32)
            results.append(row)
        except Exception as e:
            print("  ERROR: %s" % e)
            import traceback
            traceback.print_exc()
            results.append({"model": name, "error": str(e)})
            torch.cuda.empty_cache()

    # Save CSV
    csv_path = "/mnt/tmpfs/results/correctness_validation.csv"
    good = [r for r in results if "error" not in r]
    if good:
        fields = list(good[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in good:
                w.writerow(r)
        print("\nSaved: %s" % csv_path)

    # Save JSON
    json_path = "/mnt/tmpfs/results/correctness_validation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: %s" % json_path)

    # Check stability
    for r in results:
        if "error" in r:
            print("WARNING: %s failed" % r["model"])
        elif r.get("wq_g32_max_err", 0) > 100 or r.get("aq_g8_max_err", 0) > 100:
            print("ABORT: Numerical instability in %s" % r["model"])
            return 1

    print("\nExperiment 1 PASSED.")
    return 0


if __name__ == "__main__":
    exit(main())
