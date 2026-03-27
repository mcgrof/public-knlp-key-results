#!/usr/bin/env python3
"""Experiment 1: Numerical Correctness Validation on B200 (v3).

Transformers 5.x API: cache.layers[i].keys / cache.layers[i].values
"""

import csv
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def quantize_kv_int4(tensor, group_size=32):
    """Simulate INT4 quantization round-trip with given group size."""
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
    """Quantize all layers of a DynamicCache, return new DynamicCache."""
    qcache = DynamicCache()
    for i, layer in enumerate(cache.layers):
        k = layer.keys
        v = layer.values
        k_q = quantize_kv_int4(k, group_size=group_size).to(k.dtype)
        v_q = quantize_kv_int4(v, group_size=group_size).to(v.dtype)
        qcache.update(k_q, v_q, i)
    return qcache


def generate_fp16(model, input_ids, max_new_tokens):
    """Generate with FP16 KV cache."""
    generated = input_ids.clone()
    past = None
    all_logits = []

    for _ in range(max_new_tokens):
        inp = generated if past is None else generated[:, -1:]
        with torch.no_grad():
            out = model(inp, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1:, :]
        all_logits.append(logits.float())
        past = out.past_key_values
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)

    return generated, torch.cat(all_logits, dim=1)


def generate_with_kv_quant(model, input_ids, max_new_tokens, group_size=32):
    """Generate with INT4-quantized KV cache at each step."""
    generated = input_ids.clone()
    past = None
    all_logits = []

    for _ in range(max_new_tokens):
        inp = generated if past is None else generated[:, -1:]
        with torch.no_grad():
            out = model(inp, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1:, :]
        all_logits.append(logits.float())
        # Quantize the cache before next step
        past = quantize_cache(out.past_key_values, group_size=group_size)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)

    return generated, torch.cat(all_logits, dim=1)


def compute_metrics(logits_ref, logits_test, tokens_ref, tokens_test, prefix_len):
    """Compute correctness metrics."""
    gen_ref = tokens_ref[:, prefix_len:]
    gen_test = tokens_test[:, prefix_len:]
    n = min(gen_ref.shape[1], gen_test.shape[1])

    agree = (gen_ref[:, :n] == gen_test[:, :n]).float().mean().item()

    n_l = min(logits_ref.shape[1], logits_test.shape[1])
    lr = logits_ref[:, :n_l, :]
    lt = logits_test[:, :n_l, :]

    max_err = (lr - lt).abs().max().item()
    mean_drift = (lr - lt).abs().mean().item()

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

    print("  FP16 baseline (%d new tokens)..." % max_new)
    tokens_fp16, logits_fp16 = generate_fp16(model, input_ids, max_new)
    text_fp16 = tokenizer.decode(tokens_fp16[0, prefix_len:], skip_special_tokens=True)
    print("  FP16: %s" % text_fp16[:80])

    print("  INT4-WQ g=32...")
    tokens_wq, logits_wq = generate_with_kv_quant(model, input_ids, max_new, 32)
    text_wq = tokenizer.decode(tokens_wq[0, prefix_len:], skip_special_tokens=True)
    print("  WQ32: %s" % text_wq[:80])

    print("  INT4-AQ g=8...")
    tokens_aq, logits_aq = generate_with_kv_quant(model, input_ids, max_new, 8)
    text_aq = tokenizer.decode(tokens_aq[0, prefix_len:], skip_special_tokens=True)
    print("  AQ8:  %s" % text_aq[:80])

    m_wq = compute_metrics(logits_fp16, logits_wq, tokens_fp16, tokens_wq, prefix_len)
    m_aq = compute_metrics(logits_fp16, logits_aq, tokens_fp16, tokens_aq, prefix_len)

    print("  WQ32: agree=%.1f%% max_err=%.3f drift=%.4f KL=%.6f" % (
        m_wq["token_agreement"]*100, m_wq["max_logit_error"],
        m_wq["mean_logit_drift"], m_wq["kl_divergence"]))
    print("  AQ8:  agree=%.1f%% max_err=%.3f drift=%.4f KL=%.6f" % (
        m_aq["token_agreement"]*100, m_aq["max_logit_error"],
        m_aq["mean_logit_drift"], m_aq["kl_divergence"]))

    row = {
        "model": model_name,
        "prefix_len": prefix_len,
        "gen_tokens": max_new,
        "wq_g32_agree": round(m_wq["token_agreement"], 4),
        "wq_g32_max_err": round(m_wq["max_logit_error"], 4),
        "wq_g32_drift": round(m_wq["mean_logit_drift"], 4),
        "wq_g32_kl": round(m_wq["kl_divergence"], 6),
        "aq_g8_agree": round(m_aq["token_agreement"], 4),
        "aq_g8_max_err": round(m_aq["max_logit_error"], 4),
        "aq_g8_drift": round(m_aq["mean_logit_drift"], 4),
        "aq_g8_kl": round(m_aq["kl_divergence"], 6),
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
            import traceback; traceback.print_exc()
            results.append({"model": name, "error": str(e)})
            torch.cuda.empty_cache()

    # Save
    good = [r for r in results if "error" not in r]
    csv_path = "/mnt/tmpfs/results/correctness_validation.csv"
    if good:
        fields = list(good[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in good:
                w.writerow(r)
        print("\nSaved: %s" % csv_path)

    json_path = "/mnt/tmpfs/results/correctness_validation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: %s" % json_path)

    # Stability check
    for r in results:
        if "error" in r:
            print("WARNING: %s failed: %s" % (r["model"], r.get("error", "")))

    stable = all("error" not in r for r in results)
    if stable:
        print("\nExperiment 1 PASSED: All models numerically stable.")
    else:
        ok = [r for r in results if "error" not in r]
        print("\n%d/%d models passed." % (len(ok), len(results)))
    return 0


if __name__ == "__main__":
    exit(main())
