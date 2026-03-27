#!/usr/bin/env python3
"""BPA v42 Part 3: Per-layer KV quantization sensitivity (v2).

Load model ONCE, then for each layer sweep, modify k_proj/v_proj
weights in-place and restore after eval. Much faster than reloading.

Measures perplexity degradation from INT4 quantization of KV
projection weights per layer.
"""

import gc
import json
import os
import sys
import time

import torch
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODELS = {
    "qwen25_7b": "Qwen/Qwen2.5-7B",
    "mistral_7b": "mistralai/Mistral-7B-v0.1",
    "llama2_7b": "NousResearch/Llama-2-7b-hf",
}

RESULTS_DIR = "artifacts/v42"
MAX_TOKENS = 200_000
CONTEXT_LEN = 2048
STRIDE = 1024


def quantize_int4_asym(tensor, group_size=128):
    """Asymmetric INT4 quantization -> dequantization (simulate noise)."""
    shape = tensor.shape
    hd = shape[-1]
    if hd % group_size != 0:
        group_size = hd  # fallback to per-row
    ng = hd // group_size
    r = tensor.reshape(*shape[:-1], ng, group_size)
    rmin = r.amin(dim=-1, keepdim=True)
    rmax = r.amax(dim=-1, keepdim=True)
    scale = (rmax - rmin).clamp(min=1e-8) / 15.0
    q = ((r - rmin) / scale).round().clamp(0, 15)
    deq = q * scale + rmin
    return deq.reshape(shape).to(tensor.dtype)


def get_attention_layers(model):
    """Extract attention modules from model."""
    layers = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                layers.append((i, attn))
    return layers


def compute_ppl(model, input_ids, device):
    """Sliding-window perplexity computation."""
    nlls = []
    seq_len = input_ids.size(0)
    prev_end = 0

    for begin in range(0, seq_len - 1, STRIDE):
        end = min(begin + CONTEXT_LEN, seq_len)
        target_begin = max(begin, prev_end)
        if target_begin >= end - 1:
            continue

        ids = input_ids[begin:end].unsqueeze(0).to(device)
        target_len = end - target_begin

        with torch.no_grad():
            outputs = model(ids)
            logits = outputs.logits

        # Shift: predict token i+1 from position i
        shift_logits = logits[0, -(target_len):-1, :].contiguous()
        shift_labels = ids[0, -(target_len - 1) :].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        if not torch.isnan(loss):
            nlls.append(loss.item() * (target_len - 1))

        prev_end = end
        if end >= seq_len:
            break

    total_tokens = sum(1 for _ in range(0, seq_len - 1, STRIDE))
    if not nlls:
        return float("inf")

    # Proper average
    total_nll = sum(nlls)
    # Count actual predicted tokens
    n_tokens = 0
    prev_end = 0
    for begin in range(0, seq_len - 1, STRIDE):
        end = min(begin + CONTEXT_LEN, seq_len)
        target_begin = max(begin, prev_end)
        if target_begin >= end - 1:
            continue
        n_tokens += end - target_begin - 1
        prev_end = end
        if end >= seq_len:
            break

    if n_tokens == 0:
        return float("inf")

    return float(np.exp(total_nll / n_tokens))


def run_model(model_name, model_id, device="cuda"):
    """Run full layer sensitivity for one model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"  Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings.input_ids[0]
    if len(input_ids) > MAX_TOKENS:
        input_ids = input_ids[:MAX_TOKENS]
    print(f"  Eval tokens: {len(input_ids)}")

    attn_layers = get_attention_layers(model)
    n_layers = len(attn_layers)
    print(f"  Layers: {n_layers}")

    results = {
        "model": model_name,
        "model_id": model_id,
        "n_layers": n_layers,
        "n_tokens": len(input_ids),
    }

    # Step 1: FP16 baseline
    print("  FP16 baseline...", flush=True)
    t0 = time.time()
    ppl_fp16 = compute_ppl(model, input_ids, device)
    print(f"  FP16 PPL: {ppl_fp16:.4f} ({time.time()-t0:.0f}s)")
    results["ppl_fp16"] = ppl_fp16

    # Step 2: All INT4 - quantize all k_proj, v_proj
    print("  All-INT4...", flush=True)
    saved_all = {}
    for layer_idx, attn in attn_layers:
        for proj_name in ["k_proj", "v_proj"]:
            proj = getattr(attn, proj_name, None)
            if proj is not None and hasattr(proj, "weight"):
                key = f"{layer_idx}_{proj_name}"
                saved_all[key] = proj.weight.data.clone()
                proj.weight.data = quantize_int4_asym(proj.weight.data)

    t0 = time.time()
    ppl_int4_all = compute_ppl(model, input_ids, device)
    print(f"  INT4-all PPL: {ppl_int4_all:.4f} ({time.time()-t0:.0f}s)")
    results["ppl_int4_all"] = ppl_int4_all

    # Restore all weights
    for layer_idx, attn in attn_layers:
        for proj_name in ["k_proj", "v_proj"]:
            key = f"{layer_idx}_{proj_name}"
            if key in saved_all:
                proj = getattr(attn, proj_name)
                proj.weight.data = saved_all[key]
    del saved_all

    # Step 3: Per-layer sweep
    # For each layer i: quantize ALL layers to INT4, then restore
    # layer i to FP16. Measure PPL.
    layer_scores = []
    for target_layer in range(n_layers):
        print(f"  Layer {target_layer}/{n_layers}...", end="", flush=True)

        # Quantize all layers
        saved = {}
        for layer_idx, attn in attn_layers:
            for proj_name in ["k_proj", "v_proj"]:
                proj = getattr(attn, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    key = f"{layer_idx}_{proj_name}"
                    saved[key] = proj.weight.data.clone()
                    if layer_idx != target_layer:
                        proj.weight.data = quantize_int4_asym(proj.weight.data)
                    # else: keep FP16

        t0 = time.time()
        ppl_i = compute_ppl(model, input_ids, device)
        elapsed = time.time() - t0

        # Restore
        for layer_idx, attn in attn_layers:
            for proj_name in ["k_proj", "v_proj"]:
                key = f"{layer_idx}_{proj_name}"
                if key in saved:
                    proj = getattr(attn, proj_name)
                    proj.weight.data = saved[key]
        del saved

        importance = ppl_int4_all - ppl_i
        layer_scores.append(
            {
                "layer": target_layer,
                "ppl": float(ppl_i),
                "importance": float(importance),
            }
        )
        print(f" PPL={ppl_i:.4f}, imp={importance:.4f} ({elapsed:.0f}s)")

    results["layer_scores"] = layer_scores

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = {}
    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_id})")
        print(f"{'='*60}")
        t0 = time.time()
        results = run_model(model_name, model_id, device)
        elapsed = time.time() - t0
        results["elapsed_s"] = elapsed
        all_results[model_name] = results
        print(f"  Total: {elapsed:.0f}s")

        # Save incrementally
        out_path = os.path.join(RESULTS_DIR, "layer_sensitivity.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved {out_path}")

    print("\nAll models complete.")


if __name__ == "__main__":
    main()
