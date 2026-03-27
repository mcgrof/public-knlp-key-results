#!/usr/bin/env python3
"""BPA v42 Part 5: Memory tiering simulation.

Projects KV cache storage requirements at long contexts
and simulates HBM/secondary-tier usage under capacity limits.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_CONFIGS = "artifacts/v40/model_configs.json"
HW_LIMITS = "artifacts/v40/hardware_limits.json"
RESULTS_DIR = "artifacts/v42"
PLOTS_DIR = "plots"

# Context lengths to project
CONTEXTS = [32768, 65536, 131072, 262144]

# HBM capacity scenarios (in GB)
HBM_CAPACITIES = {
    "H100-80GB": 80,
    "A100-40GB": 40,
    "L40S-48GB": 48,
}

# Model weight overhead (approx, in GB)
MODEL_WEIGHT_GB = {
    "qwen25_05b": 1.0,
    "qwen25_1.8b": 3.6,
    "qwen25_7b": 14.0,
    "mistral_7b": 14.0,
    "llama31_8b": 16.0,
    "nemotron3_nano": 60.0,
    "deepseek_mla": 14.0,
    "gemma3_4b": 8.0,
    "gemma3_12b": 24.0,
    "gemma3n_e4b": 8.0,
    "deepseek_r1_7b": 14.0,
    "phi4_14b": 28.0,
    "qwen3_8b": 16.0,
    "mistral_small_24b": 48.0,
}

# Secondary tier bandwidth (GB/s)
SECONDARY_BW = {
    "NVLink": 900,
    "PCIe5-x16": 64,
    "CXL-2.0": 32,
}


def load_configs():
    with open(MODEL_CONFIGS) as f:
        return json.load(f)


def compute_kv_memory(model_cfgs):
    """Compute KV cache memory per model at each context length."""
    results = {}
    for model, cfg in model_cfgs.items():
        kv_bpt = cfg["kv_bytes_per_token"]
        n_layers = 1  # kv_bpt already includes all layers
        # kv_bpt is per-token, per-layer? Let's verify.
        # From model_configs.json, kv_bpt = n_kv_heads * head_dim * 2 * 2
        # (K + V, 2 bytes per FP16 element, per layer)
        # Wait, actually kv_bpt is PER TOKEN across ALL layers:
        # For llama31_8b: kv_bpt=4096, n_kv_heads=8, head_dim=128
        # Per layer: 8 * 128 * 2 (K+V) * 2 (bytes) = 4096
        # But that's per LAYER. For 32 layers: 32 * 4096 = 131072 bytes/token
        # The v40 benchmark uses per-LAYER kv_bpt for the kernel test.
        # For total model KV, we need to multiply by n_layers.

        # Get n_layers from model weight estimate
        # Actually, let's compute it: model config gives n_kv_heads, head_dim
        # kv_bpt = n_kv_heads * head_dim * 2 (K+V) * 2 (bytes) per layer
        n_kv = cfg["n_kv_heads"]
        hd = cfg["head_dim"]
        kv_per_layer = n_kv * hd * 2 * 2  # K+V, fp16
        # Verify against kv_bpt
        assert kv_per_layer == kv_bpt, f"{model}: {kv_per_layer} != {kv_bpt}"

        # Estimate n_layers from model size
        # Rough: params / (hidden_dim^2 * 12) for transformer
        # Better: use known values
        n_layers_map = {
            "qwen25_05b": 24,
            "qwen25_1.8b": 28,
            "qwen25_7b": 28,
            "mistral_7b": 32,
            "llama31_8b": 32,
            "nemotron3_nano": 32,
            "deepseek_mla": 28,
            "gemma3_4b": 34,
            "gemma3_12b": 48,
            "gemma3n_e4b": 35,
            "deepseek_r1_7b": 28,
            "phi4_14b": 40,
            "qwen3_8b": 36,
            "mistral_small_24b": 40,
        }

        n_layers = n_layers_map.get(model, 32)
        total_kv_per_tok = kv_per_layer * n_layers  # bytes per token, all layers

        model_results = {
            "kv_per_layer_bytes": kv_per_layer,
            "n_layers": n_layers,
            "total_kv_per_tok_bytes": total_kv_per_tok,
            "contexts": {},
        }

        for T in CONTEXTS:
            for B in [1, 32, 128]:
                kv_total = total_kv_per_tok * T * B
                kv_gb = kv_total / (1024**3)
                # INT4 version
                kv_int4_gb = kv_gb * 0.53  # ~53% of FP16 (half + scale overhead)

                key = f"T={T}_B={B}"
                model_results["contexts"][key] = {
                    "T": T,
                    "B": B,
                    "kv_fp16_gb": round(kv_gb, 3),
                    "kv_int4_gb": round(kv_int4_gb, 3),
                }

        results[model] = model_results

    return results


def compute_tiering(kv_memory, model_cfgs):
    """For each GPU capacity, compute which configs fit in HBM."""
    tiering = {}

    for gpu_name, hbm_gb in HBM_CAPACITIES.items():
        gpu_results = {}
        for model, kv_data in kv_memory.items():
            weight_gb = MODEL_WEIGHT_GB.get(model, 14.0)
            available_gb = hbm_gb - weight_gb - 2.0  # 2GB overhead
            if available_gb < 0:
                available_gb = 0

            model_tier = {}
            for ctx_key, ctx_data in kv_data["contexts"].items():
                kv_fp16 = ctx_data["kv_fp16_gb"]
                kv_int4 = ctx_data["kv_int4_gb"]

                fits_fp16 = kv_fp16 <= available_gb
                fits_int4 = kv_int4 <= available_gb

                overflow_fp16 = max(0, kv_fp16 - available_gb)
                overflow_int4 = max(0, kv_int4 - available_gb)

                model_tier[ctx_key] = {
                    "fits_fp16": fits_fp16,
                    "fits_int4": fits_int4,
                    "overflow_fp16_gb": round(overflow_fp16, 3),
                    "overflow_int4_gb": round(overflow_int4, 3),
                    "available_gb": round(available_gb, 3),
                }

            gpu_results[model] = model_tier
        tiering[gpu_name] = gpu_results

    return tiering


def plot_memory_capacity(kv_memory):
    """Plot KV memory requirements by context length for key models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    key_models = [
        "qwen25_7b",
        "mistral_7b",
        "llama31_8b",
        "phi4_14b",
        "mistral_small_24b",
    ]
    colors = ["#1f77b4", "#d62728", "#9467bd", "#ffbb78", "#ff9896"]

    for ax_idx, B in enumerate([1, 32]):
        ax = axes[ax_idx]
        for model, color in zip(key_models, colors):
            kv_data = kv_memory[model]
            xs, ys_fp16, ys_int4 = [], [], []
            for T in CONTEXTS:
                key = f"T={T}_B={B}"
                if key in kv_data["contexts"]:
                    xs.append(T)
                    ys_fp16.append(kv_data["contexts"][key]["kv_fp16_gb"])
                    ys_int4.append(kv_data["contexts"][key]["kv_int4_gb"])

            ax.plot(xs, ys_fp16, "o-", color=color, label=f"{model} FP16", linewidth=2)
            ax.plot(
                xs,
                ys_int4,
                "s--",
                color=color,
                label=f"{model} INT4",
                linewidth=1.5,
                alpha=0.7,
            )

        # HBM capacity lines
        for gpu, cap in HBM_CAPACITIES.items():
            ax.axhline(y=cap, linestyle=":", color="gray", alpha=0.5)
            ax.text(CONTEXTS[0], cap + 1, gpu, fontsize=8, color="gray")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Context length (tokens)", fontsize=11)
        ax.set_ylabel("KV cache size (GB)", fontsize=11)
        ax.set_title(f"B = {B}", fontsize=13)
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "BPA v42: KV Cache Memory Requirements by Context Length",
        fontsize=14,
    )
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "memory_capacity_projection.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def plot_bandwidth_requirements(kv_memory):
    """Plot bandwidth needed to serve KV from secondary tier."""
    fig, ax = plt.subplots(figsize=(12, 7))

    key_models = [
        "qwen25_7b",
        "mistral_7b",
        "llama31_8b",
        "phi4_14b",
        "mistral_small_24b",
    ]
    colors = ["#1f77b4", "#d62728", "#9467bd", "#ffbb78", "#ff9896"]
    B = 32
    target_latency_ms = 10.0  # target decode latency

    for model, color in zip(key_models, colors):
        kv_data = kv_memory[model]
        xs, ys = [], []
        for T in CONTEXTS:
            key = f"T={T}_B={B}"
            if key in kv_data["contexts"]:
                kv_gb = kv_data["contexts"][key]["kv_int4_gb"]
                # Bandwidth needed = KV_bytes / target_latency
                bw_needed = kv_gb * 1024 / (target_latency_ms / 1000)  # GB/s
                xs.append(T)
                ys.append(bw_needed)

        ax.plot(xs, ys, "o-", color=color, label=model, linewidth=2)

    # Secondary tier bandwidth lines
    for tier, bw in SECONDARY_BW.items():
        ax.axhline(y=bw, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(CONTEXTS[-1] * 1.05, bw, tier, fontsize=9, va="center")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Context length (tokens)", fontsize=11)
    ax.set_ylabel("Required bandwidth (GB/s)", fontsize=11)
    ax.set_title(
        f"BPA v42: Secondary Tier Bandwidth Requirements\n"
        f"B={B}, INT4 KV, target decode latency = {target_latency_ms:.0f}ms",
        fontsize=14,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "bandwidth_projection.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading model configs...")
    model_cfgs = load_configs()

    print("Computing KV memory projections...")
    kv_memory = compute_kv_memory(model_cfgs)

    print("Computing tiering analysis...")
    tiering = compute_tiering(kv_memory, model_cfgs)

    # Save results
    with open(os.path.join(RESULTS_DIR, "memory_tiering.json"), "w") as f:
        json.dump({"kv_memory": kv_memory, "tiering": tiering}, f, indent=2)
    print(f"Saved {os.path.join(RESULTS_DIR, 'memory_tiering.json')}")

    print("Plotting memory capacity...")
    plot_memory_capacity(kv_memory)

    print("Plotting bandwidth requirements...")
    plot_bandwidth_requirements(kv_memory)

    print("Part 5 complete.")


if __name__ == "__main__":
    main()
