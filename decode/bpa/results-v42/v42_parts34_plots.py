#!/usr/bin/env python3
"""BPA v42 Parts 3-4: Layer sensitivity plots and adaptive precision analysis."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "artifacts/v42"
PLOTS_DIR = "plots"

MODEL_LABELS = {
    "qwen25_7b": "Qwen2.5-7B",
    "mistral_7b": "Mistral-7B",
    "llama2_7b": "Llama-2-7B",
}

MODEL_COLORS = {
    "qwen25_7b": "#2ca02c",
    "mistral_7b": "#d62728",
    "llama2_7b": "#9467bd",
}


def load_sensitivity():
    with open(os.path.join(RESULTS_DIR, "layer_sensitivity.json")) as f:
        return json.load(f)


def plot_layer_importance(data):
    """Plot per-layer importance scores for all 3 models."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

    for ax, (model, mdata) in zip(axes, data.items()):
        scores = mdata["layer_scores"]
        layers = [s["layer"] for s in scores]
        importances = [s["importance"] for s in scores]

        color = MODEL_COLORS[model]
        bars = ax.bar(
            layers,
            importances,
            color=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.3,
        )

        # Highlight top-4 layers
        ranked = sorted(range(len(importances)), key=lambda i: -importances[i])
        for i in ranked[:4]:
            bars[i].set_edgecolor("red")
            bars[i].set_linewidth(2)
            bars[i].set_alpha(1.0)

        delta = mdata["ppl_int4_all"] - mdata["ppl_fp16"]
        pct = delta / mdata["ppl_fp16"] * 100
        ax.set_title(
            f"{MODEL_LABELS[model]}: {mdata['n_layers']} layers  |  "
            f"FP16 PPL={mdata['ppl_fp16']:.3f}  |  "
            f"INT4-all PPL={mdata['ppl_int4_all']:.3f}  |  "
            f"delta={delta:.4f} ({pct:.2f}%)",
            fontsize=11,
        )
        ax.set_ylabel("Importance\n(INT4-all PPL - layer-i-FP16 PPL)", fontsize=9)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate top-4 layers
        for i in ranked[:4]:
            ax.annotate(
                f"L{layers[i]}",
                (layers[i], importances[i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="red",
            )

    axes[-1].set_xlabel("Layer index", fontsize=11)
    fig.suptitle(
        "BPA v42: Per-Layer KV Quantization Sensitivity\n"
        "Red-outlined bars = top-4 most sensitive layers",
        fontsize=14,
    )
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "layer_importance_curve.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def compute_adaptive_precision(data):
    """Part 4: Simulate adaptive precision configs.

    Configs:
    1. INT4 everywhere
    2. Top-4 layers INT8 (rest INT4)
    3. Top-8 layers INT8 (rest INT4)
    4. Adaptive threshold (layers with importance > mean + 1*std get INT8)

    For latency: estimate from v40 saturation model.
    For PPL: approximate from layer scores (additive model).
    """
    results = {}

    for model, mdata in data.items():
        scores = mdata["layer_scores"]
        n_layers = mdata["n_layers"]
        ppl_fp16 = mdata["ppl_fp16"]
        ppl_int4 = mdata["ppl_int4_all"]
        delta_all = ppl_int4 - ppl_fp16

        importances = [s["importance"] for s in scores]
        ranked_idx = sorted(range(n_layers), key=lambda i: -importances[i])

        mean_imp = np.mean(importances)
        std_imp = np.std(importances)
        threshold = mean_imp + 1.0 * std_imp
        adaptive_layers = [i for i in range(n_layers) if importances[i] > threshold]

        configs = {}

        # Config 1: INT4 everywhere
        configs["int4_all"] = {
            "n_int8": 0,
            "n_int4": n_layers,
            "ppl": ppl_int4,
            "kv_ratio": 0.53,  # INT4 with scale overhead
            "description": "INT4 everywhere",
        }

        # Config 2: Top-4 INT8
        top4 = set(ranked_idx[:4])
        # PPL improvement from keeping these 4 in FP16:
        # sum of their importances (additive approximation)
        ppl_improvement_4 = sum(importances[i] for i in top4)
        ppl_top4 = ppl_int4 - ppl_improvement_4
        # KV ratio: 4 layers INT8 (1.0x), rest INT4 (0.53x)
        kv_ratio_4 = (4 * 1.0 + (n_layers - 4) * 0.53) / n_layers
        configs["top4_int8"] = {
            "n_int8": 4,
            "n_int4": n_layers - 4,
            "int8_layers": sorted(top4),
            "ppl": ppl_top4,
            "kv_ratio": round(kv_ratio_4, 4),
            "description": "Top-4 sensitive layers INT8",
        }

        # Config 3: Top-8 INT8
        top8 = set(ranked_idx[:8])
        ppl_improvement_8 = sum(importances[i] for i in top8)
        ppl_top8 = ppl_int4 - ppl_improvement_8
        kv_ratio_8 = (8 * 1.0 + (n_layers - 8) * 0.53) / n_layers
        configs["top8_int8"] = {
            "n_int8": 8,
            "n_int4": n_layers - 8,
            "int8_layers": sorted(top8),
            "ppl": ppl_top8,
            "kv_ratio": round(kv_ratio_8, 4),
            "description": "Top-8 sensitive layers INT8",
        }

        # Config 4: Adaptive threshold
        n_adaptive = len(adaptive_layers)
        ppl_improvement_a = sum(importances[i] for i in adaptive_layers)
        ppl_adaptive = ppl_int4 - ppl_improvement_a
        kv_ratio_a = (n_adaptive * 1.0 + (n_layers - n_adaptive) * 0.53) / n_layers
        configs["adaptive"] = {
            "n_int8": n_adaptive,
            "n_int4": n_layers - n_adaptive,
            "int8_layers": sorted(adaptive_layers),
            "ppl": ppl_adaptive,
            "kv_ratio": round(kv_ratio_a, 4),
            "threshold": round(threshold, 6),
            "description": f"Adaptive (imp > {threshold:.6f}): {n_adaptive} INT8 layers",
        }

        # Config 5: All FP16 (baseline)
        configs["fp16_all"] = {
            "n_int8": n_layers,
            "n_int4": 0,
            "ppl": ppl_fp16,
            "kv_ratio": 1.0,
            "description": "FP16 everywhere",
        }

        results[model] = {
            "configs": configs,
            "n_layers": n_layers,
            "ppl_fp16": ppl_fp16,
        }

    return results


def plot_precision_tradeoff(adaptive_results, sensitivity_data):
    """Part 4 plot: precision vs speed/accuracy tradeoff."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (model, mdata) in zip(axes, adaptive_results.items()):
        configs = mdata["configs"]
        ppl_fp16 = mdata["ppl_fp16"]

        names = []
        kv_ratios = []
        ppls = []
        ppl_deltas = []

        for cname in ["fp16_all", "top8_int8", "top4_int8", "adaptive", "int4_all"]:
            cfg = configs[cname]
            names.append(cname.replace("_", "\n"))
            kv_ratios.append(cfg["kv_ratio"])
            ppls.append(cfg["ppl"])
            ppl_deltas.append(cfg["ppl"] - ppl_fp16)

        color = MODEL_COLORS[model]

        # Plot: x = KV ratio, y = PPL delta
        ax.scatter(kv_ratios, ppl_deltas, c=color, s=100, zorder=5, edgecolors="black")
        for i, name in enumerate(names):
            ax.annotate(
                name,
                (kv_ratios[i], ppl_deltas[i]),
                textcoords="offset points",
                xytext=(10, 5),
                fontsize=8,
            )

        # Connect with line
        order = sorted(range(len(kv_ratios)), key=lambda i: kv_ratios[i])
        ax.plot(
            [kv_ratios[i] for i in order],
            [ppl_deltas[i] for i in order],
            color=color,
            linewidth=1.5,
            alpha=0.5,
        )

        ax.set_xlabel("KV cache size ratio vs FP16", fontsize=10)
        ax.set_ylabel("PPL increase vs FP16 baseline", fontsize=10)
        ax.set_title(f"{MODEL_LABELS[model]}", fontsize=12)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "BPA v42: Adaptive Precision vs Quality Tradeoff\n"
        "Lower-right = better (smaller cache, lower PPL increase)",
        fontsize=14,
    )
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "precision_vs_speed_accuracy.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading layer sensitivity data...")
    data = load_sensitivity()

    print("Plotting layer importance...")
    plot_layer_importance(data)

    print("Computing adaptive precision configs...")
    adaptive = compute_adaptive_precision(data)

    # Save
    with open(os.path.join(RESULTS_DIR, "adaptive_precision.json"), "w") as f:
        json.dump(adaptive, f, indent=2)
    print(f"Saved {os.path.join(RESULTS_DIR, 'adaptive_precision.json')}")

    print("Plotting precision tradeoff...")
    plot_precision_tradeoff(adaptive, data)

    print("Parts 3-4 complete.")


if __name__ == "__main__":
    main()
