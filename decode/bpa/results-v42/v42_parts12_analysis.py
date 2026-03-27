#!/usr/bin/env python3
"""BPA v42 Parts 1-2: Improved universal scaling plots.

Part 1: Speedup vs kv_bytes_per_token (remove B/T confounding)
Part 2: Speedup vs arithmetic intensity with roofline overlay
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUMMARY_CSV = "artifacts/v40/bench_results_summary.csv"
MODEL_CONFIGS = "artifacts/v40/model_configs.json"
HW_LIMITS = "artifacts/v40/hardware_limits.json"
SCALING_LAW = "scaling_law.json"
PLOTS_DIR = "plots"

PIPE_COLORS = {
    "P0": "#2ca02c",
    "P1": "#ff7f0e",
    "P3": "#1f77b4",
    "P5": "#d62728",
}
PIPE_LABELS = {
    "P0": "Dense FP16 (SDPA)",
    "P1": "INT4 + dequant",
    "P3": "Fused INT4",
    "P5": "Unified fused",
}

MODEL_MARKERS = {
    "qwen25_05b": "o",
    "qwen25_1.8b": "s",
    "qwen25_7b": "^",
    "mistral_7b": "D",
    "llama31_8b": "v",
    "nemotron3_nano": "P",
    "deepseek_mla": "*",
    "gemma3_4b": "X",
    "gemma3_12b": "h",
    "gemma3n_e4b": "p",
    "deepseek_r1_7b": "<",
    "phi4_14b": ">",
    "qwen3_8b": "d",
    "mistral_small_24b": "H",
}

MODEL_SHORT = {
    "qwen25_05b": "Qwen2.5-0.5B",
    "qwen25_1.8b": "Qwen2.5-1.8B",
    "qwen25_7b": "Qwen2.5-7B",
    "mistral_7b": "Mistral-7B",
    "llama31_8b": "Llama3.1-8B",
    "nemotron3_nano": "Nemotron-Nano",
    "deepseek_mla": "DeepSeek-MLA",
    "gemma3_4b": "Gemma3-4B",
    "gemma3_12b": "Gemma3-12B",
    "gemma3n_e4b": "Gemma3n-E4B",
    "deepseek_r1_7b": "DS-R1-7B",
    "phi4_14b": "Phi-4-14B",
    "qwen3_8b": "Qwen3-8B",
    "mistral_small_24b": "Mistral-24B",
}


def load_data():
    with open(MODEL_CONFIGS) as f:
        model_cfgs = json.load(f)
    with open(HW_LIMITS) as f:
        hw = json.load(f)
    with open(SCALING_LAW) as f:
        scaling = json.load(f)

    rows = {}
    with open(SUMMARY_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["config"], row["mode"], int(row["T_kv"]), int(row["B"]))
            if key not in rows:
                rows[key] = {}
            rows[key][row["pipeline"]] = {
                "mean_ms": float(row["mean_ms"]),
                "cv_pct": float(row["cv_pct"]),
            }

    return model_cfgs, hw, scaling, rows


def compute_speedup_by_model(model_cfgs, rows):
    """Compute mean speedup per model (averaging over T and B),
    keyed by kv_bytes_per_token. This removes B/T confounding."""
    from collections import defaultdict

    # For each model and pipeline pair, collect speedup = P_x / P0
    model_pipe_speedups = defaultdict(lambda: defaultdict(list))

    for (config, mode, T_kv, B), pipelines in rows.items():
        if mode != "decode":
            continue
        if B < 4:
            continue
        if "P0" not in pipelines:
            continue
        p0_ms = pipelines["P0"]["mean_ms"]
        if p0_ms <= 0:
            continue

        for pipe in ["P1", "P3", "P5"]:
            if pipe not in pipelines:
                continue
            p_ms = pipelines[pipe]["mean_ms"]
            if p_ms <= 0:
                continue
            speedup = p0_ms / p_ms
            model_pipe_speedups[config][pipe].append(speedup)

    results = {}
    for config, pipe_data in model_pipe_speedups.items():
        cfg = model_cfgs[config]
        kv_bpt = cfg["kv_bytes_per_token"]
        results[config] = {
            "kv_bpt": kv_bpt,
            "head_dim": cfg["head_dim"],
            "gqa_ratio": cfg["gqa_ratio"],
            "n_heads": cfg["n_heads"],
            "n_kv_heads": cfg["n_kv_heads"],
        }
        for pipe, su_list in pipe_data.items():
            results[config][f"{pipe}_mean"] = float(np.mean(su_list))
            results[config][f"{pipe}_std"] = float(np.std(su_list))
            results[config][f"{pipe}_n"] = len(su_list)

    return results


def plot_speedup_vs_kv_bpt(model_data):
    """Part 1: Speedup vs kv_bytes_per_token, colored by pipeline."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for pipe, color in [
        ("P1", PIPE_COLORS["P1"]),
        ("P3", PIPE_COLORS["P3"]),
        ("P5", PIPE_COLORS["P5"]),
    ]:
        xs, ys, yerr, labels = [], [], [], []
        for model, data in sorted(model_data.items()):
            key_mean = f"{pipe}_mean"
            key_std = f"{pipe}_std"
            if key_mean not in data:
                continue
            xs.append(data["kv_bpt"])
            ys.append(data[key_mean])
            yerr.append(data[key_std])
            labels.append(model)

        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt="o",
            color=color,
            markersize=8,
            capsize=4,
            linewidth=1.5,
            label=PIPE_LABELS[pipe],
            alpha=0.8,
        )

        # Annotate each point with model name
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(
                MODEL_SHORT[label],
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=6,
                alpha=0.7,
            )

    ax.set_xscale("log")
    ax.set_xlabel("KV bytes per token (log scale)", fontsize=12)
    ax.set_ylabel("Mean decode speedup vs FP16 SDPA (P0)", fontsize=12)
    ax.set_title(
        "BPA v42: Speedup vs KV Cache Size per Token\n"
        "14 models, decode B>=4, averaged over T and B",
        fontsize=14,
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "speedup_vs_KV_bytes_per_token.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def compute_ai_and_speedup(model_cfgs, rows):
    """Compute arithmetic intensity and speedup for each config."""
    results = []
    for (config, mode, T_kv, B), pipelines in rows.items():
        if mode != "decode":
            continue
        if B < 4:
            continue
        if "P0" not in pipelines:
            continue
        p0_ms = pipelines["P0"]["mean_ms"]
        if p0_ms <= 0:
            continue

        cfg = model_cfgs[config]
        kv_bpt = cfg["kv_bytes_per_token"]
        n_heads = cfg["n_heads"]
        hd = cfg["head_dim"]

        # Arithmetic intensity = FLOPs / bytes per query token
        # FLOPs = 4 * n_heads * head_dim * T_kv (Q@K^T + attn@V)
        # Bytes = kv_bpt * T_kv (read KV cache)
        # AI = 4 * n_heads * head_dim / kv_bpt (T cancels!)
        ai = 4 * n_heads * hd / kv_bpt

        for pipe in ["P1", "P3", "P5"]:
            if pipe not in pipelines:
                continue
            p_ms = pipelines[pipe]["mean_ms"]
            if p_ms <= 0:
                continue
            speedup = p0_ms / p_ms
            results.append(
                {
                    "config": config,
                    "T_kv": T_kv,
                    "B": B,
                    "pipe": pipe,
                    "ai": ai,
                    "speedup": speedup,
                    "kv_bpt": kv_bpt,
                }
            )

    return results


def plot_speedup_vs_ai(ai_data, hw):
    """Part 2: Speedup vs arithmetic intensity, with ridge overlay."""
    fig, ax = plt.subplots(figsize=(12, 8))

    peak_flops = hw["peak_fp16_TFLOPs"] * 1e12
    peak_bw = hw["hbm_bandwidth_GBs"] * 1e9
    ridge = peak_flops / peak_bw

    for pipe in ["P1", "P3", "P5"]:
        pts = [r for r in ai_data if r["pipe"] == pipe]
        if not pts:
            continue
        # Group by model to show mean
        from collections import defaultdict

        by_model = defaultdict(list)
        for p in pts:
            by_model[p["config"]].append(p)

        xs, ys, yerr = [], [], []
        for model in sorted(by_model):
            group = by_model[model]
            ai = group[0]["ai"]
            sus = [g["speedup"] for g in group]
            xs.append(ai)
            ys.append(np.mean(sus))
            yerr.append(np.std(sus))

        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt="o",
            color=PIPE_COLORS[pipe],
            markersize=7,
            capsize=3,
            linewidth=1,
            label=PIPE_LABELS[pipe],
            alpha=0.7,
        )

    ax.axvline(
        x=ridge,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"H100 ridge = {ridge:.0f} FLOP/byte",
    )

    # Annotate memory-bound region
    ax.axvspan(0, ridge, alpha=0.05, color="blue")
    ax.text(
        ridge * 0.5,
        ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 1 else 3.5,
        "memory-bound",
        ha="center",
        fontsize=10,
        color="blue",
        alpha=0.5,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Mean decode speedup vs FP16 SDPA (P0)", fontsize=12)
    ax.set_title(
        "BPA v42: Speedup vs Arithmetic Intensity\n"
        "14 models, decode B>=4, H100 ridge overlay",
        fontsize=14,
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "speedup_vs_arithmetic_intensity.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading data...")
    model_cfgs, hw, scaling, rows = load_data()

    print("Part 1: Speedup vs kv_bytes_per_token...")
    model_data = compute_speedup_by_model(model_cfgs, rows)
    plot_speedup_vs_kv_bpt(model_data)

    # Save model_data for report
    with open("artifacts/v42/part1_model_speedups.json", "w") as f:
        json.dump(model_data, f, indent=2)
    print("  Saved artifacts/v42/part1_model_speedups.json")

    print("Part 2: Speedup vs arithmetic intensity...")
    ai_data = compute_ai_and_speedup(model_cfgs, rows)
    plot_speedup_vs_ai(ai_data, hw)

    # Compute per-model AI values for report
    ai_by_model = {}
    for model, data in model_data.items():
        cfg = model_cfgs[model]
        ai = 4 * cfg["n_heads"] * cfg["head_dim"] / cfg["kv_bytes_per_token"]
        ai_by_model[model] = float(ai)
    with open("artifacts/v42/part2_arithmetic_intensity.json", "w") as f:
        json.dump(ai_by_model, f, indent=2)
    print("  Saved artifacts/v42/part2_arithmetic_intensity.json")

    print("Parts 1-2 complete.")


if __name__ == "__main__":
    main()
