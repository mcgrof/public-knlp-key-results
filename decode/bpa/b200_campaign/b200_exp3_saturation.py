#!/usr/bin/env python3
"""Experiment 3: Batch Saturation Sweep on B200.

Sweep batch 1->512 at T=4096, fit Hill model S(B) = Smax * B^g / (B_half^g + B^g).
"""

import csv
import json
import time
import gc
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_decode(model, tokenizer, device, batch_size, context_len, n_steps=16, n_warmup=3):
    """Measure decode tokens/sec."""
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
    del past, out
    torch.cuda.empty_cache()
    return tps


def hill_model(B, Smax, B_half, gamma):
    """Hill function: S(B) = Smax * B^gamma / (B_half^gamma + B^gamma)."""
    return Smax * np.power(B, gamma) / (np.power(B_half, gamma) + np.power(B, gamma))


def fit_hill(batch_sizes, throughputs):
    """Fit Hill model parameters using scipy."""
    from scipy.optimize import curve_fit
    bs = np.array(batch_sizes, dtype=float)
    tp = np.array(throughputs, dtype=float)

    try:
        popt, pcov = curve_fit(
            hill_model, bs, tp,
            p0=[tp[-1] * 2, 64.0, 1.0],
            bounds=([0, 0, 0.1], [1e8, 10000, 10]),
            maxfev=10000,
        )
        return {"Smax": popt[0], "B_half": popt[1], "gamma": popt[2]}
    except Exception as e:
        print("Hill fit failed: %s" % e)
        return None


def main():
    device = "cuda:0"
    model_name = "Qwen/Qwen2.5-7B"
    context_len = 4096

    print("Loading %s..." % model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device,
        trust_remote_code=True
    )
    model.eval()

    batch_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    results = []

    print("\nExperiment 3: Batch Saturation Sweep")
    print("Model: %s, Context: %d" % (model_name, context_len))
    print("%-8s %12s" % ("Batch", "tok/s"))
    print("-" * 22)

    for bs in batch_sizes:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            tps = measure_decode(model, tokenizer, device, bs, context_len)
            results.append({"batch_size": bs, "tokens_per_sec": round(tps, 1)})
            print("%-8d %12.1f" % (bs, tps))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("%-8d %12s" % (bs, "OOM"))
                torch.cuda.empty_cache()
                # Don't break, try smaller batch might have been a spike
                continue
            else:
                raise

    # Fit Hill model
    bs_list = [r["batch_size"] for r in results]
    tps_list = [r["tokens_per_sec"] for r in results]

    hill_params = fit_hill(bs_list, tps_list)
    if hill_params:
        print("\nHill model fit:")
        print("  Smax  = %.1f tok/s" % hill_params["Smax"])
        print("  B_half = %.1f" % hill_params["B_half"])
        print("  gamma  = %.3f" % hill_params["gamma"])

    # Save CSV
    csv_path = "/mnt/tmpfs/results/saturation_fit_b200.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["batch_size", "tokens_per_sec"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print("\nSaved: %s" % csv_path)

    # Save JSON
    json_path = "/mnt/tmpfs/results/saturation_fit_b200.json"
    with open(json_path, "w") as f:
        json.dump({
            "model": model_name,
            "context_len": context_len,
            "results": results,
            "hill_fit": hill_params,
        }, f, indent=2)
    print("Saved: %s" % json_path)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(bs_list, tps_list, "o-", color="steelblue", linewidth=2,
                markersize=6, label="Measured")

        if hill_params:
            bs_smooth = np.linspace(1, max(bs_list) * 1.2, 200)
            tps_fit = hill_model(bs_smooth, hill_params["Smax"],
                                 hill_params["B_half"], hill_params["gamma"])
            ax.plot(bs_smooth, tps_fit, "--", color="firebrick", linewidth=1.5,
                    label="Hill fit (Smax=%.0f, B\u00bd=%.0f, \u03b3=%.2f)" % (
                        hill_params["Smax"], hill_params["B_half"], hill_params["gamma"]))
            ax.axhline(y=hill_params["Smax"], color="gray", linestyle=":",
                       alpha=0.5, label="Smax=%.0f" % hill_params["Smax"])
            ax.axvline(x=hill_params["B_half"], color="gray", linestyle=":",
                       alpha=0.3)

        ax.set_xlabel("Batch Size", fontsize=12)
        ax.set_ylabel("Tokens/sec", fontsize=12)
        ax.set_title("B200 Batch Saturation (Qwen2.5-7B, T=4096)", fontsize=14)
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig_path = "/mnt/tmpfs/figures/saturation_curve_b200.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print("Saved: %s" % fig_path)
        plt.close()
    except Exception as e:
        print("Plot failed: %s" % e)

    print("\nExperiment 3 complete.")


if __name__ == "__main__":
    main()
