#!/usr/bin/env python3
"""Experiment 8: Capability Benchmarks on B200.

Run HellaSwag, MMLU, GSM8K via lm-eval harness.
Compare FP16 baseline vs INT4 KV quantized pipeline.
"""

import json
import subprocess
import sys
import os
import time


def run_lm_eval(model_name, tasks, output_dir, extra_args=None):
    """Run lm-eval harness and return results."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", "pretrained=%s,dtype=float16" % model_name,
        "--tasks", ",".join(tasks),
        "--batch_size", "auto",
        "--output_path", output_dir,
        "--log_samples",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print("Running: %s" % " ".join(cmd))
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print("STDERR: %s" % result.stderr[-2000:])
        return None, elapsed

    # Parse output
    print(result.stdout[-3000:])
    return result.stdout, elapsed


def parse_results(output_dir):
    """Parse lm-eval results from output directory."""
    results = {}
    for fname in os.listdir(output_dir):
        if fname.endswith(".json") and "results" in fname:
            fpath = os.path.join(output_dir, fname)
            with open(fpath) as f:
                data = json.load(f)
            if "results" in data:
                for task, metrics in data["results"].items():
                    acc = metrics.get("acc,none") or metrics.get("acc_norm,none")
                    if acc is not None:
                        results[task] = round(acc * 100, 2)
    return results


def main():
    model_name = "Qwen/Qwen2.5-7B"

    # HellaSwag and MMLU are the most important
    # GSM8K requires chain-of-thought which is slower
    tasks_fast = ["hellaswag", "mmlu"]
    tasks_slow = ["gsm8k"]

    all_results = {}

    # FP16 baseline
    print("=" * 60)
    print("FP16 Baseline Benchmarks")
    print("=" * 60)

    out_dir = "/mnt/tmpfs/results/benchmarks_fp16"
    stdout, elapsed = run_lm_eval(model_name, tasks_fast, out_dir)
    if stdout:
        results_fp16 = parse_results(out_dir)
        print("FP16 results: %s (%.0fs)" % (results_fp16, elapsed))
        all_results["fp16"] = results_fp16
    else:
        print("FP16 benchmark failed")
        all_results["fp16"] = {"error": "failed"}

    # Try GSM8K separately (can be slow)
    print("\n--- GSM8K ---")
    out_dir_gsm = "/mnt/tmpfs/results/benchmarks_fp16_gsm"
    stdout_gsm, elapsed_gsm = run_lm_eval(model_name, tasks_slow, out_dir_gsm,
                                            extra_args=["--num_fewshot", "5"])
    if stdout_gsm:
        results_gsm = parse_results(out_dir_gsm)
        print("GSM8K results: %s (%.0fs)" % (results_gsm, elapsed_gsm))
        if "fp16" in all_results and isinstance(all_results["fp16"], dict):
            all_results["fp16"].update(results_gsm)
    else:
        print("GSM8K failed (timeout or error)")

    # Save combined results
    json_path = "/mnt/tmpfs/results/benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "model": model_name,
            "gpu": "NVIDIA B200",
            "results": all_results,
        }, f, indent=2)
    print("\nSaved: %s" % json_path)

    # CSV
    csv_path = "/mnt/tmpfs/results/benchmark_results.csv"
    with open(csv_path, "w") as f:
        f.write("pipeline,hellaswag,mmlu,gsm8k\n")
        for pipeline, res in all_results.items():
            if isinstance(res, dict) and "error" not in res:
                hs = res.get("hellaswag", "")
                mmlu = res.get("mmlu", "")
                gsm = res.get("gsm8k", "")
                f.write("%s,%s,%s,%s\n" % (pipeline, hs, mmlu, gsm))
    print("Saved: %s" % csv_path)

    print("\nExperiment 8 complete.")


if __name__ == "__main__":
    main()
