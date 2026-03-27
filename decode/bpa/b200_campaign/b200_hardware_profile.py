#!/usr/bin/env python3
"""B200 Hardware Profiling - Roofline model parameters."""

import json
import time
import torch
import torch.nn.functional as F


def measure_hbm_bandwidth(device, sizes_mb=[64, 128, 256, 512, 1024], n_iter=50):
    """Measure HBM bandwidth via large memcpy."""
    results = []
    for size_mb in sizes_mb:
        n_elements = size_mb * 1024 * 1024 // 4  # float32
        a = torch.randn(n_elements, device=device, dtype=torch.float32)
        b = torch.empty_like(a)

        # warmup
        for _ in range(10):
            b.copy_(a)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iter):
            b.copy_(a)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        bytes_moved = 2 * n_elements * 4 * n_iter  # read + write
        bw_gb_s = bytes_moved / elapsed / 1e9
        results.append({"size_mb": size_mb, "bandwidth_gb_s": bw_gb_s})
        print("  %d MB -> %.1f GB/s" % (size_mb, bw_gb_s))

    return results


def measure_fp16_flops(device, M=4096, N=4096, K=4096, n_iter=100):
    """Measure FP16 GEMM throughput."""
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)

    # warmup
    for _ in range(20):
        torch.mm(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    flops_per_mm = 2 * M * N * K
    total_flops = flops_per_mm * n_iter
    tflops = total_flops / elapsed / 1e12
    print("  FP16 GEMM (%dx%dx%d): %.1f TFLOPS" % (M, N, K, tflops))
    return tflops


def measure_fp16_flops_large(device, sizes=None, n_iter=50):
    """Measure FP16 GEMM at multiple sizes to find peak."""
    if sizes is None:
        sizes = [(4096, 4096, 4096), (8192, 8192, 8192),
                 (16384, 16384, 16384)]
    results = []
    for M, N, K in sizes:
        try:
            a = torch.randn(M, K, device=device, dtype=torch.float16)
            b = torch.randn(K, N, device=device, dtype=torch.float16)
            for _ in range(10):
                torch.mm(a, b)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(n_iter):
                torch.mm(a, b)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            flops_per_mm = 2 * M * N * K
            tflops = flops_per_mm * n_iter / elapsed / 1e12
            results.append({"M": M, "N": N, "K": K, "tflops": tflops})
            print("  %dx%dx%d: %.1f TFLOPS" % (M, N, K, tflops))
            del a, b
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print("  %dx%dx%d: OOM (%s)" % (M, N, K, e))
    return results


def measure_memory_latency(device, n_iter=10000):
    """Estimate memory latency via dependent pointer-chase (approx)."""
    # Small random access pattern
    size = 1024 * 1024  # 1M elements
    idx = torch.randint(0, size, (n_iter,), device=device, dtype=torch.long)
    data = torch.randn(size, device=device, dtype=torch.float32)

    # warmup
    for _ in range(100):
        _ = data[idx[:100]].sum()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for i in range(0, n_iter, 100):
        _ = data[idx[i:i+100]].sum()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ns = elapsed / (n_iter // 100) * 1e9
    print("  Approx gather latency: %.0f ns per batch of 100" % avg_ns)
    return avg_ns


def main():
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)

    print("=" * 60)
    print("B200 Hardware Profile")
    print("=" * 60)
    print("GPU: %s" % props.name)
    print("Memory: %.1f GB" % (props.total_memory / 1024**3))
    print("SMs: %d" % props.multi_processor_count)
    print("Compute: %d.%d" % (props.major, props.minor))
    print()

    # HBM Bandwidth
    print("--- HBM Bandwidth ---")
    bw_results = measure_hbm_bandwidth(device)
    peak_bw = max(r["bandwidth_gb_s"] for r in bw_results)
    print("Peak measured: %.1f GB/s" % peak_bw)
    print()

    # FP16 FLOPS
    print("--- FP16 GEMM Throughput ---")
    flops_results = measure_fp16_flops_large(device)
    peak_flops = max(r["tflops"] for r in flops_results)
    print("Peak measured: %.1f TFLOPS" % peak_flops)
    print()

    # Memory Latency
    print("--- Memory Latency (approx) ---")
    latency_ns = measure_memory_latency(device)
    print()

    # Ridge point
    ridge = peak_flops * 1e12 / (peak_bw * 1e9)
    print("--- Roofline ---")
    print("Ridge point: %.1f FLOP/byte" % ridge)
    print("Operations below %.1f FLOP/byte are memory-bound" % ridge)
    print()

    # Save results
    result = {
        "gpu": props.name,
        "memory_gb": round(props.total_memory / 1024**3, 1),
        "sm_count": props.multi_processor_count,
        "compute_capability": "%d.%d" % (props.major, props.minor),
        "peak_hbm_bandwidth_gb_s": round(peak_bw, 1),
        "peak_fp16_tflops": round(peak_flops, 1),
        "ridge_point_flop_per_byte": round(ridge, 1),
        "approx_gather_latency_ns": round(latency_ns, 0),
        "bandwidth_by_size": bw_results,
        "flops_by_size": flops_results,
        "driver_version": torch.cuda.get_device_properties(0).name,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    out_path = "/home/mcgrof/paper-artifacts/b200_roofline.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved: %s" % out_path)

    # Also save to results dir
    out_path2 = "/mnt/tmpfs/results/b200_roofline.json"
    with open(out_path2, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved: %s" % out_path2)


if __name__ == "__main__":
    main()
