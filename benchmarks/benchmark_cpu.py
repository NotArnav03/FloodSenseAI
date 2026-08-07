import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import psutil
import torch
import numpy as np
from models.flood_model import FloodModelRunner, UNetMobileNetV2, LightweightFloodSegmentation


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 ** 2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / (1024 ** 2)


def run_benchmark(image_path, runs=30, warmup=5, use_lightweight=False):
    print("===== EDGE-DRIVE CPU BENCHMARK =====\n")

    runner = FloodModelRunner(use_lightweight=use_lightweight)
    model = runner.model

    model_name = "LightweightFloodSegmentation" if use_lightweight else "UNetMobileNetV2"
    print(f"Model: {model_name}")
    print(f"Device: {runner.device}")
    print(f"Model Parameters: {count_parameters(model):,}")
    print(f"Model Size (MB): {model_size_mb(model):.2f}")
    print(f"Initial Process Memory (MB): {get_memory_usage():.2f}")

    # Preprocess once
    image, tensor, h, w = runner.preprocess(image_path)
    tensor = tensor.to(runner.device)

    # Warmup
    print("\nWarming up...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(tensor)
    print("Warmup complete.\n")

    latencies = []
    cpu_usages = []

    for i in range(runs):
        cpu_start = psutil.cpu_percent(interval=None)

        start = time.time()
        with torch.no_grad():
            _ = model(tensor)
        end = time.time()

        cpu_end = psutil.cpu_percent(interval=None)

        latency = (end - start) * 1000
        latencies.append(latency)
        cpu_usages.append(cpu_end)

        print(f"Run {i+1}: {latency:.2f} ms | CPU Usage: {cpu_end}%")

    latencies = np.array(latencies)
    cpu_usages = np.array(cpu_usages)

    print(f"\n===== PERFORMANCE SUMMARY ({model_name}) =====")
    print(f"Average Latency: {latencies.mean():.2f} ms")
    print(f"Std Deviation: {latencies.std():.2f} ms")
    print(f"Min Latency: {latencies.min():.2f} ms")
    print(f"Max Latency: {latencies.max():.2f} ms")
    print(f"Throughput (FPS): {1000 / latencies.mean():.2f}")
    print(f"Average CPU Usage: {cpu_usages.mean():.2f}%")
    print(f"Final Process Memory (MB): {get_memory_usage():.2f}")
    print("=" * 50)


if __name__ == "__main__":
    image_path = os.path.join("data", "raw", "sample_flood.png")

    # Benchmark both models for comparison
    print("\n" + "=" * 60)
    print("BENCHMARKING U-NET (FULL ACCURACY MODEL)")
    print("=" * 60)
    run_benchmark(image_path, use_lightweight=False)

    print("\n\n" + "=" * 60)
    print("BENCHMARKING LIGHTWEIGHT (EDGE DEPLOYMENT MODEL)")
    print("=" * 60)
    run_benchmark(image_path, use_lightweight=True)
