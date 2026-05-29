#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test Qwen3-VL-2B inference speed on NVIDIA Jetson Orin.

Usage:
    python scripts/test_orin_inference.py
    python scripts/test_orin_inference.py --model-name Qwen/Qwen3-VL-2B-Instruct

Orin Notes:
- bfloat16 is supported on Orin (compute capability 8.7)
- For float16 fallback, modify torch_dtype=torch.float16
- Flash Attention 2 may not be available; use SDPA if needed
"""

import argparse
import time

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_memory_usage():
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.memory_allocated() / 1024 / 1024


def main():
    parser = argparse.ArgumentParser(description="Test VLM inference speed on Orin")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of inference runs for benchmarking",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print("-" * 50)

    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Load model
    print("Loading model...")
    start_mem = get_memory_usage()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    load_mem = get_memory_usage() - start_mem
    print(f"Model loaded. Memory usage: {load_mem:.1f} MB")

    # Prepare test input (simple text-only for baseline)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the capital of France?"},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)

    # Warmup run
    print("\nRunning warmup...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50)

    # Benchmark runs
    print(f"Running {args.num_runs} benchmark iterations...")
    latencies = []
    for i in range(args.num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        # Count tokens
        num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        print(f"  Run {i+1}: {elapsed:.3f}s ({num_tokens} tokens, {num_tokens/elapsed:.1f} tok/s)")

    # Summary
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Average latency: {avg_latency:.3f}s")
    print(f"Min latency: {min_latency:.3f}s")
    print(f"Max latency: {max_latency:.3f}s")
    print(f"GPU memory: {get_memory_usage():.1f} MB")

    # Decode last output for reference
    response = processor.decode(outputs[0], skip_special_tokens=True)
    print(f"\nSample output:\n{response[-200:]}")


if __name__ == "__main__":
    main()
