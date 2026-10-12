# SPDX-License-Identifier: Apache-2.0
"""Byte validation and synchronized latency of reference versus fused kernels."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization.nvfp4_kv_cache import (
    NPUNVFP4KVCacheMethod,
)


def measure(fn, repeats=8):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    method = NPUNVFP4KVCacheMethod(
        workspace_dtype=torch.bfloat16, num_layers=2, device="npu", page_size=16
    )
    results = {"quantization": [], "dequantization": []}
    for heads, dim, tokens, strided in (
        (1, 64, 1, False),
        (3, 64, 19, False),
        (8, 128, 1, False),
        (8, 128, 256, False),
        (3, 80, 17, False),
        (8, 128, 19, True),
    ):
        for dtype in (torch.bfloat16, torch.float16):
            print(
                f"CHECK heads={heads}, dim={dim}, tokens={tokens}, dtype={dtype}",
                flush=True,
            )
            source = (
                torch.randn(
                    tokens,
                    heads * (3 if strided else 1),
                    dim,
                    generator=torch.Generator().manual_seed(73),
                )
                .to(dtype)
                .npu()
            )
            source = source[:, :heads, :]
            loc = torch.arange(tokens, device="npu")
            packed = torch.empty(
                tokens, heads, dim // 2, dtype=torch.uint8, device="npu"
            )
            scales = torch.empty(
                tokens, heads, dim // 16, dtype=torch.uint8, device="npu"
            )
            globals_ = torch.empty(tokens, 1, 1, device="npu")
            global_scale = (
                source.float().abs().amax((-2, -1), keepdim=True) / 2688
            ).clamp_min(2**-126)
            expected_data, expected_scales = method._quantize(source, global_scale)

            def reference():
                g = (
                    source.float().abs().amax((-2, -1), keepdim=True) / 2688
                ).clamp_min(2**-126)
                data, block_scales = method._quantize(source, g)
                packed[loc] = data
                scales[loc] = block_scales
                globals_[loc] = g

            def fused():
                from sglang.kernels.ops.quantization import npu_nvfp4_quantize_store

                npu_nvfp4_quantize_store(
                    source, loc, packed, scales, globals_, method._e4m3
                )

            fused()
            torch.testing.assert_close(
                packed.cpu(), expected_data.cpu(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                scales.cpu(), expected_scales.cpu(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                globals_.cpu(), global_scale.cpu(), rtol=1e-6, atol=0
            )
            workspace = torch.empty(source.shape, dtype=dtype, device="npu")
            method.dequantize_rows_into(packed, scales, globals_, loc, workspace)
            expected_values = method._dequantize(
                expected_data, expected_scales, global_scale
            ).to(dtype)
            torch.testing.assert_close(
                workspace.cpu(), expected_values.cpu(), rtol=0, atol=0
            )
            results["quantization"].append(
                {
                    "heads": heads,
                    "dim": dim,
                    "tokens": tokens,
                    "dtype": str(dtype),
                    "source_strides": list(source.stride()),
                    "reference": measure(reference),
                    "fused": measure(fused),
                }
            )
    for capacity in (2048, 16384, 65536):
        packed = torch.zeros(capacity, 8, 64, dtype=torch.uint8, device="npu")
        scales = torch.zeros(capacity, 8, 8, dtype=torch.uint8, device="npu")
        globals_ = torch.ones(capacity, 1, 1, device="npu")
        workspace = torch.empty(capacity, 8, 128, dtype=torch.bfloat16, device="npu")
        slots = torch.arange(512, device="npu")
        results["dequantization"].append(
            {
                "capacity": capacity,
                "active_slots": 512,
                "whole_pool": measure(
                    lambda: method.dequantize_into(packed, scales, globals_, workspace),
                    repeats=4,
                ),
                "live_fused": measure(
                    lambda: method.dequantize_rows_into(
                        packed, scales, globals_, slots, workspace
                    )
                ),
            }
        )
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
