# SPDX-License-Identifier: Apache-2.0
"""Compare Ascend software NVFP4 with main's decoder and CPU rounding.

Run directly with ``--device cpu`` or ``--device npu --npu-device 0``. The
independent decoder runs on CPU: raw FP8 casts are not supported on every NPU.
"""

import argparse
import unittest

import torch


def reference_quantize(source, global_scale):
    """CPU E4M3 cast and nearest E2M1 value, with an independent tie rule."""
    source = source.cpu().float()
    blocks = (source / global_scale).reshape(*source.shape[:-1], -1, 16)
    block_scales = (blocks.abs().amax(-1) / 6).clamp(max=448).to(torch.float8_e4m3fn)
    normalized = blocks / block_scales.float().clamp_min(2**-9).unsqueeze(-1)
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)
    distance = (normalized.abs().clamp(max=6).double().unsqueeze(-1) - values).abs()
    nearest = distance == distance.amin(-1, keepdim=True)
    indices = torch.arange(8)
    candidates = torch.where(nearest, indices, 8)
    even_candidates = torch.where(nearest & (indices % 2 == 0), indices, 8)
    preferred_even = even_candidates.amin(-1)
    magnitude = torch.where(preferred_even < 8, preferred_even, candidates.amin(-1))
    codes = (magnitude + (normalized < 0).long() * 8).reshape_as(source)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    return packed, block_scales.view(torch.uint8)


class TestNPUNVFP4Reference(unittest.TestCase):
    device = "cpu"

    def make_method(self, dtype=torch.bfloat16):
        from sglang.srt.hardware_backend.npu.quantization.nvfp4_kv_cache import (
            NPUNVFP4KVCacheMethod,
        )

        return NPUNVFP4KVCacheMethod(
            workspace_dtype=dtype, num_layers=1, device=self.device, page_size=16
        )

    def assert_bits_equal(self, actual, expected):
        actual = actual.cpu().contiguous()
        expected = expected.cpu().contiguous()
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(
            actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
        )

    def test_all_dequant_bytes_scales_and_signed_zero(self):
        from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4

        # Every possible pair of E2M1 codes and every finite nonnegative E4M3
        # code, including scale zero. Dyadic global scales make the two valid
        # multiplication orders exactly comparable, including signed zero.
        packed = torch.arange(256, dtype=torch.uint8).reshape(256, 1, 1)
        packed = packed.expand(256, 127, 8).contiguous()
        scales = torch.arange(127, dtype=torch.uint8).reshape(1, 127, 1)
        scales = scales.expand(256, 127, 1).contiguous()
        decoded_scales = scales.view(torch.float8_e4m3fn).float()
        row_scales = torch.tensor([0.25, 0.5, 1.0, 2.0]).repeat(64).reshape(-1, 1, 1)
        loc = torch.randperm(256, generator=torch.Generator().manual_seed(123)) + 5
        untouched = torch.ones(267, dtype=torch.bool)
        untouched[loc] = False
        for dtype in (torch.bfloat16, torch.float16):
            for global_scale in (torch.tensor(0.5), row_scales):
                with self.subTest(dtype=dtype, global_shape=global_scale.shape):
                    method = self.make_method(dtype)
                    expected = dequantize_nvfp4(
                        packed, decoded_scales, global_scale, out_dtype=dtype
                    )
                    decoded = method._dequantize(
                        packed.to(self.device),
                        scales.to(self.device),
                        global_scale.to(self.device),
                    ).to(dtype)
                    self.assert_bits_equal(decoded, expected)

                    stored = torch.full(
                        (267, 127, 8), 171, dtype=torch.uint8, device=self.device
                    )
                    stored_scales = torch.full(
                        (267, 127, 1), 171, dtype=torch.uint8, device=self.device
                    )
                    globals_ = torch.full(
                        (267, 1, 1), -9, dtype=torch.float32, device=self.device
                    )
                    slots = loc.to(self.device)
                    stored[slots] = packed.to(self.device)
                    stored_scales[slots] = scales.to(self.device)
                    globals_[slots] = global_scale.to(self.device)
                    workspace = torch.full(
                        (267, 127, 16), float("nan"), dtype=dtype, device=self.device
                    )
                    # The slot list is unsorted and the destination has holes;
                    # each physical row is decoded exactly once.
                    method.dequantize_rows_into(
                        stored, stored_scales, globals_, slots, workspace
                    )
                    self.assert_bits_equal(workspace.cpu()[loc], expected)
                    self.assertTrue(torch.isnan(workspace.cpu()[untouched]).all())

    def check_quantized_store(self, source_cpu, *, strided, global_scale=None):
        tokens, heads, dim = source_cpu.shape
        if strided:
            storage = torch.zeros(
                (tokens * 2, heads * 2, dim),
                dtype=source_cpu.dtype,
                device=self.device,
            )
            source = storage[::2, ::2, :]
            source.copy_(source_cpu.to(self.device))
            self.assertEqual(source.stride(-1), 1)
            self.assertFalse(source.is_contiguous())
        else:
            source = source_cpu.to(self.device)
        capacity = tokens + 11
        loc = torch.randperm(tokens, generator=torch.Generator().manual_seed(11)) + 5
        untouched = torch.ones(capacity, dtype=torch.bool)
        untouched[loc] = False
        packed = torch.full(
            (capacity, heads, dim // 2), 171, dtype=torch.uint8, device=self.device
        )
        scales = torch.full(
            (capacity, heads, dim // 16), 171, dtype=torch.uint8, device=self.device
        )
        globals_ = torch.full(
            (capacity, 1, 1), -9, dtype=torch.float32, device=self.device
        )
        theoretical_global = (
            (source_cpu.float().abs().amax((-2, -1), keepdim=True) / 2688.0).clamp_min(
                2**-126
            )
            if global_scale is None
            else torch.full((tokens, 1, 1), global_scale, dtype=torch.float32)
        )
        expected_global = theoretical_global
        if self.device == "npu" and global_scale is None:
            # Independently run the pre-optimization Torch path. Ascend's
            # scalar division can differ from the CPU quotient by one ULP;
            # the fused reduction must preserve its existing platform bits.
            expected_global = (
                (source.float().abs().amax((-2, -1), keepdim=True) / 2688.0)
                .clamp_min(2**-126)
                .cpu()
            )
            distance = (
                expected_global.view(torch.int32).long()
                - theoretical_global.view(torch.int32).long()
            ).abs()
            self.assertLessEqual(int(distance.max()), 1)
            minimum_rows = theoretical_global == 2**-126
            torch.testing.assert_close(
                expected_global.view(torch.int32)[minimum_rows],
                theoretical_global.view(torch.int32)[minimum_rows],
                rtol=0,
                atol=0,
            )
        expected_packed, expected_scales = reference_quantize(
            source_cpu, expected_global
        )
        method = self.make_method(
            torch.float16 if source_cpu.dtype == torch.float16 else torch.bfloat16
        )
        method.quantize_and_store(
            packed,
            packed,
            scales,
            scales,
            loc.to(self.device),
            source,
            source,
            global_scale,
            global_scale,
            k_global_scale_buffer=globals_,
            v_global_scale_buffer=globals_,
        )
        torch.testing.assert_close(
            globals_.cpu()[loc].view(torch.int32),
            expected_global.view(torch.int32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(packed.cpu()[loc], expected_packed, rtol=0, atol=0)
        torch.testing.assert_close(scales.cpu()[loc], expected_scales, rtol=0, atol=0)
        self.assertTrue((packed.cpu()[untouched] == 171).all())
        self.assertTrue((scales.cpu()[untouched] == 171).all())
        self.assertTrue((globals_.cpu()[untouched] == -9).all())

    def test_online_global_reduction_and_bytes_across_shapes(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for heads, dim in ((3, 80), (8, 128), (33, 128), (64, 128)):
                source = torch.randn(
                    19, heads, dim, generator=torch.Generator().manual_seed(42)
                )
                source[0] = 0
                source[1] *= 1e-3
                source[2] *= 1e3
                source = source.to(dtype)
                for strided in (False, True):
                    with self.subTest(dtype=dtype, shape=source.shape, strided=strided):
                        self.check_quantized_store(source, strided=strided)

    def test_explicit_global_scales(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            source = torch.randn(
                19, 3, 80, generator=torch.Generator().manual_seed(7)
            ).to(dtype)
            source[0] = 0
            source[1] *= 1e-3
            source[2, 0, :8] = torch.tensor(
                [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], dtype=dtype
            )
            for global_scale in (1.0, 0.1):
                with self.subTest(dtype=dtype, global_scale=global_scale):
                    self.check_quantized_store(
                        source, strided=True, global_scale=global_scale
                    )

    def test_extreme_fp32_inputs_keep_saturating_codes(self):
        # A finite quotient may overflow its multiply-back product. The
        # residual correction must preserve saturation rather than emit NaN
        # and silently select an all-zero FP4 code.
        source = torch.full((2, 1, 16), torch.finfo(torch.float32).max)
        source[..., 1::2] *= -1
        for global_scale in (None, 0.1):
            with self.subTest(global_scale=global_scale):
                self.check_quantized_store(
                    source, strided=False, global_scale=global_scale
                )

    def test_fp4_midpoints_for_every_positive_e4m3_scale(self):
        # Choose each block's maximum so its encoded scale is known. Check
        # exact ties and four neighboring FP32 values on each side, rather
        # than relying on random inputs to find discontinuous rounding cases.
        scales = (
            torch.arange(1, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        )
        midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        thresholds = (scales[:, None] * midpoints[None, :]).flatten()
        neighbors = (
            (thresholds.view(torch.int32)[:, None] + torch.arange(-4, 5)[None, :])
            .to(torch.int32)
            .view(torch.float32)
            .flatten()
        )
        source = torch.zeros(neighbors.numel() + 1, 1, 16)
        source[:-1, 0, 0] = neighbors
        source[:-1, 0, 1] = -neighbors
        source[:-1, 0, -1] = (
            scales[:, None].expand(-1, 7).flatten() * 6
        ).repeat_interleave(9)
        self.check_quantized_store(source, strided=False, global_scale=1.0)

    def test_empty_selections_leave_buffers_unchanged(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                method = self.make_method(dtype)
                packed = torch.full(
                    (8, 3, 40), 171, dtype=torch.uint8, device=self.device
                )
                scales = torch.full(
                    (8, 3, 5), 171, dtype=torch.uint8, device=self.device
                )
                globals_ = torch.full(
                    (8, 1, 1), -9, dtype=torch.float32, device=self.device
                )
                workspace = torch.full((8, 3, 80), 17, dtype=dtype, device=self.device)
                slots = torch.empty(0, dtype=torch.int64, device=self.device)
                source = torch.empty(0, 3, 80, dtype=dtype, device=self.device)
                method.quantize_and_store(
                    packed,
                    packed,
                    scales,
                    scales,
                    slots,
                    source,
                    source,
                    k_global_scale_buffer=globals_,
                    v_global_scale_buffer=globals_,
                )
                method.dequantize_rows_into(packed, scales, globals_, slots, workspace)
                self.assertTrue((packed.cpu() == 171).all())
                self.assertTrue((scales.cpu() == 171).all())
                self.assertTrue((globals_.cpu() == -9).all())
                self.assertTrue((workspace.cpu() == 17).all())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--npu-device", type=int, default=0)
    args = parser.parse_args()
    if args.device == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.npu_device)
    TestNPUNVFP4Reference.device = args.device
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestNPUNVFP4Reference)
    )
    raise SystemExit(not result.wasSuccessful())
