# SPDX-License-Identifier: Apache-2.0
"""NVFP4 rounding and paging compared with an independent CPU oracle."""

import unittest
from types import SimpleNamespace

import torch
from sglang.srt.hardware_backend.npu.nvfp4_memory_pool import NPUNVFP4TokenToKVPool
from sglang.srt.hardware_backend.npu.quantization.nvfp4_kv_cache import (
    NPUNVFP4KVCacheMethod,
)
from sglang.test.test_utils import CustomTestCase


def reference_quantize(source, global_scale):
    source = source.cpu().float()
    blocks = (source / global_scale).reshape(*source.shape[:-1], -1, 16)
    scales = (blocks.abs().amax(-1) / 6).clamp(max=448).to(torch.float8_e4m3fn)
    normalized = blocks / scales.float().clamp_min(2**-9).unsqueeze(-1)
    distances = (
        normalized.abs().unsqueeze(-1) - torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    ).abs()
    ties = distances == distances.amin(-1, keepdim=True)
    codes = torch.where(ties, torch.tensor([0, 9, 2, 11, 4, 13, 6, 15]), 99).argmin(-1)
    codes = (codes + (normalized < 0) * 8).reshape(source.shape)
    packed = (codes[..., ::2] + 16 * codes[..., 1::2]).to(torch.uint8)
    signed = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    decoded = (
        signed[codes].reshape_as(blocks) * scales.float().unsqueeze(-1)
    ).reshape_as(source) * global_scale
    return packed, scales.view(torch.uint8), decoded


class TestNPUNVFP4KVCache(CustomTestCase):
    device = "cpu"

    def make_method(self, dtype=torch.bfloat16):
        return NPUNVFP4KVCacheMethod(
            workspace_dtype=dtype, num_layers=4, device=self.device, page_size=16
        )

    def test_e4m3_midpoint_rounding(self):
        table = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        midpoints = (table[:-1] + table[1:]) / 2
        samples = torch.cat((table, midpoints, midpoints * 0.999, midpoints * 1.001))
        expected = samples.clamp(max=448).to(torch.float8_e4m3fn).view(torch.uint8)
        torch.testing.assert_close(
            self.make_method()._encode_scale(samples.to(self.device)).cpu(),
            expected,
            rtol=0,
            atol=0,
        )

    def test_bytes_and_dequant_match_reference(self):
        source = torch.randn(19, 3, 64, generator=torch.Generator().manual_seed(42))
        source[0] = 0
        source[1, 0, :16] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6] * 2)
        for dtype in (torch.bfloat16, torch.float16):
            for scale in (1.0, 0.1):
                with self.subTest(dtype=dtype, scale=scale):
                    method = self.make_method(dtype)
                    expected_packed, expected_scales, expected_values = (
                        reference_quantize(source, scale)
                    )
                    packed, scales = method._quantize(source.to(self.device), scale)
                    torch.testing.assert_close(
                        packed.cpu(), expected_packed, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        scales.cpu(), expected_scales, rtol=0, atol=0
                    )
                    decoded = method._dequantize(packed, scales, scale).to(dtype).cpu()
                    torch.testing.assert_close(
                        decoded, expected_values.to(dtype), rtol=0, atol=0
                    )

    def test_paged_scatter_move_and_workspace_memory(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                method = self.make_method(dtype)
                method.k_scales_gpu.fill_(0.1)
                method.v_scales_gpu.fill_(0.2)
                method.k_scales_float = [0.1] * 4
                method.v_scales_float = [0.2] * 4
                pool = NPUNVFP4TokenToKVPool(
                    size=512,
                    page_size=16,
                    dtype=torch.float4_e2m1fn_x2,
                    head_num=2,
                    head_dim=64,
                    layer_num=2,
                    start_layer=2,
                    end_layer=3,
                    device=self.device,
                    enable_memory_saver=False,
                    enable_alt_stream=False,
                    quant_method=method,
                )
                source = torch.randn(
                    3, 2, 64, generator=torch.Generator().manual_seed(123)
                )
                loc = torch.tensor([1, 17, 300], device=self.device)
                for layer_id in (2, 3):
                    pool.set_kv_buffer(
                        SimpleNamespace(layer_id=layer_id),
                        loc,
                        source.to(self.device) * layer_id,
                        source.to(self.device),
                        k_scale=0.1,
                        v_scale=0.2,
                    )
                workspace_ptr = pool.get_key_buffer(2).data_ptr()
                for layer_id in (3, 2):
                    key = pool.get_key_buffer(layer_id)
                    self.assertEqual(key.data_ptr(), workspace_ptr)
                    expected_k = reference_quantize(source * layer_id, 0.1)[2].to(dtype)
                    expected_v = reference_quantize(source, 0.2)[2].to(dtype)
                    torch.testing.assert_close(
                        key.reshape(-1, 2, 64)[loc].cpu(), expected_k, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        pool.get_value_buffer(layer_id).reshape(-1, 2, 64)[loc].cpu(),
                        expected_v,
                        rtol=0,
                        atol=0,
                    )
                pool.move_kv_cache(torch.tensor([32, 33, 34], device=self.device), loc)
                key = pool.get_key_buffer(2).reshape(-1, 2, 64)
                torch.testing.assert_close(key[32:35], key[loc], rtol=0, atol=0)
                self.assertEqual(
                    sum(pool.get_kv_size_bytes()),
                    method.compute_cell_size(2, 64, 2, 1) * 528,
                )
                self.assertEqual(pool.k_buffer[0].dtype, torch.uint8)

    def test_online_global_scale_preserves_small_activations(self):
        method = self.make_method()
        pool = NPUNVFP4TokenToKVPool(
            size=32,
            page_size=16,
            dtype=torch.float4_e2m1fn_x2,
            head_num=2,
            head_dim=64,
            layer_num=1,
            device=self.device,
            enable_memory_saver=False,
            enable_alt_stream=False,
            quant_method=method,
        )
        source = (
            torch.randn(3, 2, 64, generator=torch.Generator().manual_seed(7)) * 1e-3
        )
        source[0] = 0
        global_scale = (
            source.abs().amax((-2, -1), keepdim=True) / (6 * 448)
        ).clamp_min(2**-126)
        expected = reference_quantize(source, global_scale)
        loc = torch.tensor([1, 16, 33], device=self.device)
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=0),
            loc,
            source.to(self.device),
            source.to(self.device),
        )
        torch.testing.assert_close(
            pool.k_buffer[0][loc].cpu(), expected[0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            pool.k_scale_buffer[0][loc].cpu(), expected[1], rtol=0, atol=0
        )
        actual = pool.get_key_buffer(0).reshape(-1, 2, 64)[loc].cpu()
        torch.testing.assert_close(actual, expected[2].bfloat16(), rtol=0.01, atol=1e-7)
        self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
