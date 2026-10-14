# SPDX-License-Identifier: Apache-2.0
"""NVFP4 rounding and paging compared with an independent CPU oracle."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_store_scale_and_fp4_midpoints(self):
        table = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        midpoints = (table[:-1] + table[1:]) / 2
        samples = torch.cat((table, midpoints, midpoints * 0.999, midpoints * 1.001))
        source = torch.zeros(samples.numel() + 1, 1, 16)
        source[:-1, 0, -1] = samples * 6
        source[-1, 0] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6] * 2)
        expected = reference_quantize(source, 1.0)
        packed = torch.empty(
            source.shape[0], 1, 8, dtype=torch.uint8, device=self.device
        )
        scales = torch.empty(
            source.shape[0], 1, 1, dtype=torch.uint8, device=self.device
        )
        global_buffer = torch.empty(source.shape[0], 1, 1, device=self.device)
        self.make_method().quantize_and_store(
            packed,
            packed,
            scales,
            scales,
            torch.arange(source.shape[0], device=self.device),
            source.to(self.device),
            source.to(self.device),
            1.0,
            1.0,
            k_global_scale_buffer=global_buffer,
            v_global_scale_buffer=global_buffer,
        )
        torch.testing.assert_close(packed.cpu(), expected[0], rtol=0, atol=0)
        torch.testing.assert_close(scales.cpu(), expected[1], rtol=0, atol=0)

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

    def test_live_pages_and_unquantized_current_prefill(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                method = self.make_method(dtype)
                pool = NPUNVFP4TokenToKVPool(
                    size=512,
                    page_size=16,
                    dtype=torch.float4_e2m1fn_x2,
                    head_num=2,
                    head_dim=64,
                    layer_num=2,
                    device=self.device,
                    enable_memory_saver=False,
                    enable_alt_stream=False,
                    quant_method=method,
                )
                source = torch.randn(
                    32, 2, 64, generator=torch.Generator().manual_seed(89)
                ).to(dtype)
                loc = torch.arange(16, 48, device=self.device)
                for layer_id in (0, 1):
                    pool.set_kv_buffer(
                        SimpleNamespace(layer_id=layer_id),
                        loc,
                        source.to(self.device),
                        source.to(self.device),
                        k_scale=0.1,
                        v_scale=0.1,
                    )
                pool.dq_k_buffer.fill_(float("nan"))
                pool.dq_v_buffer.fill_(float("nan"))
                # Two requests share a prefix page; the second table's padded
                # page 20 must never be read or written.
                pool.prepare_forward(
                    torch.tensor([[1, 2], [1, 20]], device=self.device),
                    torch.tensor([32, 16], device=self.device),
                    preserve_current=True,
                )
                fresh = source[16:].to(self.device)
                pool.set_kv_buffer(
                    SimpleNamespace(layer_id=0),
                    loc[16:],
                    fresh,
                    fresh,
                    k_scale=0.1,
                    v_scale=0.1,
                )
                key = pool.get_key_buffer(0).reshape(-1, 2, 64)
                value = pool.get_value_buffer(0).reshape(-1, 2, 64)
                expected_prefix = reference_quantize(source[:16], 0.1)[2].to(dtype)
                torch.testing.assert_close(
                    key[16:32].cpu(), expected_prefix, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    key[32:48].cpu(), source[16:], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    value[32:48].cpu(), source[16:], rtol=0, atol=0
                )
                self.assertTrue(torch.isnan(key[320:336]).all())
                # Decode uses persistent NVFP4 rows, with no stale prefill override.
                pool.prepare_forward(
                    torch.tensor([[1, 2]], device=self.device),
                    torch.tensor([32], device=self.device),
                    preserve_current=False,
                )
                decoded = pool.get_key_buffer(0).reshape(-1, 2, 64)[16:48].cpu()
                torch.testing.assert_close(
                    decoded,
                    reference_quantize(source, 0.1)[2].to(dtype),
                    rtol=0,
                    atol=0,
                )

    def test_exact_prefix_rows_and_fia_views(self):
        """Shared and partial prefix pages must not decode fresh or padded rows."""
        for dtype in (torch.bfloat16, torch.float16):
            for use_fia in (False, True):
                with self.subTest(dtype=dtype, use_fia=use_fia):
                    method = self.make_method(dtype)
                    with patch.dict(os.environ, ASCEND_USE_FIA=str(int(use_fia))):
                        pool = NPUNVFP4TokenToKVPool(
                            size=512,
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
                    # Requests 1 and 3 share a full prefix page. Request 2's
                    # partial prefix uses another page so its fresh suffix
                    # cannot overwrite the other request's cached tokens.
                    prefix_slots = torch.tensor(
                        list(range(16, 32)) + [48, 49, 50, 80, 81, 82],
                        device=self.device,
                    )
                    fresh_slots = torch.tensor(
                        [51, 52, 53, 54, 83, 84, 112, 113], device=self.device
                    )
                    source = torch.randn(
                        30, 2, 64, generator=torch.Generator().manual_seed(1234)
                    ).to(dtype)
                    value = (source.float() * 0.35 + 0.015).to(dtype)
                    pool.set_kv_buffer(
                        SimpleNamespace(layer_id=0),
                        prefix_slots,
                        source[:22].to(self.device),
                        value[:22].to(self.device),
                        k_scale=0.1,
                        v_scale=0.2,
                    )
                    pool.dq_k_buffer.fill_(float("nan"))
                    pool.dq_v_buffer.fill_(float("nan"))
                    tables = torch.tensor(
                        [[1, 3, 20], [5, 21, 22], [1, 7, 23]], device=self.device
                    )
                    lengths = torch.tensor([23, 5, 18], device=self.device)
                    pool.prepare_forward(
                        tables,
                        lengths,
                        preserve_current=True,
                        prefix_lens=torch.tensor([19, 3, 16], device=self.device),
                        prefix_lens_cpu=[19, 3, 16],
                    )
                    self.assertEqual(
                        sorted(pool._active_slots.cpu().tolist()),
                        prefix_slots.cpu().tolist(),
                    )
                    pool.set_kv_buffer(
                        SimpleNamespace(layer_id=0),
                        fresh_slots,
                        source[22:].to(self.device),
                        value[22:].to(self.device),
                        k_scale=0.1,
                        v_scale=0.2,
                    )
                    key_view = pool.get_key_buffer(0)
                    value_view = pool.get_value_buffer(0)
                    expected_shape = (528, 1, 2, 64) if use_fia else (33, 16, 2, 64)
                    self.assertEqual(tuple(key_view.shape), expected_shape)
                    self.assertEqual(tuple(value_view.shape), expected_shape)
                    self.assertEqual(key_view.data_ptr(), pool.dq_k_buffer.data_ptr())
                    self.assertEqual(value_view.data_ptr(), pool.dq_v_buffer.data_ptr())
                    key = key_view.reshape(-1, 2, 64)
                    val = value_view.reshape(-1, 2, 64)
                    for workspace, original, scale in (
                        (key, source, 0.1),
                        (val, value, 0.2),
                    ):
                        torch.testing.assert_close(
                            workspace[prefix_slots].cpu(),
                            reference_quantize(original[:22], scale)[2].to(dtype),
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(
                            workspace[fresh_slots].cpu(),
                            original[22:],
                            rtol=0,
                            atol=0,
                        )
                        # Tail rows of each live page, a padding page and an
                        # unrelated page must stay untouched during prefill.
                        for start, end in ((55, 64), (85, 96), (114, 128), (320, 384)):
                            self.assertTrue(torch.isnan(workspace[start:end]).all())
                    # FIA receives a [pages, page_size, H*D] view of exactly
                    # this storage; both public layouts must flatten identically.
                    torch.testing.assert_close(
                        key_view.view(-1, 16, 128)[3, 3:7].cpu(),
                        source[22:26].reshape(4, 128),
                        rtol=0,
                        atol=0,
                    )
                    pool.prepare_forward(tables, lengths, preserve_current=False)
                    decoded_k = pool.get_key_buffer(0).reshape(-1, 2, 64)
                    decoded_v = pool.get_value_buffer(0).reshape(-1, 2, 64)
                    for workspace, original, scale in (
                        (decoded_k, source, 0.1),
                        (decoded_v, value, 0.2),
                    ):
                        expected = reference_quantize(original[22:], scale)[2].to(dtype)
                        torch.testing.assert_close(
                            workspace[fresh_slots].cpu(), expected, rtol=0, atol=0
                        )
                        self.assertTrue((expected != original[22:]).any())

    def test_zero_prefix_skips_dequant_and_preserves_fresh_rows(self):
        for dtype in (torch.bfloat16, torch.float16):
            for use_fia in (False, True):
                with self.subTest(dtype=dtype, use_fia=use_fia):
                    method = self.make_method(dtype)
                    with patch.dict(os.environ, ASCEND_USE_FIA=str(int(use_fia))):
                        pool = NPUNVFP4TokenToKVPool(
                            size=512,
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
                    pool.dq_k_buffer.fill_(float("nan"))
                    pool.dq_v_buffer.fill_(float("nan"))
                    pool.prepare_forward(
                        torch.tensor([[8, 20], [6, 21]], device=self.device),
                        torch.tensor([3, 2], device=self.device),
                        preserve_current=True,
                        prefix_lens=torch.zeros(
                            2, dtype=torch.int32, device=self.device
                        ),
                        prefix_lens_cpu=[0, 0],
                    )
                    self.assertEqual(pool._active_slots.numel(), 0)
                    slots = torch.tensor([128, 129, 130, 96, 97], device=self.device)
                    source = torch.randn(
                        5, 2, 64, generator=torch.Generator().manual_seed(4321)
                    ).to(dtype)
                    value = (-source.float() * 0.4).to(dtype)
                    pool.set_kv_buffer(
                        SimpleNamespace(layer_id=0),
                        slots,
                        source.to(self.device),
                        value.to(self.device),
                    )
                    for workspace, expected in (
                        (pool.get_key_buffer(0).reshape(-1, 2, 64), source),
                        (pool.get_value_buffer(0).reshape(-1, 2, 64), value),
                    ):
                        torch.testing.assert_close(
                            workspace[slots].cpu(), expected, rtol=0, atol=0
                        )
                        self.assertEqual(
                            int(torch.isfinite(workspace).all(-1).all(-1).sum()), 5
                        )


if __name__ == "__main__":
    unittest.main()
