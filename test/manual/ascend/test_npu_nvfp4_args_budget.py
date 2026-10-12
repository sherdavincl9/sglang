# SPDX-License-Identifier: Apache-2.0
"""Manual regressions for NPU NVFP4 argument resolution and KV budgeting."""

import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.kv_cache_hook import handle_nvfp4_prefill_kv_dequant_dtype
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.runtime_context import get_context, get_server_args, override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

# Reuse the existing CPU budget fixture; this manual file has no CI registration.
_fixture = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "registered/unit/model_executor/test_pool_configurator.py"
    )
)
_make_model_runner = _fixture["_make_model_runner"]
mock_cpu_env = _fixture["mock_cpu_env"]


class TestNPUNVFP4Args(CustomTestCase):
    def setUp(self):
        patcher = patch(
            "sglang.srt.arg_groups.kv_cache_hook.use_mla_backend", return_value=False
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _make_unrouted_nvfp4_args(**overrides):
        return ServerArgs(model_path="dummy", kv_cache_dtype="nvfp4", **overrides)

    @override_platform(is_npu=True, is_cuda=False)
    def test_npu_nvfp4_workspace_resolution_preserves_raw_input(self):
        import torch
        from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
        from transformers import Qwen3Config

        for dtype, requested, expected in (
            (torch.bfloat16, "auto", "bf16"),
            (torch.bfloat16, "bf16", "bf16"),
            (torch.float16, "auto", "fp16"),
            (torch.float16, "fp16", "fp16"),
        ):
            with self.subTest(dtype=dtype, requested=requested):
                model_config = SimpleNamespace(
                    hf_config=Qwen3Config(),
                    dtype=dtype,
                    is_hybrid_swa=False,
                    linear_attn_registry_result=None,
                )
                args = self._make_unrouted_nvfp4_args(
                    prefill_kv_cache_dequant_dtype=requested,
                    cuda_graph_config=CudaGraphConfig(),
                )
                # The model metadata boundary replaces filesystem/HF loading.
                with patch(
                    "sglang.srt.arg_groups.kv_cache_hook.model_config_of",
                    return_value=model_config,
                ):
                    handle_nvfp4_prefill_kv_dequant_dtype(args)
                self.assertEqual(args.prefill_kv_cache_dequant_dtype, requested)
                self.assertEqual(
                    resolution_result(args, "prefill_kv_cache_dequant_dtype"), expected
                )
                self.assertEqual(
                    resolution_result(args, "prefill_attention_backend"), "ascend"
                )
                self.assertEqual(
                    resolution_result(args, "decode_attention_backend"), "ascend"
                )
                self.assertTrue(resolution_result(args, "disable_overlap_schedule"))
                graph = resolution_result(args, "cuda_graph_config")
                self.assertEqual(graph.prefill.backend, "disabled")
                self.assertEqual(graph.decode.backend, "disabled")

    @override_platform(is_npu=True, is_cuda=False)
    def test_npu_nvfp4_rejects_incompatible_workspaces_and_storage(self):
        import torch
        from transformers import Qwen3Config

        model_config = SimpleNamespace(
            hf_config=Qwen3Config(),
            dtype=torch.bfloat16,
            is_hybrid_swa=False,
            linear_attn_registry_result=None,
        )
        for workspace, message in (
            ("nvfp4", "native NVFP4"),
            ("fp8_e4m3", "native NVFP4"),
            ("fp16", "must match"),
        ):
            args = self._make_unrouted_nvfp4_args(
                prefill_kv_cache_dequant_dtype=workspace
            )
            with patch(
                "sglang.srt.arg_groups.kv_cache_hook.model_config_of",
                return_value=model_config,
            ):
                with self.assertRaisesRegex(ValueError, message):
                    handle_nvfp4_prefill_kv_dequant_dtype(args)
        args = ServerArgs(model_path="dummy", prefill_kv_cache_dequant_dtype="bf16")
        with self.assertRaisesRegex(ValueError, "applies only"):
            handle_nvfp4_prefill_kv_dequant_dtype(args)

    @override_platform(is_npu=False, is_cuda=True, is_sm100=True)
    def test_cuda_nvfp4_rejects_npu_workspace_dtypes(self):
        for dtype in ("bf16", "fp16"):
            args = self._make_unrouted_nvfp4_args(prefill_kv_cache_dequant_dtype=dtype)
            with self.assertRaisesRegex(ValueError, "only on NPU"):
                handle_nvfp4_prefill_kv_dequant_dtype(args)


class TestNPUNVFP4Budget(CustomTestCase):
    def test_npu_nvfp4_prices_global_scales_and_shared_workspace(self):
        """CUDA-only dtype detection must not bypass NPU MHA recipe pricing."""
        import torch
        from sglang.srt.hardware_backend.npu.quantization.nvfp4_kv_cache import (
            NPUNVFP4KVCacheMethod,
        )
        from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator

        mr = _make_model_runner(
            self, num_kv_heads=2, head_dim=64, num_layers=3, page_size=16
        )
        mr.kv_cache_dtype = torch.float4_e2m1fn_x2
        mr.kv_cache_dtype_str = "nvfp4"
        method = NPUNVFP4KVCacheMethod(
            workspace_dtype=torch.bfloat16, num_layers=3, device="cpu", page_size=16
        )
        with (
            mock_cpu_env(kv_size=1),
            get_context().override_server_args(
                kv_cache_dtype="nvfp4",
                prefill_kv_cache_dequant_dtype="bf16",
                attention_backend="ascend",
            ),
            patch("sglang.srt.model_executor.pool_configurator._is_npu", True),
            patch(
                "sglang.srt.model_executor.pool_configurator.is_float4_e2m1fn_x2",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.quantization.fp4_kv_cache_quant_method.get_kv_cache_quant_method",
                return_value=method,
            ),
        ):
            mr.server_args = get_server_args()
            cfg = DefaultPoolConfigurator(mr)
        # Packed data + E4M3 blocks + FP32 token scales + one BF16 workspace.
        self.assertEqual(cfg._cell_size, 2 * 3 * 2 * (32 + 4) + 8 * 3 + 4 * 2 * 64)


if __name__ == "__main__":
    unittest.main()
