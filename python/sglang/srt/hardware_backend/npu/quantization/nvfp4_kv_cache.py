# SPDX-License-Identifier: Apache-2.0
"""Software NVFP4 KV storage for Ascend devices without FP4 instructions."""

import torch
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    KVCacheAttentionAccess,
    KVCacheAttentionAccessKind,
    KVCacheAttentionPhase,
    KVCacheBackendMatcher,
    NVFP4KVCacheMethod,
)


class NPUNVFP4KVCacheMethod(NVFP4KVCacheMethod):
    """E2M1 pairs and E4M3 block scales stored as bytes, decoded on device.

    Each token's K/V tensor has an FP32 global scale and block-16 E4M3 scales.
    This keeps small activations in E4M3's normal range without calibration. No
    operation casts an NPU tensor to float4 or float8. The small lookup tables
    are built on CPU once, before serving; KV data always stays on the NPU.
    """

    def __init__(self, *, workspace_dtype, **kwargs):
        super().__init__(native_scale_layout=False, **kwargs)
        if workspace_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("NPU NVFP4 requires a BF16 or FP16 workspace.")
        self.workspace_dtype = workspace_dtype
        self._e4m3 = (
            torch.arange(127, dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .float()
            .to(self.device)
        )
        self._e2m1 = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float32,
            device=self.device,
        )
        self._scale_steps = torch.tensor(
            [2.0 ** (exponent - 3) for exponent in range(-6, 9)],
            dtype=torch.float32,
            device=self.device,
        )
        self._scale_inverse_steps = torch.tensor(
            [2.0 ** (3 - exponent) for exponent in range(-6, 9)],
            dtype=torch.float32,
            device=self.device,
        )

    def attention_accesses(self):
        return tuple(
            KVCacheAttentionAccess(
                phase=phase,
                kind=KVCacheAttentionAccessKind.DEQUANT_WORKSPACE,
                backend_matcher=KVCacheBackendMatcher(exact=frozenset({"ascend"})),
                storage_dtype=torch.uint8,
                attention_kv_dtype=self.workspace_dtype,
                scale_recipe="nvfp4",
                workspace_dtype=self.workspace_dtype,
            )
            for phase in KVCacheAttentionPhase
        )

    def scale_buffer_view_dtype(self):
        return None

    def load_scales_from_model(self, model_runner):
        # Online per-token scales are stored alongside the packed KV rows.
        pass

    def compute_cell_size(self, head_num, head_dim, num_layers, kv_size):
        return (
            super().compute_cell_size(head_num, head_dim, num_layers, kv_size)
            + 8 * num_layers
        )

    def create_buffers(self, size, head_num, head_dim, layer_num, device):
        if head_dim % self.SCALE_BLOCK_SIZE:
            raise ValueError("NPU NVFP4 requires head_dim divisible by 16.")
        return super().create_buffers(size, head_num, head_dim, layer_num, device)

    @staticmethod
    def _round_even(value):
        lower = torch.floor(value)
        fraction = value - lower
        increment = (fraction > 0.5) | ((fraction == 0.5) & (lower % 2 == 1))
        return lower + increment.to(value.dtype)

    def _encode_scale(self, scale):
        # Positive E4M3FN, round-to-nearest-even, including subnormals.
        scale = scale.clamp(0, 448)
        exponent = torch.floor(torch.log2(scale.clamp_min(2**-9))).clamp(-6, 8)
        step_index = (exponent + 6).long()
        step = self._scale_steps[step_index]
        rounded = self._round_even(scale * self._scale_inverse_steps[step_index]) * step
        exponent = torch.floor(torch.log2(rounded.clamp_min(2**-9))).clamp(-6, 8)
        mantissa = self._round_even(
            rounded * self._scale_inverse_steps[(exponent + 6).long()]
        )
        code = torch.where(
            rounded < 2**-6,
            self._round_even(rounded * 512),
            (exponent + 7) * 8 + mantissa - 8,
        )
        return code.to(torch.uint8)

    def _quantize(self, tensor, global_scale):
        shape = tensor.shape
        blocks = (tensor.float() / global_scale).reshape(*shape[:-1], -1, 16)
        scale_bytes = self._encode_scale(blocks.abs().amax(dim=-1) / 6)
        scales = self._e4m3[scale_bytes.long()]
        normalized = blocks / scales.clamp_min(2**-9).unsqueeze(-1)
        magnitude = normalized.abs()
        codes = torch.zeros_like(normalized, dtype=torch.int32)
        # Midpoints alternate between ties rounded down and up (even code).
        for index, midpoint in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
            above = magnitude >= midpoint if index % 2 else magnitude > midpoint
            codes = codes + above.to(torch.int32)
        codes = (codes + (normalized < 0).to(torch.int32) * 8).reshape(shape)
        packed = (codes[..., 0::2] + codes[..., 1::2] * 16).to(torch.uint8)
        return packed, scale_bytes

    def quantize_and_store(
        self,
        k_buffer,
        v_buffer,
        k_scale_buffer,
        v_scale_buffer,
        loc,
        cache_k,
        cache_v,
        k_scale=None,
        v_scale=None,
        native_k_scale_buffer=None,
        native_v_scale_buffer=None,
        *,
        k_global_scale_buffer,
        v_global_scale_buffer,
    ):
        loc = loc.long()
        for source, data, scales, global_scale, global_buffer in (
            (cache_k, k_buffer, k_scale_buffer, k_scale, k_global_scale_buffer),
            (cache_v, v_buffer, v_scale_buffer, v_scale, v_global_scale_buffer),
        ):
            source = source.reshape(loc.numel(), *data.shape[1:-1], data.shape[-1] * 2)
            if global_scale is None:
                global_scale = (
                    source.float().abs().amax(dim=(-2, -1), keepdim=True) / (6 * 448)
                ).clamp_min(2**-126)
            packed, block_scales = self._quantize(source, global_scale)
            data[loc] = packed
            scales[loc] = block_scales
            global_buffer[loc] = global_scale

    def _dequantize(self, packed, scales, global_scale):
        codes = packed.to(torch.int32)
        codes = torch.stack((codes % 16, codes // 16), dim=-1)
        values = self._e2m1[codes.long()].reshape(
            *packed.shape[:-1], -1, self.SCALE_BLOCK_SIZE
        )
        block_scales = self._e4m3[scales.long()]
        return (values * block_scales.unsqueeze(-1)).flatten(-2) * global_scale

    def dequantize_into(self, packed, scales, global_scale, workspace):
        # Bound FP32 scratch instead of materializing the entire cache in FP32.
        for start in range(0, packed.shape[0], 256):
            end = start + 256
            chunk_scale = (
                global_scale[start:end]
                if isinstance(global_scale, torch.Tensor) and global_scale.numel() > 1
                else global_scale
            )
            workspace[start:end].copy_(
                self._dequantize(packed[start:end], scales[start:end], chunk_scale)
            )
        return workspace

    def dequantize_prev_kv(self, k_fp4, k_scales, v_fp4, v_scales, layer_id):
        raise NotImplementedError(
            "NPU NVFP4 requires the pool's per-token FP32 global scale buffers."
        )
