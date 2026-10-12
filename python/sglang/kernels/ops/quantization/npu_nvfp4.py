# SPDX-License-Identifier: Apache-2.0
"""Ascend software E2M1 packing and live-row dequantization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _round_even(value):
    lower = tl.floor(value)
    fraction = value - lower
    return lower + (
        (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
    )


@triton.jit
def _encode_e4m3(value):
    value = tl.minimum(tl.maximum(value, 0.0), 448.0)
    exponent = ((value.to(tl.int32, bitcast=True) >> 23) & 255) - 127
    exponent = tl.minimum(tl.maximum(exponent, -6), 8)
    inverse_step = ((130 - exponent) << 23).to(tl.float32, bitcast=True)
    mantissa = _round_even(value * inverse_step)
    return tl.where(
        value < 0.015625, _round_even(value * 512.0), (exponent + 7) * 8 + mantissa - 8
    ).to(tl.int32)


@triton.jit
def _quantize_store_kernel(
    SOURCE,
    LOC,
    DATA,
    SCALES,
    GLOBALS,
    E4M3,
    ROW_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    ONLINE: tl.constexpr,
    GLOBAL: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.arange(0, B)
    value = tl.load(
        SOURCE + token * ROW_STRIDE + head * HEAD_STRIDE + offsets,
        offsets < D,
        0,
    ).to(tl.float32)
    if ONLINE:
        global_scale = tl.load(GLOBALS + tl.load(LOC + token).to(tl.int32))
    else:
        global_scale = GLOBAL
    normalized = tl.div_rn(value, global_scale)
    blocks = tl.reshape(normalized, (B // 16, 16))
    scale_code = _encode_e4m3(tl.max(tl.abs(blocks), 1) / 6.0)
    scale = tl.load(E4M3 + scale_code)
    scaled = tl.div_rn(blocks, tl.maximum(scale[:, None], 0.001953125))
    magnitude = tl.abs(scaled)
    code = (magnitude > 0.25).to(tl.int32)
    code += (magnitude >= 0.75).to(tl.int32)
    code += (magnitude > 1.25).to(tl.int32)
    code += (magnitude >= 1.75).to(tl.int32)
    code += (magnitude > 2.5).to(tl.int32)
    code += (magnitude >= 3.5).to(tl.int32)
    code += (magnitude > 5.0).to(tl.int32)
    code += (scaled < 0).to(tl.int32) * 8
    # Ascend's gather supports floating values; all FP4 codes are exact FP32.
    code = tl.reshape(code, (B,)).to(tl.float32)
    even = tl.arange(0, B // 2) * 2
    packed = (tl.gather(code, even, 0) + tl.gather(code, even + 1, 0) * 16).to(tl.int32)
    slot = tl.load(LOC + token).to(tl.int32)
    byte_offsets = tl.arange(0, B // 2)
    scale_offsets = tl.arange(0, B // 16)
    tl.store(
        DATA + slot * (N // 2) + head * (D // 2) + byte_offsets,
        packed,
        byte_offsets < D // 2,
    )
    tl.store(
        SCALES + slot * (N // 16) + head * (D // 16) + scale_offsets,
        scale_code,
        scale_offsets < D // 16,
    )
    if not ONLINE:
        if head == 0:
            tl.store(GLOBALS + slot, global_scale)


@triton.jit
def _dequantize_rows_kernel(
    DATA,
    SCALES,
    GLOBALS,
    SLOTS,
    WORKSPACE,
    E4M3,
    E2M1,
    COUNT: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    slots = tl.load(SLOTS + rows, rows < COUNT, 0).to(tl.int32)
    head = tl.program_id(1)
    offsets = tl.arange(0, B)
    valid = (rows[:, None] < COUNT) & (offsets[None, :] < D)
    packed = tl.load(
        DATA + slots[:, None] * (N // 2) + head * (D // 2) + offsets[None, :] // 2,
        valid,
        0,
    ).to(tl.int32)
    code = (packed >> ((offsets[None, :] & 1) * 4)) & 15
    value = tl.load(E2M1 + code)
    scale_code = tl.load(
        SCALES + slots[:, None] * (N // 16) + head * (D // 16) + offsets[None, :] // 16,
        valid,
        0,
    ).to(tl.int32)
    scale = tl.load(E4M3 + scale_code)
    global_scale = tl.load(GLOBALS + slots, rows < COUNT, 0)
    decoded = (value * scale) * global_scale[:, None]
    tl.store(
        WORKSPACE + slots[:, None] * N + head * D + offsets[None, :], decoded, valid
    )


def quantize_store(source, loc, data, scales, globals_, e4m3, global_scale=None):
    if source.device.type != "npu" or source.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        raise ValueError("Ascend NVFP4 quantization requires NPU BF16/FP16/FP32 input.")
    if data.ndim != 3 or data.dtype != torch.uint8 or not data.is_contiguous():
        raise ValueError("NVFP4 data must be contiguous uint8 [slots, heads, dim/2].")
    if data.shape[-1] % 8 or not scales.is_contiguous() or scales.dtype != torch.uint8:
        raise ValueError(
            "NVFP4 requires block-16 head dimensions and contiguous uint8 scales."
        )
    if scales.shape != (data.shape[0], data.shape[1], data.shape[2] // 8):
        raise ValueError("NVFP4 block-scale shape does not match packed data.")
    if globals_.dtype != torch.float32 or globals_.shape != (data.shape[0], 1, 1):
        raise ValueError("NVFP4 global scales must be FP32 [slots, 1, 1].")
    n = data.shape[1] * data.shape[2] * 2
    source = source.reshape(loc.numel(), data.shape[1], data.shape[2] * 2)
    if source.stride(-1) != 1:
        source = source.contiguous()
    loc = loc.contiguous()
    if loc.numel():
        if global_scale is None:
            # Match PyTorch's FP32 global-scale rounding before FP4 tie tests.
            # Quantization and scattering of block data remain fused below.
            globals_[loc] = (
                source.float().abs().amax((-2, -1), keepdim=True) / 2688.0
            ).clamp_min(2**-126)
        _quantize_store_kernel[(loc.numel(), data.shape[1])](
            source,
            loc,
            data,
            scales,
            globals_,
            e4m3,
            source.stride(0),
            source.stride(1),
            source.shape[2],
            n,
            triton.next_power_of_2(source.shape[2]),
            global_scale is None,
            1.0 if global_scale is None else float(global_scale),
            enable_fp_fusion=False,
        )


def dequantize_rows(data, scales, globals_, slots, workspace, e4m3, e2m1):
    if workspace.device.type != "npu" or workspace.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        raise ValueError(
            "Ascend NVFP4 dequantization requires an NPU BF16/FP16 workspace."
        )
    if not workspace.is_contiguous() or workspace.shape != (
        *data.shape[:-1],
        data.shape[-1] * 2,
    ):
        raise ValueError("NVFP4 workspace must be contiguous and match packed data.")
    n = workspace.shape[1] * workspace.shape[2]
    if slots.numel():
        _dequantize_rows_kernel[(triton.cdiv(slots.numel(), 4), workspace.shape[1])](
            data,
            scales,
            globals_,
            slots,
            workspace,
            e4m3,
            e2m1,
            slots.numel(),
            n,
            workspace.shape[2],
            triton.next_power_of_2(workspace.shape[2]),
            4,
            enable_fp_fusion=False,
        )
