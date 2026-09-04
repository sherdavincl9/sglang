"""A/B accuracy test for the fused Ascend KDA extend implementation.

The reference below intentionally preserves the decomposed extend path that
preceded ``cann_ops_transformer.ops.chunk_kda_fwd``.  Both paths receive clones
of the same inputs and persistent state pool so in-place writes cannot leak
from one run into the other.
"""

import math
import os
import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=180, suite="base-a-test-1-npu-a2")

pytest.importorskip("torch_npu", reason="Ascend KDA equivalence requires torch_npu")

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend KDA equivalence requires an NPU"
)

from sgl_kernel_npu.fla.kda_chunk_delta_h import (  # noqa: E402
    chunk_gated_delta_rule_fwd_h_npu,
)
from sgl_kernel_npu.fla.kda_prefill import (  # noqa: E402
    chunk_gla_fwd_o_gk_npu,
    recompute_w_u_fwd_npu,
)
from sgl_kernel_npu.fla.solve_tril import solve_tril_npu  # noqa: E402
from sgl_kernel_npu.fla.utils import prepare_chunk_indices  # noqa: E402

from sglang.kernels.ops.attention.fla.cumsum import (  # noqa: E402
    chunk_local_cumsum,
)
from sglang.kernels.ops.attention.fla.kda import (  # noqa: E402
    chunk_kda_scaled_dot_kkt_fwd,
)
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd  # noqa: E402
from sglang.srt.hardware_backend.npu.attention.ascend_kda_backend import (  # noqa: E402
    _AscendKDAExtendKernel,
)


CHUNK_SIZE = 64
KEY_DIM = 128
NUM_HEADS = 2
POOL_SIZE = 8
BF16_RTOL = 1e-2
BF16_ATOL = 5e-5
FP32_GK_RTOL = 1e-6
FP32_GK_ATOL = 1e-5
FP32_STATE_RTOL = 1e-3
FP32_STATE_ATOL = 1e-4
FP32_RECON_RTOL = 1e-3
FP32_RECON_ATOL = 1e-5
BF16_POINTWISE_ULPS = 1
BF16_REDUCTION_ULPS = 2

# Keep the historical names as the default numerical comparison profile.
# BF16 tensors that expose a stable public contract use ULP checks below.
RTOL = BF16_RTOL
ATOL = BF16_ATOL
_LOG2_E = math.log2(math.e)


def _legacy_extend(
    q,
    k,
    v,
    g,
    beta,
    *,
    ssm_states,
    cache_indices,
    query_start_loc,
    return_intermediate_states,
    debug=None,
    normalize_qk=True,
):
    """The exact decomposed implementation used before chunk_kda_fwd."""
    if normalize_qk:
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())
    else:
        q = q.contiguous()
        k = k.contiguous()
    v = v.contiguous()
    beta = beta.contiguous()
    chunk_indices = prepare_chunk_indices(query_start_loc, CHUNK_SIZE)
    g = chunk_local_cumsum(
        g.contiguous(),
        chunk_size=CHUNK_SIZE,
        scale=_LOG2_E,
        cu_seqlens=query_start_loc,
        chunk_indices=chunk_indices,
    )
    if debug is not None:
        debug["gk"] = g
        # Public qg is the unscaled gated query. The attention kernel applies
        # the QK scale separately when consuming it.
        debug["qg"] = (q.float() * torch.exp2(g.float())).to(q.dtype)

    triangular, query_key = chunk_kda_scaled_dot_kkt_fwd(
        q=q,
        k=k,
        gk=g,
        beta=beta,
        scale=k.shape[-1] ** -0.5,
        cu_seqlens=query_start_loc,
        output_dtype=torch.float32,
    )
    triangular = solve_tril_npu(
        A=triangular,
        cu_seqlens=query_start_loc,
        output_dtype=k.dtype,
    )
    if debug is not None:
        debug["aqk"] = query_key
        debug["akk"] = triangular
    w, u, gated_k = recompute_w_u_fwd_npu(
        k=k,
        v=v,
        beta=beta,
        A=triangular,
        gk=g,
        cu_seqlens=query_start_loc,
        chunk_indices=chunk_indices,
    )
    if debug is not None:
        debug["w"] = w
        debug["u"] = u
        debug["kg"] = gated_k
    chunk_states, new_values = chunk_gated_delta_rule_fwd_h_npu(
        k=gated_k,
        w=w,
        u=u,
        gk=g,
        initial_state=ssm_states,
        initial_state_indices=cache_indices,
        cu_seqlens=query_start_loc,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )
    if debug is not None:
        debug["h"] = chunk_states.transpose(-1, -2).contiguous()
        debug["v_new"] = new_values
    out = chunk_gla_fwd_o_gk_npu(
        q=q,
        v=new_values,
        g=g,
        A=query_key,
        h=chunk_states,
        out=v,
        scale=k.shape[-1] ** -0.5,
        cu_seqlens=query_start_loc,
        chunk_size=CHUNK_SIZE,
        chunk_indices=chunk_indices,
    )
    if return_intermediate_states:
        return out, chunk_states.transpose(-1, -2).contiguous()
    return out


def _make_inputs(seq_lens, value_dim):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260828 + sum(seq_lens))
    total_tokens = sum(seq_lens)
    qk_shape = (1, total_tokens, NUM_HEADS, KEY_DIM)
    v_shape = (1, total_tokens, NUM_HEADS, value_dim)

    def bf16_randn(shape, scale=0.05):
        return (torch.randn(shape, generator=generator) * scale).to(
            device="npu", dtype=torch.bfloat16
        )

    q = bf16_randn(qk_shape)
    k = bf16_randn(qk_shape)
    v = bf16_randn(v_shape)
    # This test starts at _AscendKDAExtendKernel.extend, not at the model
    # projection. At this boundary raw BF16 forget_gate has already been
    # activated by AscendKDAAttnBackend._prepare_extend_gate_inputs, whose
    # fused_kda_gate_npu output is FP32.
    g = (torch.randn(qk_shape, generator=generator) * 0.02 - 0.5).to(
        device="npu", dtype=torch.float32
    )
    # Kimi extend applies beta.float().sigmoid() before entering the backend.
    beta = torch.sigmoid(
        torch.randn(qk_shape[:-1], generator=generator)
    ).to(device="npu", dtype=torch.float32)
    initial_pool = (
        torch.randn(
            (POOL_SIZE, NUM_HEADS, value_dim, KEY_DIM), generator=generator
        )
        * 0.01
    ).to(device="npu", dtype=torch.float32)

    cache_indices = torch.tensor(
        (5, 1, 6)[: len(seq_lens)], dtype=torch.int64, device="npu"
    )
    cumulative = [0]
    for seq_len in seq_lens:
        cumulative.append(cumulative[-1] + seq_len)
    query_start_loc = torch.tensor(cumulative, dtype=torch.int64, device="npu")
    return (q, k, v, g, beta), initial_pool, cache_indices, query_start_loc


def _clone_tensors(tensors):
    return tuple(tensor.clone() for tensor in tensors)


def _run_fused_operator_debug(
    q,
    k,
    v,
    g,
    beta,
    *,
    ssm_states,
    cache_indices,
    query_start_loc,
    output_h,
    output_gk=False,
    output_w=False,
    output_u=False,
    output_qg=False,
    output_kg=False,
    output_v_new=False,
    scale=None,
    normalize_qk=True,
):
    """Run the same custom op as the new adapter, but retain all public outputs."""
    if normalize_qk:
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())
    else:
        q = q.contiguous()
        k = k.contiguous()
    num_sequences = query_start_loc.shape[0] - 1
    source_indices = cache_indices[:num_sequences].to(torch.long)
    initial_state = (
        ssm_states.index_select(0, source_indices)
        .to(dtype=torch.float32)
        .contiguous()
    )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return torch.ops.npu.chunk_kda_fwd(
        q,
        k,
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        output_gk=output_gk,
        output_w=output_w,
        output_u=output_u,
        output_qg=output_qg,
        output_kg=output_kg,
        output_v_new=output_v_new,
        cu_seqlens=query_start_loc,
        chunk_size=CHUNK_SIZE,
        layout="BSND",
        safe_gate=False,
        use_gate_in_kernel=False,
        state_v_first=True,
        output_h=output_h,
    )


def _aqk_to_token_major(aqk):
    """Convert migrated BSND Aqk from [B, H, T, C] to [B, T, H, C]."""
    assert aqk.dim() == 4
    return aqk.permute(0, 2, 1, 3).contiguous()


def _check_aqk_scale_response(aqk_scale_1, aqk_scale_half, seq_lens):
    """Check Aqk(0.5) == Aqk(1.0) * 0.5 on valid causal entries."""
    aqk_scale_1 = _aqk_to_token_major(aqk_scale_1).float()
    aqk_scale_half = _aqk_to_token_major(aqk_scale_half).float()
    expected = aqk_scale_1 * 0.5
    failures = []
    seq_start = 0

    for seq_idx, seq_len in enumerate(seq_lens):
        local_start = 0
        chunk_idx = 0
        while local_start < seq_len:
            chunk_len = min(CHUNK_SIZE, seq_len - local_start)
            global_start = seq_start + local_start
            global_end = global_start + chunk_len

            # Aqk is causal inside each logical chunk. Only columns 0..row are
            # meaningful; padded columns and the upper triangle are excluded.
            actual_chunk = aqk_scale_half[
                :, global_start:global_end, :, :chunk_len
            ]
            expected_chunk = expected[
                :, global_start:global_end, :, :chunk_len
            ]
            causal_mask = torch.ones(
                (chunk_len, chunk_len), dtype=torch.bool
            ).tril_().to(actual_chunk.device)
            causal_mask = causal_mask.view(1, chunk_len, 1, chunk_len).expand_as(
                actual_chunk
            )
            actual_valid = actual_chunk[causal_mask]
            expected_valid = expected_chunk[causal_mask]

            abs_diff = (actual_valid - expected_valid).abs()
            tolerance = 5e-4 + 3e-2 * expected_valid.abs()
            mismatch = abs_diff > tolerance
            mismatch_count = mismatch.sum().item()
            max_abs = abs_diff.max().item()

            # Report an element with a useful magnitude. Ratios at values near
            # zero are unstable and can obscure whether scale was applied.
            sample_flat_idx = expected_valid.abs().argmax().item()
            scale_1_sample = (expected_valid[sample_flat_idx] * 2.0).item()
            scale_half_sample = actual_valid[sample_flat_idx].item()
            expected_sample = expected_valid[sample_flat_idx].item()
            observed_factor = (
                scale_half_sample / scale_1_sample
                if abs(scale_1_sample) > 1e-12
                else float("nan")
            )
            print(
                "Aqk scale response: "
                f"seq={seq_idx}, chunk={chunk_idx}, chunk_len={chunk_len}, "
                f"max_abs={max_abs:.6e}, "
                f"mismatches={mismatch_count}/{actual_valid.numel()}, "
                f"scale_1={scale_1_sample:.6e}, "
                f"scale_half={scale_half_sample:.6e}, "
                f"expected_half={expected_sample:.6e}, "
                f"observed_factor={observed_factor:.6f}"
            )
            if mismatch_count:
                failures.append(
                    f"seq={seq_idx} chunk={chunk_idx} chunk_len={chunk_len}: "
                    f"Aqk does not respond to scale (max_abs={max_abs:.6e}, "
                    f"mismatches={mismatch_count}/{actual_valid.numel()}, "
                    f"observed_factor={observed_factor:.6f}, expected=0.5)"
                )

            local_start += chunk_len
            chunk_idx += 1
        seq_start += seq_len

    return failures


def _compare(
    name,
    actual,
    expected,
    *,
    rtol=RTOL,
    atol=ATOL,
    require_same_dtype=True,
    max_bf16_ulps=None,
    valid_mask=None,
):
    """Print complete error statistics and return None or a failure summary."""
    if actual is None or expected is None:
        if actual is None and expected is None:
            print(f"{name}: both are None")
            return None
        return (
            f"{name}: optionality mismatch new={actual is not None}, "
            f"legacy={expected is not None}"
        )
    if actual.shape != expected.shape:
        return (
            f"{name}: shape mismatch new={tuple(actual.shape)}, "
            f"legacy={tuple(expected.shape)}"
        )
    dtype_failure = None
    if actual.dtype != expected.dtype:
        print(f"{name}: dtype new={actual.dtype}, legacy={expected.dtype}")
        if require_same_dtype:
            dtype_failure = (
                f"{name}: dtype mismatch new={actual.dtype}, legacy={expected.dtype}"
            )
    actual_values = actual
    expected_values = expected
    selected_indices = None
    if valid_mask is not None:
        mask = valid_mask.to(device=actual.device, dtype=torch.bool)
        if mask.shape != actual.shape:
            try:
                mask = mask.expand_as(actual)
            except RuntimeError:
                return (
                    f"{name}: valid-mask shape {tuple(valid_mask.shape)} cannot "
                    f"expand to {tuple(actual.shape)}"
                )
        selected_indices = mask.nonzero(as_tuple=False)
        actual_values = actual[mask]
        expected_values = expected[mask]
        if actual_values.numel() == 0:
            return f"{name}: valid mask selected no elements"

    actual_fp32 = actual_values.float()
    expected_fp32 = expected_values.float()
    finite = torch.isfinite(actual_fp32) & torch.isfinite(expected_fp32)
    raw_abs_diff = (actual_fp32 - expected_fp32).abs()
    abs_diff = torch.where(
        finite, raw_abs_diff, torch.full_like(raw_abs_diff, float("inf"))
    )
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    max_rel = (abs_diff / expected_fp32.abs().clamp_min(1e-6)).max().item()
    criterion = f"rtol={rtol:.1e}, atol={atol:.1e}"
    max_ulp = None
    if max_bf16_ulps is not None:
        actual_ordered = _bf16_ordered_int(actual_values)
        expected_ordered = _bf16_ordered_int(expected_values)
        ulp_diff = (actual_ordered - expected_ordered).abs()
        max_ulp = ulp_diff.max().item()
        mismatch = (ulp_diff > max_bf16_ulps).to(actual.device)
        mismatch |= ~finite
        criterion = f"bf16_ulps<={max_bf16_ulps}, max_ulp={max_ulp}"
    else:
        tolerance = atol + rtol * expected_fp32.abs()
        mismatch = (abs_diff > tolerance) | ~finite
    mismatch_count = mismatch.sum().item()
    total = mismatch.numel()
    mismatch_ratio = mismatch_count / total
    flat_max_index = abs_diff.argmax().item()
    if selected_indices is not None:
        max_index = tuple(selected_indices[flat_max_index].cpu().tolist())
        actual_at_max = actual.float()[max_index].item()
        expected_at_max = expected.float()[max_index].item()
    else:
        max_index = []
        remaining = flat_max_index
        for dim_size in reversed(abs_diff.shape):
            max_index.append(remaining % dim_size)
            remaining //= dim_size
        max_index = tuple(reversed(max_index))
        actual_at_max = actual_fp32[max_index].item()
        expected_at_max = expected_fp32[max_index].item()
    print(
        f"{name}: max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, "
        f"max_rel={max_rel:.6e}, {criterion}, "
        f"mismatches={mismatch_count}/{total} "
        f"({mismatch_ratio:.4%}), max_index={max_index}, "
        f"new={actual_at_max:.6e}, legacy={expected_at_max:.6e}"
    )
    value_failure = None
    if mismatch_count:
        value_failure = (
            f"{name}: max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, "
            f"{criterion}, mismatches={mismatch_count}/{total} "
            f"({mismatch_ratio:.4%})"
        )
    return "; ".join(
        failure for failure in (dtype_failure, value_failure) if failure is not None
    ) or None


def _record_compare(failures, name, actual, expected, **kwargs):
    failure = _compare(name, actual, expected, **kwargs)
    if failure is not None:
        failures.append(failure)


def _bf16_ordered_int(tensor):
    """Map BF16 bit patterns to monotonically ordered integers on CPU."""
    quantized = tensor.detach().float().cpu().to(torch.bfloat16).contiguous()
    bits = quantized.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)
    negative = bits.bitwise_and(0x8000).ne(0)
    magnitude = bits.bitwise_and(0x7FFF)
    return torch.where(negative, 0x8000 - magnitude, 0x8000 + bits)


def _chunk_causal_mask(tensor, seq_lens):
    """Select only meaningful lower-triangular Aqk/Akk entries."""
    assert tensor.dim() == 4
    assert sum(seq_lens) == tensor.shape[1]
    assert tensor.shape[-1] >= min(CHUNK_SIZE, max(seq_lens))
    mask = torch.zeros_like(tensor, dtype=torch.bool)
    seq_start = 0
    for seq_len in seq_lens:
        local_start = 0
        while local_start < seq_len:
            chunk_len = min(CHUNK_SIZE, seq_len - local_start)
            global_start = seq_start + local_start
            global_end = global_start + chunk_len
            causal = torch.ones(
                (chunk_len, chunk_len), dtype=torch.bool, device=tensor.device
            ).tril_()
            mask[:, global_start:global_end, :, :chunk_len] = causal.view(
                1, chunk_len, 1, chunk_len
            ).expand(
                tensor.shape[0], chunk_len, tensor.shape[2], chunk_len
            )
            local_start += chunk_len
        seq_start += seq_len
    return mask


def _record_compare_causal(
    failures, name, actual, expected, seq_lens, **kwargs
):
    _record_compare(
        failures,
        name,
        actual,
        expected,
        valid_mask=_chunk_causal_mask(actual, seq_lens),
        **kwargs,
    )


def _diagnose_output_slices(actual, expected, seq_lens, **compare_kwargs):
    """Print errors per logical sequence and per 64-token chunk."""
    seq_start = 0
    for seq_idx, seq_len in enumerate(seq_lens):
        seq_end = seq_start + seq_len
        _compare(
            f"  output seq={seq_idx} tokens=[{seq_start}:{seq_end})",
            actual[:, seq_start:seq_end],
            expected[:, seq_start:seq_end],
            **compare_kwargs,
        )
        local_start = 0
        chunk_idx = 0
        while local_start < seq_len:
            local_end = min(local_start + CHUNK_SIZE, seq_len)
            global_start = seq_start + local_start
            global_end = seq_start + local_end
            _compare(
                f"    output seq={seq_idx} chunk={chunk_idx} "
                f"tokens=[{global_start}:{global_end})",
                actual[:, global_start:global_end],
                expected[:, global_start:global_end],
                **compare_kwargs,
            )
            local_start = local_end
            chunk_idx += 1
        seq_start = seq_end


@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param((1,), id="one_token_chunk"),
        pytest.param((2,), id="partial_chunk_cur_t_2"),
        pytest.param((31,), id="partial_chunk_cur_t_31"),
        pytest.param((32,), id="partial_chunk_cur_t_32"),
        pytest.param((33,), id="partial_chunk_cur_t_33"),
        pytest.param((64,), id="one_exact_chunk"),
        pytest.param((65,), id="one_exact_chunk_plus_one_token_tail"),
        pytest.param((66,), id="one_exact_chunk_plus_tail_cur_t_2"),
        pytest.param((95,), id="one_exact_chunk_plus_tail_cur_t_31"),
        pytest.param((96,), id="one_exact_chunk_plus_tail_cur_t_32"),
        pytest.param((97,), id="one_exact_chunk_plus_tail_cur_t_33"),     
        pytest.param((1, 63, 65), id="packed_mixed_chunks"),
    ],
)
def test_migrated_operator_aqk_responds_to_scale(seq_lens):
    """The migrated operator must apply the supplied scale to every Aqk chunk."""
    inputs, initial_pool, cache_indices, query_start_loc = _make_inputs(
        seq_lens, value_dim=128
    )

    # Both launches receive clones of exactly the same tensors and state. The
    # only intentional difference is the scale attribute.
    outputs_scale_1 = _run_fused_operator_debug(
        *_clone_tensors(inputs),
        ssm_states=initial_pool.clone(),
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        output_h=False,
        scale=1.0,
    )
    outputs_scale_half = _run_fused_operator_debug(
        *_clone_tensors(inputs),
        ssm_states=initial_pool.clone(),
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        output_h=False,
        scale=0.5,
    )
    torch.npu.synchronize()

    aqk_scale_1 = outputs_scale_1[3]
    aqk_scale_half = outputs_scale_half[3]
    assert aqk_scale_1.shape == aqk_scale_half.shape
    print(
        f"\nAqk scale probe: seq_lens={seq_lens}, "
        f"shape={tuple(aqk_scale_1.shape)}, dtype={aqk_scale_1.dtype}"
    )
    failures = _check_aqk_scale_response(
        aqk_scale_1, aqk_scale_half, seq_lens
    )
    if failures:
        pytest.fail("\n".join(["Aqk scale-response failures:", *failures]))


def test_rectangular_w_is_independent_of_value_dimension():
    """Separate the generic V=64 W path from downstream state/output math.

    W is a function of Q, K, gate, beta, Aqk, and Akk.  It must not depend on
    V, the value-head dimension, or the initial recurrent state.  On A5 this
    comparison also contrasts the V=64 generic dispatch with the V=128
    Arch35 dispatch while keeping every W input bitwise identical.
    """
    seq_lens = (1, 63, 65)
    inputs64, _, cache_indices, query_start_loc = _make_inputs(
        seq_lens, value_dim=64
    )
    q, k, v64, g, beta = inputs64
    common_inputs = (
        l2norm_fwd(q.contiguous()),
        l2norm_fwd(k.contiguous()),
        g.contiguous(),
        beta.contiguous(),
    )
    q, k, g, beta = common_inputs
    v128 = torch.zeros(
        (*v64.shape[:-1], 128), device="npu", dtype=v64.dtype
    )
    state64 = torch.zeros(
        (POOL_SIZE, NUM_HEADS, 64, KEY_DIM),
        device="npu",
        dtype=torch.float32,
    )
    state128 = torch.zeros(
        (POOL_SIZE, NUM_HEADS, 128, KEY_DIM),
        device="npu",
        dtype=torch.float32,
    )
    torch.npu.synchronize()

    legacy_debug = {}
    _legacy_extend(
        q.clone(),
        k.clone(),
        v64.clone(),
        g.clone(),
        beta.clone(),
        ssm_states=state64.clone(),
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        return_intermediate_states=False,
        debug=legacy_debug,
        normalize_qk=False,
    )
    fused64 = _run_fused_operator_debug(
        q.clone(),
        k.clone(),
        v64.clone(),
        g.clone(),
        beta.clone(),
        ssm_states=state64,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        output_h=False,
        output_w=True,
        normalize_qk=False,
    )
    fused128 = _run_fused_operator_debug(
        q.clone(),
        k.clone(),
        v128,
        g.clone(),
        beta.clone(),
        ssm_states=state128,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        output_h=False,
        output_w=True,
        normalize_qk=False,
    )
    torch.npu.synchronize()

    legacy_w = legacy_debug["w"]
    fused64_w = fused64[5].permute(0, 2, 1, 3).contiguous()
    fused128_w = fused128[5].permute(0, 2, 1, 3).contiguous()
    failures = []
    _record_compare(
        failures,
        "rectangular V=64 W vs legacy",
        fused64_w,
        legacy_w,
        require_same_dtype=False,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    _record_compare(
        failures,
        "square V=128 W vs legacy",
        fused128_w,
        legacy_w,
        require_same_dtype=False,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    _record_compare(
        failures,
        "same-input W, V=64 vs V=128",
        fused64_w,
        fused128_w,
        require_same_dtype=False,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    if failures:
        pytest.fail("\n".join(["value-dimension W isolation failures:", *failures]))


@pytest.mark.parametrize("value_dim", [32, 64, 96, 128, 192])
@pytest.mark.parametrize("cur_t", [1, 2, 15, 16, 17, 31, 32, 33, 63])
def test_w_fallback_dimension_and_tail_sweep(value_dim, cur_t):
    """Map W failures across value dimensions and tail boundaries.

    W does not depend on V. Q/K/g/beta come from one V=64 fixture, while
    only the fused launch's value tensor and state use the parameterized V.
    """
    seq_lens = (cur_t,)
    inputs64, _, cache_indices, query_start_loc = _make_inputs(
        seq_lens, value_dim=64
    )
    q, k, v64, g, beta = inputs64
    q = l2norm_fwd(q.contiguous())
    k = l2norm_fwd(k.contiguous())
    value = torch.zeros(
        (*v64.shape[:-1], value_dim), device="npu", dtype=v64.dtype
    )
    fused_state = torch.zeros(
        (POOL_SIZE, NUM_HEADS, value_dim, KEY_DIM),
        device="npu",
        dtype=torch.float32,
    )
    legacy_state = torch.zeros(
        (POOL_SIZE, NUM_HEADS, 64, KEY_DIM),
        device="npu",
        dtype=torch.float32,
    )
    torch.npu.synchronize()

    legacy_debug = {}
    _legacy_extend(
        q.clone(),
        k.clone(),
        v64.clone(),
        g.clone(),
        beta.clone(),
        ssm_states=legacy_state,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        return_intermediate_states=False,
        debug=legacy_debug,
        normalize_qk=False,
    )
    fused = _run_fused_operator_debug(
        q.clone(),
        k.clone(),
        value,
        g.clone(),
        beta.clone(),
        ssm_states=fused_state,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        output_h=False,
        output_w=True,
        normalize_qk=False,
    )
    torch.npu.synchronize()

    fused_w = fused[5].permute(0, 2, 1, 3).contiguous()
    failure = _compare(
        f"W sweep V={value_dim}, curT={cur_t}",
        fused_w,
        legacy_debug["w"],
        require_same_dtype=False,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    if failure:
        pytest.fail(failure)


def test_rectangular_initial_state_contribution_matches_legacy():
    """Split local attention from the contribution of recurrent state.

    Subtracting the zero-state output from the random-state output cancels the
    local Aqk/V-new term.  A failure only in this difference points to the
    generic QG x H/output path; a zero-state failure points upstream instead.
    """
    seq_lens = (1, 63, 65)
    inputs, initial_pool, cache_indices, query_start_loc = _make_inputs(
        seq_lens, value_dim=64
    )
    comparison_inputs = (
        l2norm_fwd(inputs[0].contiguous()),
        l2norm_fwd(inputs[1].contiguous()),
        inputs[2].contiguous(),
        inputs[3].contiguous(),
        inputs[4].contiguous(),
    )
    zero_pool = torch.zeros_like(initial_pool)
    torch.npu.synchronize()

    def run_legacy(state_pool):
        return _legacy_extend(
            *_clone_tensors(comparison_inputs),
            ssm_states=state_pool.clone(),
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            return_intermediate_states=False,
            normalize_qk=False,
        )

    def run_fused(state_pool):
        return _run_fused_operator_debug(
            *_clone_tensors(comparison_inputs),
            ssm_states=state_pool,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            output_h=False,
            normalize_qk=False,
        )[0]

    legacy_random = run_legacy(initial_pool)
    legacy_zero = run_legacy(zero_pool)
    fused_random = run_fused(initial_pool.clone())
    fused_zero = run_fused(zero_pool.clone())
    torch.npu.synchronize()

    legacy_state_contribution = legacy_random.float() - legacy_zero.float()
    fused_state_contribution = fused_random.float() - fused_zero.float()
    failures = []
    _record_compare(
        failures,
        "rectangular zero-state attention",
        fused_zero,
        legacy_zero,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    _record_compare(
        failures,
        "rectangular initial-state contribution",
        fused_state_contribution,
        legacy_state_contribution,
        require_same_dtype=False,
        rtol=FP32_RECON_RTOL,
        atol=FP32_RECON_ATOL,
    )
    _diagnose_output_slices(
        fused_zero,
        legacy_zero,
        seq_lens,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    _diagnose_output_slices(
        fused_state_contribution,
        legacy_state_contribution,
        seq_lens,
        rtol=FP32_RECON_RTOL,
        atol=FP32_RECON_ATOL,
    )
    if failures:
        pytest.fail("\n".join(["rectangular state isolation failures:", *failures]))


def _explicit_causal_aqk_vnew_fp32(aqk, v_new):
    """Reconstruct the causal local term in FP32 from public outputs."""
    cur_t = v_new.shape[1]
    causal = torch.tril(
        torch.ones((cur_t, cur_t), device=aqk.device, dtype=torch.float32)
    ).view(1, cur_t, 1, cur_t)
    coefficients = aqk[..., :cur_t].float() * causal
    return torch.einsum("bthj,bjhv->bthv", coefficients, v_new.float())


def _explicit_causal_aqk_vnew(aqk, v_new, output_dtype):
    """Reconstruct the causal local term from public operator outputs."""
    return _explicit_causal_aqk_vnew_fp32(aqk, v_new).to(output_dtype)


def _single_token_effective_weight_diagnostics(output, aqk, v_new, qg):
    """Determine whether a bad one-token output used a bad scalar weight.

    For a single token with zero initial state, every output row must be one
    scalar multiple of the corresponding V-new row.  A least-squares fit
    separates a wrong/stale scalar coefficient from corruption in the vector
    load, multiply, accumulation, or writeback path.
    """
    assert output.shape == v_new.shape
    assert output.shape[:3] == aqk.shape[:3] and aqk.shape[-1] == 1
    assert output.shape[:3] == qg.shape[:3]
    output_fp32 = output.float()
    v_new_fp32 = v_new.float()
    expected_weight = aqk[..., 0].float()
    qg_fp32 = qg.float()
    lines = []

    for batch in range(output.shape[0]):
        for token in range(output.shape[1]):
            for head in range(output.shape[2]):
                actual_row = output_fp32[batch, token, head]
                value_row = v_new_fp32[batch, token, head]
                denominator = value_row.square().sum().clamp_min(1e-20)
                fitted_weight = (actual_row * value_row).sum() / denominator
                fitted_row = fitted_weight * value_row
                residual = actual_row - fitted_row
                residual_ratio = (
                    residual.square().sum().sqrt()
                    / actual_row.square().sum().sqrt().clamp_min(1e-20)
                )
                expected = expected_weight[batch, token, head]
                qg_row = qg_fp32[batch, token, head]
                qg_distance = (qg_row - fitted_weight).abs()
                closest_qg_index = qg_distance.argmin()
                qg0 = qg_row[0]
                closest_qg = qg_row[closest_qg_index]

                block_fits = []
                for col in range(0, output.shape[-1], 64):
                    block_actual = actual_row[col : col + 64]
                    block_value = value_row[col : col + 64]
                    block_denominator = block_value.square().sum().clamp_min(1e-20)
                    block_weight = (
                        (block_actual * block_value).sum() / block_denominator
                    )
                    block_fits.append(
                        f"[{col}:{min(col + 64, output.shape[-1])})="
                        f"{block_weight.item():.6e}"
                    )

                qg0_matches = (fitted_weight - qg0).abs().item() <= (
                    5e-4 + 2e-2 * qg0.abs().item()
                )
                if residual_ratio.item() > 5e-2:
                    classification = "vector/load/writeback path suspect"
                elif qg0_matches:
                    classification = "stale prior QG[0] coefficient suspect"
                else:
                    classification = "scalar-weight path suspect (not QG[0])"
                lines.append(
                    f"effective-weight b={batch}, t={token}, h={head}: "
                    f"expected_aqk={expected.item():.6e}, "
                    f"fitted={fitted_weight.item():.6e}, "
                    f"weight_abs_diff={(fitted_weight - expected).abs().item():.6e}, "
                    f"prior_qg0={qg0.item():.6e}, "
                    f"qg0_abs_diff={(fitted_weight - qg0).abs().item():.6e}, "
                    f"closest_qg_index={closest_qg_index.item()}, "
                    f"closest_qg={closest_qg.item():.6e}, "
                    f"closest_qg_abs_diff={qg_distance[closest_qg_index].item():.6e}, "
                    f"fit_residual_max={residual.abs().max().item():.6e}, "
                    f"fit_residual_ratio={residual_ratio.item():.6e}, "
                    f"classification={classification}, "
                    f"64-col fits: {', '.join(block_fits)}"
                )
    return lines


def _short_tail_effective_coefficient_diagnostics(output, aqk, v_new, qg):
    """Recover the short-tail coefficient matrix actually applied to V-new."""
    assert output.shape == v_new.shape
    token_count = output.shape[1]
    assert token_count in (2, 3)
    output_cpu = output.float().cpu()
    v_new_cpu = v_new.float().cpu()
    aqk_cpu = aqk[..., :token_count].float().cpu()
    qg_cpu = qg.float().cpu()
    lines = []

    for batch in range(output.shape[0]):
        for head in range(output.shape[2]):
            values = v_new_cpu[batch, :, head, :]
            actual = output_cpu[batch, :, head, :]
            gram = values @ values.transpose(0, 1)
            effective = actual @ values.transpose(0, 1) @ torch.linalg.pinv(gram)
            reconstructed = effective @ values
            residual_ratio = (
                (actual - reconstructed).square().sum().sqrt()
                / actual.square().sum().sqrt().clamp_min(1e-20)
            )
            qg_flat = qg_cpu[batch, :, head, :].reshape(-1)

            lines.append(
                f"effective-matrix b={batch}, h={head}: "
                f"fit_residual_ratio={residual_ratio.item():.6e}"
            )
            for row in range(token_count):
                for col in range(token_count):
                    fitted = effective[row, col]
                    expected = aqk_cpu[batch, row, head, col]
                    distances = (qg_flat - fitted).abs()
                    closest_flat = distances.argmin()
                    closest_token = closest_flat.item() // qg.shape[-1]
                    closest_dim = closest_flat.item() % qg.shape[-1]
                    same_dim_qg = ", ".join(
                        f"qg_token{token}_dim{col}="
                        f"{qg_cpu[batch, token, head, col].item():.6e}"
                        for token in range(token_count)
                    )
                    lines.append(
                        f"  a{row}{col}: expected_aqk={expected.item():.6e}, "
                        f"fitted={fitted.item():.6e}, "
                        f"{same_dim_qg}, "
                        f"closest_qg=(token={closest_token}, dim={closest_dim}, "
                        f"value={qg_flat[closest_flat].item():.6e}, "
                        f"abs_diff={distances[closest_flat].item():.6e})"
                    )
    return lines


def _explicit_single_chunk_output(qg, h, aqk, v_new, scale):
    """Reconstruct both terms of one logical chunk in FP32.

    Public layouts at this point are:

      QG/Aqk/V-new: [B, T, H, K-or-V]
      H:             [B, 1, H, V, K] (state_v_first=True)

    Aqk already contains the QK scale.  The recurrent-state term consumes
    unscaled QG, so its scale is applied explicitly here just as finalize does.
    """
    assert h.dim() == 5 and h.shape[1] == 1
    state = torch.einsum(
        "bthk,bhvk->bthv", qg.float(), h[:, 0].float()
    ) * scale
    local = _explicit_causal_aqk_vnew_fp32(aqk, v_new)
    return state, local, state + local


@pytest.mark.parametrize("value_dim", [32, 64, 96, 128, 192])
def test_rectangular_single_token_local_output_matches_explicit_aqk_vnew(
    value_dim,
):
    """Verify the generic local-output GEMM from its own public inputs.

    With one token and a zero initial state, the recurrent-state contribution
    is exactly zero and the local term has no reduction-order ambiguity:

        attention[b, 0, h, v] = Aqk[b, 0, h, 0] * V_new[b, 0, h, v]

    Comparing the fused output with a product reconstructed from that same
    launch distinguishes a finalize-kernel error from upstream Aqk/V-new
    precision differences.  The legacy self-check guards the test formula.
    """
    seq_lens = (1,)
    inputs, initial_pool, cache_indices, query_start_loc = _make_inputs(
        seq_lens, value_dim=value_dim
    )
    initial_pool.zero_()
    comparison_inputs = (
        l2norm_fwd(inputs[0].contiguous()),
        l2norm_fwd(inputs[1].contiguous()),
        inputs[2].contiguous(),
        inputs[3].contiguous(),
        inputs[4].contiguous(),
    )
    torch.npu.synchronize()

    legacy_debug = {}
    legacy_out = _legacy_extend(
        *_clone_tensors(comparison_inputs),
        ssm_states=initial_pool.clone(),
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        return_intermediate_states=False,
        debug=legacy_debug,
        normalize_qk=False,
    )
    fused = _run_fused_operator_debug(
        *_clone_tensors(comparison_inputs),
        ssm_states=initial_pool.clone(),
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        output_h=False,
        output_qg=True,
        output_v_new=True,
        normalize_qk=False,
    )
    torch.npu.synchronize()

    fused_out = fused[0]
    fused_aqk = fused[3].permute(0, 2, 1, 3).contiguous()[..., :1]
    fused_qg = fused[7].permute(0, 2, 1, 3).contiguous()
    fused_v_new = fused[9].permute(0, 2, 1, 3).contiguous()
    fused_explicit_local = _explicit_causal_aqk_vnew(
        fused_aqk, fused_v_new, fused_out.dtype
    )

    legacy_aqk = legacy_debug["aqk"][..., :1]
    legacy_v_new = legacy_debug["v_new"]
    legacy_explicit_local = _explicit_causal_aqk_vnew(
        legacy_aqk, legacy_v_new, legacy_out.dtype
    )

    failures = []
    _record_compare(
        failures,
        f"fused one-token V={value_dim} output vs fused Aqk * V_new",
        fused_out,
        fused_explicit_local,
        max_bf16_ulps=BF16_POINTWISE_ULPS,
    )
    _record_compare(
        failures,
        f"legacy one-token V={value_dim} output vs legacy Aqk * V_new",
        legacy_out,
        legacy_explicit_local,
        max_bf16_ulps=BF16_POINTWISE_ULPS,
    )
    _record_compare(
        failures,
        f"explicit one-token V={value_dim} local term, fused vs legacy",
        fused_explicit_local,
        legacy_explicit_local,
        require_same_dtype=False,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    if failures:
        effective_weight_diagnostics = _single_token_effective_weight_diagnostics(
            fused_out, fused_aqk, fused_v_new, fused_qg
        )
        pytest.fail(
            "\n".join(
                [
                    "one-token local-output isolation failures:",
                    *effective_weight_diagnostics,
                    *failures,
                ]
            )
        )


@pytest.mark.parametrize("value_dim", [64, 96, 128])
@pytest.mark.parametrize("token_count", [2, 3])
def test_rectangular_short_tail_effective_coefficients(value_dim, token_count):
    """Identify whether the generic local path reuses QG[1] or QG[2]."""
    seq_lens = (token_count,)
    inputs, initial_pool, cache_indices, query_start_loc = _make_inputs(
        seq_lens, value_dim=value_dim
    )
    initial_pool.zero_()
    comparison_inputs = (
        l2norm_fwd(inputs[0].contiguous()),
        l2norm_fwd(inputs[1].contiguous()),
        inputs[2].contiguous(),
        inputs[3].contiguous(),
        inputs[4].contiguous(),
    )
    fused = _run_fused_operator_debug(
        *_clone_tensors(comparison_inputs),
        ssm_states=initial_pool.clone(),
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        output_h=False,
        output_qg=True,
        output_v_new=True,
        normalize_qk=False,
    )
    torch.npu.synchronize()

    fused_out = fused[0]
    fused_aqk = fused[3].permute(0, 2, 1, 3).contiguous()
    fused_qg = fused[7].permute(0, 2, 1, 3).contiguous()
    fused_v_new = fused[9].permute(0, 2, 1, 3).contiguous()
    fused_explicit = _explicit_causal_aqk_vnew(
        fused_aqk, fused_v_new, fused_out.dtype
    )

    failure = _compare(
        f"short-tail T={token_count}, V={value_dim} output vs fused Aqk * V_new",
        fused_out,
        fused_explicit,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    if failure is not None:
        diagnostics = _short_tail_effective_coefficient_diagnostics(
            fused_out, fused_aqk, fused_v_new, fused_qg
        )
        pytest.fail(
            "\n".join(
                [
                    f"T={token_count} effective-coefficient isolation failure:",
                    *diagnostics,
                    failure,
                ]
            )
        )


@pytest.mark.parametrize("value_dim", [64, 96, 128])
@pytest.mark.parametrize("cur_t", range(2, 18))
def test_local_output_dimension_and_tail_sweep(value_dim, cur_t):
    """Locate every local-output transition through the Cube-tail boundary."""
    seq_lens = (cur_t,)
    inputs, initial_pool, cache_indices, query_start_loc = _make_inputs(
        seq_lens, value_dim=value_dim
    )
    initial_pool.zero_()
    comparison_inputs = (
        l2norm_fwd(inputs[0].contiguous()),
        l2norm_fwd(inputs[1].contiguous()),
        inputs[2].contiguous(),
        inputs[3].contiguous(),
        inputs[4].contiguous(),
    )
    torch.npu.synchronize()

    legacy_debug = {}
    legacy_out = _legacy_extend(
        *_clone_tensors(comparison_inputs),
        ssm_states=initial_pool.clone(),
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        return_intermediate_states=False,
        debug=legacy_debug,
        normalize_qk=False,
    )
    fused = _run_fused_operator_debug(
        *_clone_tensors(comparison_inputs),
        ssm_states=initial_pool.clone(),
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        output_h=False,
        output_v_new=True,
        normalize_qk=False,
    )
    torch.npu.synchronize()

    fused_out = fused[0]
    fused_aqk = fused[3].permute(0, 2, 1, 3).contiguous()
    fused_v_new = fused[9].permute(0, 2, 1, 3).contiguous()
    fused_explicit = _explicit_causal_aqk_vnew(
        fused_aqk, fused_v_new, fused_out.dtype
    )
    legacy_aqk = legacy_debug["aqk"]
    legacy_v_new = legacy_debug["v_new"]
    legacy_explicit = _explicit_causal_aqk_vnew(
        legacy_aqk, legacy_v_new, legacy_out.dtype
    )

    label = f"V={value_dim}, curT={cur_t}"
    failures = []
    _record_compare_causal(
        failures,
        f"{label} Aqk",
        fused_aqk[..., :cur_t],
        legacy_aqk[..., :cur_t],
        (cur_t,),
        require_same_dtype=False,
        max_bf16_ulps=BF16_POINTWISE_ULPS,
    )
    _record_compare(
        failures,
        f"{label} V_new",
        fused_v_new,
        legacy_v_new,
        require_same_dtype=False,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    _record_compare(
        failures,
        f"{label} explicit local, fused vs legacy",
        fused_explicit,
        legacy_explicit,
        require_same_dtype=False,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    _record_compare(
        failures,
        f"{label} fused output vs its explicit local",
        fused_out,
        fused_explicit,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    _record_compare(
        failures,
        f"{label} legacy output vs its explicit local",
        legacy_out,
        legacy_explicit,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    if failures:
        pytest.fail(
            "\n".join([f"local-output sweep failures ({label}):", *failures])
        )


@pytest.mark.parametrize("value_dim", [64, 128])
@pytest.mark.parametrize("cur_t", [1, 2, 3, 7, 14, 15])
@pytest.mark.parametrize(
    "zero_initial_state",
    [
        pytest.param(True, id="zero_state"),
        pytest.param(False, id="random_state"),
    ],
)
def test_finalize_output_matches_all_public_terms(
    value_dim, cur_t, zero_initial_state
):
    """Reconstruct finalize output from QG/H and Aqk/V-new.

    This isolates three regions without relying only on the legacy result:

      state = scale * QG @ H
      local = causal(Aqk) @ V-new
      output = state + local

    The fused output is compared with a reconstruction made from that same
    launch.  The legacy self-check validates the formula, while component-wise
    fused/legacy comparisons show whether an upstream public value is already
    wrong before finalize consumes it.  V=128 is the dispatch control for the
    rectangular V=64 path.
    """
    seq_lens = (cur_t,)
    inputs, initial_pool, cache_indices, query_start_loc = _make_inputs(
        seq_lens, value_dim=value_dim
    )
    if zero_initial_state:
        initial_pool.zero_()
    comparison_inputs = (
        l2norm_fwd(inputs[0].contiguous()),
        l2norm_fwd(inputs[1].contiguous()),
        inputs[2].contiguous(),
        inputs[3].contiguous(),
        inputs[4].contiguous(),
    )
    scale = KEY_DIM**-0.5
    torch.npu.synchronize()

    legacy_debug = {}
    legacy_out = _legacy_extend(
        *_clone_tensors(comparison_inputs),
        ssm_states=initial_pool.clone(),
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        return_intermediate_states=False,
        debug=legacy_debug,
        normalize_qk=False,
    )
    fused = _run_fused_operator_debug(
        *_clone_tensors(comparison_inputs),
        ssm_states=initial_pool.clone(),
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        output_h=True,
        output_qg=True,
        output_v_new=True,
        normalize_qk=False,
    )
    torch.npu.synchronize()

    fused_out = fused[0]
    fused_qg = fused[7].permute(0, 2, 1, 3).contiguous()
    fused_aqk = fused[3].permute(0, 2, 1, 3).contiguous()
    fused_v_new = fused[9].permute(0, 2, 1, 3).contiguous()
    fused_h = fused[10]
    fused_state, fused_local, fused_reconstructed = (
        _explicit_single_chunk_output(
            fused_qg, fused_h, fused_aqk, fused_v_new, scale
        )
    )

    legacy_state, legacy_local, legacy_reconstructed = (
        _explicit_single_chunk_output(
            legacy_debug["qg"],
            legacy_debug["h"],
            legacy_debug["aqk"],
            legacy_debug["v_new"],
            scale,
        )
    )

    state_label = "zero" if zero_initial_state else "random"
    label = f"V={value_dim}, curT={cur_t}, state={state_label}"
    failures = []
    if zero_initial_state:
        _record_compare(
            failures,
            f"{label} public H must be zero",
            fused_h,
            torch.zeros_like(fused_h),
            rtol=0.0,
            atol=0.0,
        )
    _record_compare(
        failures,
        f"{label} public state term, fused vs legacy",
        fused_state,
        legacy_state,
        require_same_dtype=False,
        rtol=FP32_RECON_RTOL,
        atol=FP32_RECON_ATOL,
    )
    _record_compare(
        failures,
        f"{label} public local term, fused vs legacy",
        fused_local,
        legacy_local,
        require_same_dtype=False,
        rtol=FP32_RECON_RTOL,
        atol=FP32_RECON_ATOL,
    )
    _record_compare(
        failures,
        f"{label} all public terms, fused vs legacy",
        fused_reconstructed,
        legacy_reconstructed,
        require_same_dtype=False,
        rtol=FP32_RECON_RTOL,
        atol=FP32_RECON_ATOL,
    )
    _record_compare(
        failures,
        f"{label} fused output vs all its public terms",
        fused_out,
        fused_reconstructed.to(fused_out.dtype),
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    _record_compare(
        failures,
        f"{label} legacy output vs all its public terms",
        legacy_out,
        legacy_reconstructed.to(legacy_out.dtype),
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    if failures:
        pytest.fail(
            "\n".join(
                [f"full finalize reconstruction failures ({label}):", *failures]
            )
        )


@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param((1,), id="cur_t_1_single_row"),
        pytest.param((2,), id="cur_t_2_one_row_per_subblock"),
        pytest.param((3,), id="cur_t_3_first_multi_row_subblock"),
        pytest.param((15,), id="cur_t_15_multi_row_tail"),
        pytest.param((16,), id="cur_t_16_cube_control"),
        pytest.param((1, 63, 65), id="packed_len1_first"),
        pytest.param((63, 1, 65), id="packed_len1_middle"),
        pytest.param((63, 65, 1), id="packed_len1_last"),
    ],
)
@pytest.mark.parametrize(
    "zero_initial_state",
    [
        pytest.param(False, id="random_state"),
        pytest.param(True, id="zero_state"),
    ],
)
def test_tail_v_new_matches_explicit_w_h(seq_lens, zero_initial_state):
    """Check each <=16-token V-new block against explicit U - W @ H.

    This launches the fused operator exactly once.  It distinguishes the
    short-tail workspace calculation from the upstream W/U/H producers and
    covers the first boundary where an AIV sub-block processes multiple token
    rows (curT=3), the observed curT=15 failure, and curT=16 as a Cube control.
    """
    inputs, initial_pool, cache_indices, query_start_loc = _make_inputs(
        seq_lens, value_dim=128
    )
    if zero_initial_state:
        initial_pool.zero_()

    normalized_inputs = (
        l2norm_fwd(inputs[0].contiguous()),
        l2norm_fwd(inputs[1].contiguous()),
        inputs[2].contiguous(),
        inputs[3].contiguous(),
        inputs[4].contiguous(),
    )
    torch.npu.synchronize()

    outputs = _run_fused_operator_debug(
        *_clone_tensors(normalized_inputs),
        ssm_states=initial_pool,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        output_h=True,
        output_w=True,
        output_u=True,
        output_v_new=True,
        normalize_qk=False,
    )
    torch.npu.synchronize()

    failures = []
    required_outputs = ((5, "w"), (6, "u"), (9, "v_new"), (10, "h"))
    for output_idx, output_name in required_outputs:
        if outputs[output_idx] is None:
            failures.append(f"{output_name}: optional output is None")
        elif not torch.isfinite(outputs[output_idx]).all():
            failures.append(f"{output_name}: contains NaN or Inf")
    if failures:
        pytest.fail("\n".join(["tail V-new prerequisite failures:", *failures]))

    # Fused BSND intermediates are head-major:
    #   W     [B, H, total_T, K]
    #   U/Vn  [B, H, total_T, V]
    #   H     [B, total_chunks, H, V, K]  (state_v_first=True)
    w = outputs[5].detach().cpu().float()
    u_bf16 = outputs[6].detach().cpu()
    u = outputs[6].detach().cpu().float()
    v_new = outputs[9].detach().cpu()
    h = outputs[10].detach().cpu().float()

    token_offset = 0
    global_chunk_idx = 0
    checked_chunks = 0
    for seq_idx, seq_len in enumerate(seq_lens):
        local_chunk_start = 0
        while local_chunk_start < seq_len:
            cur_t = min(CHUNK_SIZE, seq_len - local_chunk_start)
            global_token_start = token_offset + local_chunk_start
            global_token_end = global_token_start + cur_t

            # curT<16 uses ComputeTailVWorkspace; curT=16 is the nearest Cube
            # control. Larger chunks are irrelevant to this synchronization
            # boundary and are intentionally skipped.
            if cur_t <= 16:
                w_chunk = w[:, :, global_token_start:global_token_end, :]
                u_chunk = u[:, :, global_token_start:global_token_end, :]
                h_chunk = h[:, global_chunk_idx, :, :, :]
                wh = torch.zeros_like(u_chunk, dtype=torch.float32, device="cpu")

                # Match ComputeTailVWorkspace's deterministic K-order:
                # wh[b,h,t,v] += W[b,h,t,k] * H[b,h,v,k].
                for k_idx in range(w_chunk.shape[-1]):
                    wh.add_(
                        w_chunk[..., k_idx].unsqueeze(-1)
                        * h_chunk[..., k_idx].unsqueeze(-2)
                    )

                expected_v_new = (u_chunk - wh).to(torch.bfloat16)
                actual_v_new = v_new[
                    :, :, global_token_start:global_token_end, :
                ]
                path = "vector" if cur_t < 16 else "cube-control"
                _record_compare(
                    failures,
                    f"seq={seq_idx} chunk={global_chunk_idx} "
                    f"tokens=[{global_token_start}:{global_token_end}) "
                    f"cur_t={cur_t} path={path} v_new vs u-w@h",
                    actual_v_new,
                    expected_v_new,
                    max_bf16_ulps=BF16_REDUCTION_ULPS,
                )

                # The first chunk of every sequence receives the supplied
                # initial H. With an all-zero state, W @ H is exactly zero, so
                # BF16 V-new must be bitwise identical to BF16 U. This strict
                # control detects stale/nonzero workspace values that a normal
                # numerical tolerance could hide.
                if zero_initial_state and local_chunk_start == 0:
                    _record_compare(
                        failures,
                        f"seq={seq_idx} chunk={global_chunk_idx} "
                        f"cur_t={cur_t} zero-state h",
                        h[:, global_chunk_idx, :, :, :],
                        torch.zeros_like(h[:, global_chunk_idx, :, :, :]),
                        rtol=0.0,
                        atol=0.0,
                    )
                    _record_compare(
                        failures,
                        f"seq={seq_idx} chunk={global_chunk_idx} "
                        f"cur_t={cur_t} zero-state v_new vs u",
                        actual_v_new,
                        u_bf16[:, :, global_token_start:global_token_end, :],
                        rtol=0.0,
                        atol=0.0,
                    )
                checked_chunks += 1

            local_chunk_start += cur_t
            global_chunk_idx += 1
        token_offset += seq_len

    assert checked_chunks > 0
    if failures:
        pytest.fail("\n".join(["tail V-new formula failures:", *failures]))


def test_extend_padding_cache_index_does_not_overwrite_last_state_slot():
    """A -1 padded request must not alias the final persistent-state slot."""
    inputs, initial_pool, _, query_start_loc = _make_inputs((1, 1), value_dim=128)
    cache_indices = torch.tensor((5, -1), dtype=torch.int64, device="npu")
    fused_pool = initial_pool.clone()
    last_slot_before = fused_pool[-1].clone()

    _AscendKDAExtendKernel().extend(
        *_clone_tensors(inputs),
        ssm_states=fused_pool,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        extend_seq_lens_cpu=[1, 1],
        return_intermediate_states=False,
    )
    torch.npu.synchronize()

    assert torch.equal(fused_pool[-1], last_slot_before)


@pytest.mark.parametrize(
    "seq_lens,value_dim",
    [
        pytest.param((1,), 128, id="one_token_tail"),
        pytest.param((2,), 128, id="tail_vector_cur_t_2"),
        pytest.param((15,), 128, id="tail_vector_cur_t_15"),
        pytest.param((16,), 128, id="tail_boundary_cur_t_16"),
        pytest.param((17,), 128, id="post_tail_boundary_cur_t_17"),
        pytest.param((63,), 128, id="one_partial_chunk"),
        pytest.param((64,), 128, id="one_exact_chunk"),
        pytest.param((65,), 128, id="one_chunk_plus_tail"),
        pytest.param((128,), 128, id="two_exact_chunks"),
        pytest.param((64, 64), 128, id="packed_two_exact_chunks"),
        pytest.param((1, 63, 65), 128, id="packed_varlen_len1_first"),
        pytest.param((63, 1, 65), 128, id="packed_varlen_len1_middle"),
        pytest.param((63, 65, 1), 128, id="packed_varlen_len1_last"),
        pytest.param((1, 63, 65), 64, id="packed_rectangular_state"),
    ],
)
def test_fused_extend_matches_legacy(
    seq_lens,
    value_dim,
):
    inputs, initial_pool, cache_indices, query_start_loc = _make_inputs(
        seq_lens, value_dim
    )
    print(f"\ncase: seq_lens={seq_lens}, value_dim={value_dim}")
    print(
        "kernel zero-init diagnostics: "
        "workspace="
        f"{os.getenv('SGLANG_CHUNK_KDA_DIAG_ZERO_WORKSPACE', '0')}, "
        "final_state="
        f"{os.getenv('SGLANG_CHUNK_KDA_DIAG_ZERO_FINAL_STATE', '0')}"
    )
    print(
        "input dtypes: "
        f"q={inputs[0].dtype}, k={inputs[1].dtype}, v={inputs[2].dtype}, "
        f"g={inputs[3].dtype}, beta={inputs[4].dtype}, "
        f"state={initial_pool.dtype}"
    )
    legacy_pool = initial_pool.clone()
    fused_pool = initial_pool.clone()
    legacy_debug = {}

    # Normalize Q/K once, then give clones of these exact tensors to the old
    # decomposed path and the fused operator. Each implementation runs once.
    comparison_inputs = (
        l2norm_fwd(inputs[0].contiguous()),
        l2norm_fwd(inputs[1].contiguous()),
        inputs[2].contiguous(),
        inputs[3].contiguous(),
        inputs[4].contiguous(),
    )
    torch.npu.synchronize()

    legacy_out, legacy_h = _legacy_extend(
        *_clone_tensors(comparison_inputs),
        ssm_states=legacy_pool,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        return_intermediate_states=True,
        debug=legacy_debug,
        normalize_qk=False,
    )
    fused_debug = _run_fused_operator_debug(
        *_clone_tensors(comparison_inputs),
        ssm_states=fused_pool,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        output_h=True,
        output_gk=True,
        output_w=True,
        output_u=True,
        output_qg=True,
        output_kg=True,
        output_v_new=True,
        normalize_qk=False,
    )
    torch.npu.synchronize()
    failures = []

    fused_out = fused_debug[0]
    fused_final_state = fused_debug[1]
    fused_h = fused_debug[10]
    expected_chunks = sum(
        (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE for seq_len in seq_lens
    )
    assert fused_h.shape == (
        1,
        expected_chunks,
        NUM_HEADS,
        value_dim,
        KEY_DIM,
    )
    _record_compare(
        failures,
        "intermediate chunk state h",
        fused_h,
        legacy_h,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    for chunk_idx in range(expected_chunks):
        _compare(
            f"  intermediate h chunk={chunk_idx}",
            fused_h[:, chunk_idx],
            legacy_h[:, chunk_idx],
            max_bf16_ulps=BF16_REDUCTION_ULPS,
        )

    assert fused_out.shape == (1, sum(seq_lens), NUM_HEADS, value_dim)
    _record_compare(
        failures,
        "attention output",
        fused_out,
        legacy_out,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    _diagnose_output_slices(
        fused_out,
        legacy_out,
        seq_lens,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )

    num_sequences = len(seq_lens)
    source_indices = cache_indices[:num_sequences].to(torch.long)
    legacy_final_state = legacy_pool.index_select(0, source_indices)
    _record_compare(
        failures,
        "final state",
        fused_final_state,
        legacy_final_state,
        rtol=FP32_STATE_RTOL,
        atol=FP32_STATE_ATOL,
    )
    for seq_idx, slot in enumerate(source_indices.cpu().tolist()):
        _compare(
            f"  final state seq={seq_idx} slot={slot}",
            fused_final_state[seq_idx],
            legacy_pool[slot],
            rtol=FP32_STATE_RTOL,
            atol=FP32_STATE_ATOL,
        )
    # Reproduce the adapter's state writeback without launching the fused op a
    # second time, then compare the complete persistent pool as well.
    fused_pool.index_copy_(0, source_indices, fused_final_state)
    _record_compare(
        failures,
        "persistent state pool",
        fused_pool,
        legacy_pool,
        rtol=FP32_STATE_RTOL,
        atol=FP32_STATE_ATOL,
    )

    def migrated_to_token_major(tensor):
        # The migrated BSND operator contract is always [B, H, T, C], while
        # the decomposed SGLang intermediates are [B, T, H, C]. Do not infer
        # layout from shape: when T == H (for example T=H=2), both layouts have
        # the same visible shape and a shape-based check silently skips the
        # required permutation.
        if tensor is not None and tensor.dim() == 4:
            return tensor.permute(0, 2, 1, 3).contiguous()
        return tensor

    fused_gk = migrated_to_token_major(fused_debug[2])
    fused_aqk = migrated_to_token_major(fused_debug[3])
    fused_akk = migrated_to_token_major(fused_debug[4])
    fused_w = migrated_to_token_major(fused_debug[5])
    fused_u = migrated_to_token_major(fused_debug[6])
    fused_qg = migrated_to_token_major(fused_debug[7])
    fused_kg = migrated_to_token_major(fused_debug[8])
    fused_v_new = migrated_to_token_major(fused_debug[9])
    _record_compare(
        failures,
        "operator intermediate gk",
        fused_gk,
        legacy_debug["gk"],
        rtol=FP32_GK_RTOL,
        atol=FP32_GK_ATOL,
    )
    _record_compare_causal(
        failures,
        "operator intermediate Aqk",
        fused_aqk,
        legacy_debug["aqk"],
        seq_lens,
        require_same_dtype=False,
        max_bf16_ulps=BF16_POINTWISE_ULPS,
    )
    _record_compare_causal(
        failures,
        "operator intermediate Akk",
        fused_akk,
        legacy_debug["akk"],
        seq_lens,
        require_same_dtype=False,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    _record_compare(
        failures,
        "operator intermediate w",
        fused_w,
        legacy_debug["w"],
        require_same_dtype=False,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    _record_compare(
        failures,
        "operator intermediate u",
        fused_u,
        legacy_debug["u"],
        require_same_dtype=False,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )
    _record_compare(
        failures,
        "operator intermediate qg",
        fused_qg,
        legacy_debug["qg"],
        require_same_dtype=False,
        max_bf16_ulps=BF16_POINTWISE_ULPS,
    )
    _record_compare(
        failures,
        "operator intermediate kg",
        fused_kg,
        legacy_debug["kg"],
        require_same_dtype=False,
        max_bf16_ulps=BF16_POINTWISE_ULPS,
    )
    _record_compare(
        failures,
        "operator intermediate v_new",
        fused_v_new,
        legacy_debug["v_new"],
        require_same_dtype=False,
        max_bf16_ulps=BF16_REDUCTION_ULPS,
    )

    active_slots = set(cache_indices.cpu().tolist())
    inactive_indices = torch.tensor(
        [index for index in range(POOL_SIZE) if index not in active_slots],
        dtype=torch.int64,
        device="npu",
    )
    inactive_unchanged = torch.equal(
        fused_pool.index_select(0, inactive_indices),
        initial_pool.index_select(0, inactive_indices),
    )
    print(f"inactive state-cache slots unchanged: {inactive_unchanged}")
    if not inactive_unchanged:
        failures.append("the fused extend path modified an inactive state-cache slot")

    if failures:
        pytest.fail("\n".join(["A/B comparisons failed:", *failures]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", *sys.argv[1:]]))
