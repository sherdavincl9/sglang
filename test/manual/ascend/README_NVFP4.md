# Software NVFP4 KV cache on Ascend NPU

Use `--kv-cache-dtype nvfp4 --prefill-kv-cache-dequant-dtype auto` with a
BF16/FP16 dense MHA model. `auto` selects the model compute dtype. Explicit
`bf16` and `fp16` must match `--dtype bfloat16` and `--dtype float16` respectively.
Native `nvfp4` and `fp8_e4m3` attention inputs remain CUDA-only.

The NPU pool stores packed E2M1 pairs in uint8, one uint8 E4M3 scale per 16
values, and an online FP32 global scale for each token's K and V tensors.
Per-token global scales keep small activations in the normal E4M3 range without
a calibrated checkpoint. This is an NPU-specific physical scale layout.
Prefill and decode both read a shared BF16/FP16 workspace; persistent full
precision copies are not retained per layer. KV data stays on the device.

For `L` layers, `H` KV heads and equal K/V head dimensions `D`, persistent
storage per token is `2 * L * H * (D/2 + D/16) + 8 * L` bytes. The shared
workspace adds `4 * H * D` bytes per token. Allocation sizing includes both.
Dequantization uses chunks of 256 slots to bound temporary FP32 scratch.

This implementation prioritizes functionality and memory reduction. Each
attention read dequantizes the whole layer cache, so latency grows with pool
capacity. Bound `--max-total-tokens` for initial validation. Graph capture and
overlap scheduling are disabled automatically. MLA, hybrid linear/SWA pools,
speculative decoding, CP, KV sharding, KV offload/transfer and unified/page-major
pools are not supported by this initial implementation. Their incompatible
configurations are rejected; the NPU software format is not transferred to
CUDA backends.

From this checkout, inside an existing NPU environment:

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=4  # select a free device on your machine
python test/manual/ascend/test_npu_nvfp4_kv_cache.py
python test/manual/ascend/test_npu_nvfp4_args_budget.py
python test/manual/ascend/validate_npu_nvfp4_numerics.py
python test/manual/ascend/validate_npu_nvfp4.py \
    --model /home/weights/Qwen3-0.6B \
    --output-dir /home/c00985000/npu-prefill-kv-dequant-20261012/logs/e2e
```

The inference script runs BF16/FP16 baselines and NVFP4 auto/BF16/FP16. It checks
multi-token decoding, batched requests, chunked prefill and cached-prefix
extension. It retains commands, server logs, responses and the input-logprob
comparison summary. The 0.25 mean absolute logprob error bound is a smoke-test
criterion for the selected prompt, not a model accuracy benchmark. Quantized
inference can change greedy output tokens.
