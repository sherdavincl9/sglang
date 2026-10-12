# SPDX-License-Identifier: Apache-2.0
"""Paged Ascend KV views backed by packed NVFP4 storage."""

import logging

import torch
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

logger = logging.getLogger(__name__)


class NPUNVFP4TokenToKVPool(NPUMHATokenToKVPool):
    def _create_buffers(self):
        if self.head_dim != self.v_head_dim:
            raise ValueError(
                "NPU NVFP4 currently requires equal K and V head dimensions."
            )
        if self.use_hnd:
            raise ValueError("NPU NVFP4 requires the NHD KV layout.")
        MHATokenToKVPool._create_buffers(self)
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k_global_scale_buffer = [
                torch.zeros(
                    (self.size + self.page_size, 1, 1),
                    dtype=torch.float32,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_global_scale_buffer = [
                torch.zeros_like(buffer) for buffer in self.k_global_scale_buffer
            ]
        logger.info(
            "NPU NVFP4 KV pool: packed uint8 E2M1 data, uint8 E4M3 block scales, "
            "per-token FP32 global scales, shared %s workspace; layers=%s, slots=%s, heads=%s, head_dim=%s.",
            self.quant_method.workspace_dtype,
            self.layer_num,
            self.size + self.page_size,
            self.head_num,
            self.head_dim,
        )

    def _init_data_ptrs_and_strides(self):
        # Pointer tables are metadata only: this pool uses torch device ops,
        # not the CUDA/Triton byte-copy kernels that read device uint64 tables.
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer], dtype=torch.uint64
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer], dtype=torch.uint64
        )
        buffers = self._slot_move_pointer_buffers()
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in buffers], dtype=torch.uint64
        )
        self.data_strides = torch.tensor(
            [x.stride(0) * x.element_size() for x in buffers]
        )

    def _clear_buffers(self):
        super()._clear_buffers()
        del self.k_global_scale_buffer
        del self.v_global_scale_buffer

    def move_kv_cache(self, tgt_loc, src_loc):
        for buffers in (
            self.k_buffer,
            self.v_buffer,
            self.k_scale_buffer,
            self.v_scale_buffer,
            self.k_global_scale_buffer,
            self.v_global_scale_buffer,
        ):
            for buffer in buffers:
                buffer[tgt_loc.long()] = buffer[src_loc.long()]

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k,
        cache_v,
        k_scale=None,
        v_scale=None,
        layer_id_override=None,
        dcp_kv_mask=None,
    ):
        if dcp_kv_mask is not None:
            raise ValueError("NPU NVFP4 does not support decode context parallelism.")
        layer_id = layer.layer_id if layer_id_override is None else layer_id_override
        local_id = layer_id - self.start_layer
        from sglang.srt.mem_cache.memory_pool import unwrap_write_loc

        loc, _, _ = unwrap_write_loc(loc_info)
        self.quant_method.quantize_and_store(
            self.k_buffer[local_id],
            self.v_buffer[local_id],
            self.k_scale_buffer[local_id],
            self.v_scale_buffer[local_id],
            loc,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            k_global_scale_buffer=self.k_global_scale_buffer[local_id],
            v_global_scale_buffer=self.v_global_scale_buffer[local_id],
        )

    def _dequantized_view(self, *, layer_id, is_key):
        local_id = layer_id - self.start_layer
        data = self.k_buffer if is_key else self.v_buffer
        scales = self.k_scale_buffer if is_key else self.v_scale_buffer
        workspace = self.dq_k_buffer if is_key else self.dq_v_buffer
        global_scales = (
            self.k_global_scale_buffer if is_key else self.v_global_scale_buffer
        )
        self.quant_method.dequantize_into(
            data[local_id], scales[local_id], global_scales[local_id], workspace
        )
        if self.use_fia:
            return workspace.view(-1, 1, self.head_num, self.head_dim)
        return workspace.view(-1, self.page_size, self.head_num, self.head_dim)

    def _get_key_buffer(self, layer_id):
        return self._dequantized_view(layer_id=layer_id, is_key=True)

    def _get_value_buffer(self, layer_id):
        return self._dequantized_view(layer_id=layer_id, is_key=False)

    def get_contiguous_buf_infos(self):
        raise NotImplementedError("NPU NVFP4 KV transfer is not supported.")

    def get_kv_size_bytes(self):
        key_bytes, value_bytes = super().get_kv_size_bytes()
        if hasattr(self, "k_global_scale_buffer"):
            key_bytes += sum(buffer.nbytes for buffer in self.k_global_scale_buffer)
            value_bytes += sum(buffer.nbytes for buffer in self.v_global_scale_buffer)
        return key_bytes, value_bytes

    def get_cpu_copy(self, *args, **kwargs):
        raise NotImplementedError("NPU NVFP4 CPU offload is not supported.")

    def load_cpu_copy(self, *args, **kwargs):
        raise NotImplementedError("NPU NVFP4 CPU offload is not supported.")
