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
        self._active_slots = None
        self._workspace_layer = None
        self._preserve_current = False
        self._current_kv = None
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
        self._current_kv = None
        self._active_slots = None
        self._workspace_layer = None
        super()._clear_buffers()
        del self.k_global_scale_buffer
        del self.v_global_scale_buffer

    def move_kv_cache(self, tgt_loc, src_loc):
        self._workspace_layer = None
        self._current_kv = None
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

    def prepare_forward(
        self,
        block_tables,
        seq_lens,
        *,
        preserve_current,
        prefix_lens=None,
        prefix_lens_cpu=None,
    ):
        """Limit workspace reads to live pages of this attention batch."""
        if preserve_current and prefix_lens is not None:
            # Fresh prefill rows are copied in model precision below. Decode
            # only historical rows, including the exact part of a shared tail
            # page; never decode fresh/padded rows just to overwrite them.
            max_prefix = (
                max(prefix_lens_cpu, default=0)
                if prefix_lens_cpu is not None
                else block_tables.shape[1] * self.page_size
            )
            num_pages = (max_prefix + self.page_size - 1) // self.page_size
            if num_pages == 0:
                self._active_slots = torch.empty(
                    0, dtype=torch.int64, device=block_tables.device
                )
            else:
                positions = torch.arange(
                    num_pages * self.page_size, device=block_tables.device
                )
                physical = (
                    block_tables[:, :num_pages, None] * self.page_size
                    + positions[: self.page_size][None, None, :]
                ).flatten(1)
                self._active_slots = (
                    physical[positions[None, :] < prefix_lens[:, None]]
                    .long()
                    .unique(sorted=False)
                )
        else:
            page_indices = torch.arange(
                block_tables.shape[1], device=block_tables.device
            )
            valid = page_indices[None, :] * self.page_size < seq_lens[:, None]
            # Shared radix prefixes can occur in several requests. Decode each
            # physical page once so concurrent programs never race on workspace.
            pages = block_tables[valid].long().unique(sorted=False)
            offsets = torch.arange(self.page_size, device=block_tables.device)
            self._active_slots = (pages[:, None] * self.page_size + offsets).flatten()
        self._workspace_layer = None
        self._preserve_current = preserve_current
        self._current_kv = None

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
        self._workspace_layer = None
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
        if self._preserve_current:
            # Attention on freshly computed prefill rows uses the model dtype.
            # Only the persistent cache is quantized; cached prefixes still
            # pass through the selected BF16/FP16 dequant workspace.
            self._current_kv = (
                layer_id,
                loc.long(),
                cache_k.reshape(-1, self.head_num, self.head_dim),
                cache_v.reshape(-1, self.head_num, self.head_dim),
            )

    def _prepare_workspace(self, layer_id):
        if self._workspace_layer == layer_id:
            return
        local_id = layer_id - self.start_layer
        for data, scales, globals_, workspace in (
            (
                self.k_buffer,
                self.k_scale_buffer,
                self.k_global_scale_buffer,
                self.dq_k_buffer,
            ),
            (
                self.v_buffer,
                self.v_scale_buffer,
                self.v_global_scale_buffer,
                self.dq_v_buffer,
            ),
        ):
            if self._active_slots is None:
                self.quant_method.dequantize_into(
                    data[local_id], scales[local_id], globals_[local_id], workspace
                )
            else:
                self.quant_method.dequantize_rows_into(
                    data[local_id],
                    scales[local_id],
                    globals_[local_id],
                    self._active_slots,
                    workspace,
                )
        if self._current_kv is not None and self._current_kv[0] == layer_id:
            _, slots, key, value = self._current_kv
            self.dq_k_buffer[slots] = key.to(self.dq_k_buffer.dtype)
            self.dq_v_buffer[slots] = value.to(self.dq_v_buffer.dtype)
        self._workspace_layer = layer_id

    def _dequantized_view(self, *, layer_id, is_key):
        self._prepare_workspace(layer_id)
        workspace = self.dq_k_buffer if is_key else self.dq_v_buffer
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
