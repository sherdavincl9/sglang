"""Single-NPU PLE pinned/file lookups against a CPU embedding reference.

Run: python3 test/manual/ascend/test_npu_ple_offload.py -v
No model weights, UVA configuration, or custom host-mapping operators needed.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
from torch import nn

from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding
from sglang.srt.models.qwen4_exp_ple_table import PLE_FILE_PREFETCH_MIN_ROWS


class TestNpuPleOffload(unittest.TestCase):
    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def make_embedding(self, backend, dtype, dim, start=0, end=8):
        # The real constructor builds the table on meta before offloading.
        weight = nn.Parameter(
            torch.empty((end - start, dim), dtype=dtype, device="meta"),
            requires_grad=False,
        )
        weight.output_dim = 0
        shard = VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index=start,
            padded_org_vocab_end_index=end,
            padded_added_vocab_start_index=8,
            padded_added_vocab_end_index=8,
            org_vocab_start_index=start,
            org_vocab_end_index=end,
            added_vocab_start_index=8,
            added_vocab_end_index=8,
        )
        source = SimpleNamespace(
            weight=weight,
            quant_config=None,
            enable_tp=True,
            use_attn_tp_group=False,
            tp_size=1,
            num_embeddings=8,
            org_vocab_size=8,
            padding_size=1,
            num_added_embeddings=0,
            use_presharded_weights=False,
            org_vocab_size_padded=8,
            num_embeddings_padded=8,
            shard_indices=shard,
            embedding_dim=dim,
            weight_scale=torch.tensor([0.5], device=self.device),
            quant_method=UnquantizedEmbeddingMethod(),
            num_embeddings_per_partition=end - start,
            num_org_embeddings_per_partition=end - start,
            num_added_embeddings_per_partition=0,
        )
        table_dir = self.directory / f"{backend}_{dtype}_{dim}_{start}_{end}"
        embedding = Qwen4ExpPinnedHostEmbedding(
            source, backend=backend, table_dir=str(table_dir)
        )
        # Keep the mapped storage alive until its background workers stop.
        self.addCleanup(self.close_embedding, embedding)
        rows = (
            ((torch.arange(8 * dim, dtype=torch.float32) % 31 - 15) / 4)
            .reshape(8, dim)
            .to(dtype)
        )
        ptr = embedding.weight.data_ptr()
        embedding.weight_loader(embedding.weight, rows)
        self.assertEqual(embedding.weight.data_ptr(), ptr)
        self.assertEqual(embedding.weight.device.type, "cpu")
        self.assertEqual(embedding.weight.dtype, dtype)
        self.assertEqual(embedding.weight.is_pinned(), backend == "pinned")
        self.assertIs(embedding.weight_scale, source.weight_scale)
        return embedding, rows, table_dir

    @staticmethod
    def close_embedding(embedding):
        for worker in (embedding._file_prefetcher, embedding._file_rss_trimmer):
            if worker is not None:
                worker.close()

    def test_pinned_and_file_gather(self):
        for backend in ("pinned", "file"):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                for dim in (7, 64, 257):
                    for id_dtype in (torch.int32, torch.int64):
                        with self.subTest(
                            backend=backend, dtype=dtype, dim=dim, ids=id_dtype
                        ):
                            embedding, rows, _ = self.make_embedding(
                                backend, dtype, dim
                            )
                            ids = torch.tensor(
                                [[7, 0, 3], [3, 1, 6]],
                                dtype=id_dtype,
                                device=self.device,
                            )
                            actual = embedding(ids)
                            expected = rows.to(torch.bfloat16)[ids.cpu().long()]
                            torch.testing.assert_close(
                                actual.cpu(), expected, rtol=0, atol=0
                            )

    def test_shard_boundaries_duplicates_and_output_buffer(self):
        # Simulate two disjoint TP shards without an HCCL process group.
        ids = torch.tensor([[-1, 0, 3], [4, 7, 8], [7, 4, 99]], device=self.device)
        for backend in ("pinned", "file"):
            with self.subTest(backend=backend):
                combined = torch.zeros((*ids.shape, 13), dtype=torch.bfloat16)
                for start, end in ((0, 4), (4, 8)):
                    embedding, rows, _ = self.make_embedding(
                        backend, torch.bfloat16, 13, start, end
                    )
                    out = torch.full(
                        (*ids.shape, 13),
                        float("nan"),
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    actual = embedding.gather(ids, out=out)
                    self.assertIs(actual, out)
                    expected = torch.zeros_like(combined)
                    ids_cpu = ids.cpu()
                    valid = (ids_cpu >= start) & (ids_cpu < end)
                    expected[valid] = rows[ids_cpu[valid]]
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                    combined += actual.cpu()
                expected = torch.zeros_like(combined)
                valid = (ids_cpu >= 0) & (ids_cpu < 8)
                expected[valid] = rows[ids_cpu[valid]]
                torch.testing.assert_close(combined, expected, rtol=0, atol=0)

    def test_file_directory_persistence_and_updated_rows(self):
        embedding, rows, directory = self.make_embedding("file", torch.bfloat16, 13)
        files = list(directory.iterdir())
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].stat().st_size, rows.numel() * rows.element_size())
        reopened = torch.from_file(
            str(files[0]), shared=True, size=rows.numel(), dtype=rows.dtype
        ).reshape_as(rows)
        torch.testing.assert_close(reopened, rows, rtol=0, atol=0)
        ids = torch.tensor([2, 5, 2], device=self.device)
        first = embedding.gather(ids).cpu()
        reopened[2].fill_(42)
        second = embedding.gather(ids).cpu()
        self.assertFalse(torch.equal(first[0], second[0]))
        torch.testing.assert_close(second, reopened[[2, 5, 2]], rtol=0, atol=0)

    def test_file_prefetch_hint_from_host_ids(self):
        """A prefill-sized NPU gather must still hint the file's page cache."""
        # bf16 rows of 4096 = two 4 KiB pages; shard rows 4..7 are local rows 0..3.
        embedding, rows, _ = self.make_embedding("file", torch.bfloat16, 4096, 4, 8)
        prefetcher = embedding._file_prefetcher
        self.assertIsNotNone(prefetcher, "SGLANG_QWEN4_PLE_FILE_PREFETCH is off")
        ids = torch.tensor([1, 5, 7, 5], device=self.device).repeat(
            PLE_FILE_PREFETCH_MIN_ROWS // 4
        )
        with (
            patch.object(prefetcher, "enqueue", wraps=prefetcher.enqueue) as enqueue,
            patch.object(
                prefetcher._pool, "submit", wraps=prefetcher._pool.submit
            ) as submit,
        ):
            actual = embedding.gather(ids)
        self.assertEqual(enqueue.call_args.args[0].device.type, "cpu")
        submit.assert_called_once()
        # Global ids 5 and 7 are local rows 1 and 3; id 1 belongs to another shard.
        self.assertEqual(submit.call_args.args[1], [2, 3, 6, 7])
        ids_cpu = ids.cpu()
        valid = (ids_cpu >= 4) & (ids_cpu < 8)
        expected = torch.zeros((ids.numel(), 4096), dtype=torch.bfloat16)
        expected[valid] = rows[ids_cpu[valid]]
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_side_stream_copy_and_repeated_lookup(self):
        for backend in ("pinned", "file"):
            with self.subTest(backend=backend):
                embedding, rows, _ = self.make_embedding(
                    backend, torch.bfloat16, 257
                )
                stream = torch.npu.Stream(device=self.device)
                for offset in range(4):
                    ids = (torch.arange(128, device=self.device) + offset) % 8
                    stream.wait_stream(torch.npu.current_stream())
                    ids.record_stream(stream)
                    with torch.npu.stream(stream):
                        actual = embedding.gather(ids)
                    torch.npu.current_stream().wait_stream(stream)
                    expected = rows[ids.cpu()]
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_empty_input(self):
        for backend in ("pinned", "file"):
            with self.subTest(backend=backend):
                embedding, _, _ = self.make_embedding(backend, torch.bfloat16, 7)
                actual = embedding.gather(
                    torch.empty((0, 3), dtype=torch.int64, device=self.device)
                )
                self.assertEqual(actual.shape, (0, 3, 7))

    def test_capture_rejected_before_host_row_selection(self):
        embedding, _, _ = self.make_embedding("file", torch.bfloat16, 7)
        ids = torch.tensor([1], device=self.device)
        with patch(
            "sglang.srt.models.qwen4_exp.get_is_capture_mode", return_value=True
        ):
            with self.assertRaisesRegex(RuntimeError, "requires eager execution"):
                embedding.gather(ids)


if __name__ == "__main__":
    unittest.main()
