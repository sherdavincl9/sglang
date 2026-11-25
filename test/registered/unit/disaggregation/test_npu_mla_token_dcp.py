import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sglang.srt.disaggregation.base.conn import NpuDcpLayout
from sglang.srt.disaggregation.mooncake.buffers import BufferKeys
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _kv_args(layer_ids, page_lens, replicated, tiles, num_draft, *, decode=False):
    return SimpleNamespace(
        page_size=4,
        engine_rank=2 if decode else 0,
        kv_data_ptrs=[4096 * (i + 1) for i in range(len(layer_ids))],
        kv_item_lens=page_lens,
        kv_layer_ids=layer_ids if decode else [],
        aux_data_ptrs=[],
        state_data_ptrs=[],
        state_item_lens=[],
        state_layer_ids=[],
        draft_rows_dcp_collapsed=False,
        num_draft_entries=num_draft,
        npu_dcp_layout=NpuDcpLayout(4, layer_ids, replicated, tiles),
    )


def _make_peers(*, draft=True, nz=False):
    src = object.__new__(MooncakeKVManager)
    src.dcp_size = 1
    src.dcp_rank = 0
    src.pp_size = 2
    src.is_mla_backend = True
    src.is_hybrid_mla_backend = False
    src.max_transfer_batch_indices = 2
    src._transfer_copies = Mock(return_value=0)

    # Prefill owns only layer 11. Latent, RoPE and index-K repeat its ID.
    tile = 32 if nz else 0
    num_draft = 2 if draft else 0
    src.kv_args = _kv_args(
        [11] * 3 + [78] * num_draft,
        [256, 128, 48] + [256, 128][:num_draft],
        [False, False, True] + [True] * num_draft,
        [tile, tile, 0] + [tile] * num_draft,
        num_draft,
    )

    dst = object.__new__(MooncakeKVManager)
    dst.get_session_id = lambda: "decode"
    dst.local_ip = "127.0.0.1"
    dst.rank_port = 12345
    dst.attn_tp_size = dst.dcp_size = 4
    dst.dcp_rank = 2
    dst.enable_staging = dst.exchange_aux_item_lens = False
    dst.decode_enable_radix_cache = False
    dst.kv_args = _kv_args(
        [9, 11] * 3 + [78, 78],
        [256, 256, 128, 128, 192, 192, 1024, 512],
        [False] * 4 + [True] * 4,
        [tile] * 4 + [0, 0, tile, tile],
        2,
        decode=True,
    )
    src._local_keys = BufferKeys.for_buffers(src.kv_args.kv_data_ptrs, [], [])
    dst_keys = BufferKeys.for_buffers(dst.kv_args.kv_data_ptrs, [], [])
    src._peers = {"decode": SimpleNamespace(keys=dst_keys)}
    return src, dst


def _send_and_get_ranges(manager, peer):
    result = manager.send_kvcache_dcp(
        "decode",
        np.array([5, 6, 7, 8, 9], dtype=np.int32),
        peer.dst_kv_ptrs,
        np.array([10, 12], dtype=np.int32),
        dcp_token_item_lens=None,
        dst_dcp_rank=peer.dst_dcp_rank,
        dst_dcp_size=peer.dst_dcp_size,
        dst_layer_ids=peer.dst_kv_layer_ids,
        dst_kv_item_lens=peer.dst_kv_item_lens,
        executor=None,
        npu_dcp_transfer_entries=peer.npu_dcp_transfer_entries,
        src_page_offset=1,
        decode_prefix_len=16,
        num_kv_tokens=18,
    )
    assert result == 0
    ranges = defaultdict(list)
    for call in manager._transfer_copies.call_args_list:
        session, copies = call.args
        assert session == "decode"
        for copy in copies:
            assert copy.ranges.dtype == np.uint64
            ranges[copy.src, copy.dst].extend(copy.ranges.tolist())
    return ranges


def _byte_offset(row, token_width, tile_width, tile):
    page, slot = divmod(row, 4)
    return page * 4 * token_width + (tile * 4 + slot) * tile_width


class TestNpuMlaTokenDcp(unittest.TestCase):
    def test_pp_registration_and_nd_nz_token_copy(self):
        for nz, draft in ((False, False), (False, True), (True, True)):
            with self.subTest(nz=nz, draft=draft):
                src, dst = _make_peers(nz=nz, draft=draft)
                if not draft:
                    src.kv_args.kv_item_lens[1] = dst.kv_args.kv_item_lens[3] = 0
                frames = dst.build_receiver_registration_message()
                peer = src._prepare_decode_registration("decode", frames)
                self.assertTrue(peer.requires_dcp_relayout)
                tile = 32 if nz else 0
                entries = [(0, 1, 64, False, tile), (2, 5, 12, True, 0)]
                if draft:
                    entries += [
                        (1, 3, 32, False, tile),
                        (3, 6, 64, True, tile),
                        (4, 7, 32, True, tile),
                    ]

                target_rows = [(22, 41), (26, 42), (30, 43), (34, 48)]
                global_dst = [*range(164, 176), *range(192, 198)]
                global_rows = list(zip(range(20, 38), global_dst))
                ranges = _send_and_get_ranges(src, peer)
                self.assertEqual(len(ranges), len(entries))
                for i, j, width, replicated, tile_bytes in entries:
                    key = src._local_keys.kv[i], src._peers["decode"].keys.kv[j]
                    rows = global_rows if replicated else target_rows
                    chunk = tile_bytes or width

                    expected = [
                        (
                            _byte_offset(a, width, chunk, t),
                            _byte_offset(b, width, chunk, t),
                        )
                        for a, b in rows
                        for t in range(width // chunk)
                    ]
                    actual = []
                    for a, b, n in ranges[key]:
                        self.assertEqual(n % chunk, 0)
                        actual.extend((a + k, b + k) for k in range(0, n, chunk))
                    self.assertCountEqual(actual, expected)

    def test_legacy_and_truncated_registration(self):
        src, dst = _make_peers()
        frames = dst.build_receiver_registration_message()
        with self.assertLogs(MooncakeKVManager.__module__, level="ERROR"):
            self.assertIsNone(src._prepare_decode_registration("decode", frames[:-1]))
        dst.kv_args.npu_dcp_layout = None
        legacy = dst.build_receiver_registration_message()
        self.assertEqual(legacy, frames[:-2])
        self.assertEqual(legacy[-1], b"0")
        with self.assertRaisesRegex(RuntimeError, "matching NPU token KV layouts"):
            src._prepare_decode_registration("decode", legacy)
        src.kv_args.npu_dcp_layout = None
        with self.assertRaisesRegex(RuntimeError, "matching NPU token KV layouts"):
            src._prepare_decode_registration("decode", frames)

    def test_registration_rejects_mismatched_widths_before_copy(self):
        for entry in (3, 5, 6):
            with self.subTest(entry=entry):
                src, dst = _make_peers()
                dst.kv_args.kv_item_lens[entry] += 4
                with self.assertRaisesRegex(RuntimeError, "page stride differs"):
                    src._prepare_decode_registration(
                        "decode", dst.build_receiver_registration_message()
                    )
                src._transfer_copies.assert_not_called()


if __name__ == "__main__":
    unittest.main()
