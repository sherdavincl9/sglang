"""Exercise lazy MemFabric reservation without an NPU or Offload runtime."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import torch
from sglang.srt.mem_cache.pool_host import base, common, npu_host_memory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

GIB = 1024**3


class TestMemfabricHiCacheBudget(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            hicache_size=15, hicache_ratio=3, hicache_host_memory_fraction=None
        )
        self.barrier = Mock()
        self.enterContext(
            patch.object(
                torch,
                "distributed",
                SimpleNamespace(
                    is_available=lambda: True,
                    is_initialized=lambda: True,
                    barrier=self.barrier,
                ),
                create=True,
            )
        )
        self.enterContext(
            patch.object(
                npu_host_memory,
                "get_parallel",
                return_value=SimpleNamespace(
                    world_group=SimpleNamespace(cpu_group=sentinel.cpu_world)
                ),
            )
        )
        self.enterContext(
            patch.dict(os.environ, {"SGLANG_HICACHE_HOST_MEM_BACKEND": "memfabric"})
        )
        self.enterContext(patch.object(npu_host_memory, "_backend", None))
        self.is_npu = self.enterContext(
            patch.object(npu_host_memory, "is_npu", return_value=True)
        )
        self.enterContext(
            patch.object(npu_host_memory, "get_memory", return_value=self.config)
        )
        self.enterContext(
            patch.object(
                torch, "npu", SimpleNamespace(current_device=lambda: 2), create=True
            )
        )
        self.host_budget = self.enterContext(
            patch.object(
                base, "host_memory_budget_plain_hugetlb", return_value=(24 * GIB, 0)
            )
        )
        self.ensure = self.enterContext(
            patch.object(
                npu_host_memory.MemfabricHostMemoryBackend,
                "ensure_capacity",
                autospec=True,
                side_effect=self.initialize,
            )
        )
        self.memfabric_alloc = self.enterContext(
            patch.object(
                npu_host_memory.MemfabricHostMemoryBackend,
                "alloc",
                autospec=True,
                side_effect=self.allocate_memfabric,
            )
        )

    def initialize(self, backend, total_bytes, device_id):
        backend._initialized = True

    def allocate_memfabric(self, backend, *args, **kwargs):
        self.assertTrue(backend._initialized)
        return sentinel.memfabric_buffer

    def allocate_npu(self, dims=(2,)):
        return common.alloc_with_pin_memory_npu(dims, torch.uint8, "cpu", True, None)

    def test_standalone_scope_without_allocation_does_not_initialize(self):
        with npu_host_memory.memfabric_host_memory_budget(object()):
            self.assertIsNone(npu_host_memory._backend)
        self.assertIsNone(npu_host_memory._backend)
        self.ensure.assert_not_called()

    def test_fixed_size_reserves_on_first_npu_allocation_without_device_budget(self):
        def build(label):
            self.assertEqual(label, "mla")
            self.assertIsNone(npu_host_memory._backend)
            self.ensure.assert_not_called()
            self.assertIs(self.allocate_npu(), sentinel.memfabric_buffer)
            return self.allocate_npu()

        result = npu_host_memory.build_memfabric_kv_host_pool(object(), build, "mla")

        self.assertIs(result, sentinel.memfabric_buffer)
        self.ensure.assert_called_once_with(npu_host_memory._backend, 15 * GIB, 2)
        self.assertEqual(self.memfabric_alloc.call_count, 2)

    def test_ratio_caps_to_preflight_snapshot_shared_across_pools(self):
        self.config.hicache_size = 0
        self.config.hicache_ratio = 2
        self.host_budget.return_value = (9 * GIB + GIB // 2, 0)
        pool = SimpleNamespace(memfabric_device_budget_bytes=5 * GIB)

        with npu_host_memory.memfabric_host_memory_budget(pool):
            first = base.host_memory_preflight([GIB], device="npu")
            self.assertEqual(first.available_bytes, 9 * GIB + GIB // 2)
            self.host_budget.return_value = (GIB, 0)
            with npu_host_memory.memfabric_host_memory_budget(object()):
                self.allocate_npu()
            second = base.host_memory_preflight([GIB], device="npu")
            self.assertEqual(second.available_bytes, 8 * GIB + GIB // 2)
            self.allocate_npu()

        self.ensure.assert_called_once_with(npu_host_memory._backend, 9 * GIB, 2)
        self.host_budget.assert_called_once()

    def test_ratio_uses_pre_split_device_budget(self):
        self.config.hicache_size = 0
        self.config.hicache_ratio = 2
        pool = SimpleNamespace(memfabric_device_budget_bytes=GIB)

        with npu_host_memory.memfabric_host_memory_budget(pool):
            self.allocate_npu()

        self.ensure.assert_called_once_with(npu_host_memory._backend, 3 * GIB, 2)

    def test_ratio_requires_pre_split_device_budget_on_allocation(self):
        self.config.hicache_size = 0
        with (
            npu_host_memory.memfabric_host_memory_budget(object()),
            self.assertRaisesRegex(ValueError, "pre-split device memory budget"),
        ):
            self.allocate_npu()

        self.ensure.assert_not_called()
        self.memfabric_alloc.assert_not_called()

    def test_insufficient_aligned_host_budget_fails_before_allocation(self):
        self.config.hicache_size = 2
        self.host_budget.return_value = (2_000_000_001, 0)
        with (
            npu_host_memory.memfabric_host_memory_budget(object()),
            self.assertRaisesRegex(ValueError, "Not enough host memory"),
        ):
            self.allocate_npu()

        self.ensure.assert_not_called()
        self.memfabric_alloc.assert_not_called()

    def test_gpu_dispatch_does_not_prepare_memfabric(self):
        self.is_npu.return_value = False
        allocator = Mock()
        with npu_host_memory.memfabric_host_memory_budget(object()):
            result = common.alloc_with_host_register(
                (2,), torch.uint8, "cpu", False, allocator
            )

        self.assertIs(result, allocator.allocate.return_value)
        self.assertIsNone(npu_host_memory._backend)
        self.host_budget.assert_not_called()
        self.ensure.assert_not_called()

    def test_acl_dispatch_does_not_prepare_memfabric(self):
        with (
            patch.dict(os.environ, {"SGLANG_HICACHE_HOST_MEM_BACKEND": "acl"}),
            patch.object(
                npu_host_memory.AclHostMemoryBackend,
                "alloc",
                return_value=sentinel.acl_buffer,
            ) as acl_alloc,
            npu_host_memory.memfabric_host_memory_budget(object()),
        ):
            self.assertIs(self.allocate_npu(), sentinel.acl_buffer)

        acl_alloc.assert_called_once_with((2,), torch.uint8, "cpu", True, None)
        self.host_budget.assert_not_called()
        self.ensure.assert_not_called()

    def test_explicit_scope_samples_before_one_world_barrier(self):
        calls = Mock()
        calls.attach_mock(self.host_budget, "sample")
        calls.attach_mock(self.barrier, "barrier")
        with (
            npu_host_memory.memfabric_host_memory_budget(object(), sync_snapshot=True),
            npu_host_memory.memfabric_host_memory_budget(object(), sync_snapshot=True),
        ):
            self.assertEqual(
                [call[0] for call in calls.mock_calls], ["sample", "barrier"]
            )
            self.ensure.assert_not_called()
            self.allocate_npu()

        self.barrier.assert_called_once_with(group=sentinel.cpu_world)
        self.host_budget.assert_called_once()

    def test_auto_size_scope_does_not_synchronize_again(self):
        self.config.hicache_host_memory_fraction = 0.8
        with (
            base.host_memory_budget_scope(20 * GIB),
            npu_host_memory.memfabric_host_memory_budget(object(), sync_snapshot=True),
        ):
            self.allocate_npu()

        self.barrier.assert_not_called()
        self.host_budget.assert_not_called()

    def test_empty_block_does_not_reserve_or_allocate_from_memfabric(self):
        with (
            patch.object(torch, "empty", return_value=sentinel.empty_buffer) as empty,
            npu_host_memory.memfabric_host_memory_budget(object()),
        ):
            self.assertIs(self.allocate_npu((2, 0)), sentinel.empty_buffer)

        empty.assert_called_once_with((2, 0), dtype=torch.uint8, device="cpu")
        self.ensure.assert_not_called()
        self.memfabric_alloc.assert_not_called()

    def test_first_block_raises_the_reservation_above_the_ratio_estimate(self):
        self.config.hicache_size = 0
        self.config.hicache_ratio = 1
        pool = SimpleNamespace(memfabric_device_budget_bytes=GIB)

        with npu_host_memory.memfabric_host_memory_budget(pool):
            self.allocate_npu((3 * GIB + 1,))

        self.ensure.assert_called_once_with(npu_host_memory._backend, 4 * GIB, 2)

    def test_first_block_cannot_exceed_the_aligned_host_budget(self):
        self.config.hicache_size = 0
        self.config.hicache_ratio = 1
        self.host_budget.return_value = (3 * GIB + GIB // 2, 0)
        pool = SimpleNamespace(memfabric_device_budget_bytes=GIB)

        with (
            npu_host_memory.memfabric_host_memory_budget(pool),
            self.assertRaisesRegex(ValueError, "Not enough host memory"),
        ):
            self.allocate_npu((3 * GIB + 1,))

        self.ensure.assert_not_called()
        self.memfabric_alloc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
