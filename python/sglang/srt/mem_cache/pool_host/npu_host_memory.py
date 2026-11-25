"""NPU host DRAM mapped into device VA space for the AIV IO path.

torch pin_memory buffers are only reachable by the SDMA engine; AIV kernels
(e.g. kv_exchange) can only de-reference host VAs that were mapped into the
device VA space first.  SGLANG_HICACHE_HOST_MEM_BACKEND selects the backend
(a NpuHostMemoryBackend subclass resolved by get_host_memory_backend()):

- ``memfabric`` (MemfabricHostMemoryBackend): a MemFabric ``Offload``
  instance owns the host pool; ``pool.initialize`` reserves and
  pre-registers the whole DRAM range and ``pool.empty`` hands out blocks.
  The HiCache host pool is allocated through the Offload instance and the
  L2<->L1 IO uses the AIV kv_exchange kernel.
- ``acl`` (AclHostMemoryBackend): standard CANN runtime registration
  (aclrtHostRegister with ACL_HOST_REGISTER_MAPPED, see the kernel-side
  ``host_memory`` module).  Host blocks come from the shared mmap
  allocator (alloc_mmap), so SGLANG_HUGEPAGE_SIZE (2MB/1GB hugetlb
  pages) applies to the host pool; blocks are registered so the AIV
  kv_exchange kernel can resolve their device pointers. The kernel-side
  auto routing sends every transfer to the AIV operator.

ascendc_io_enabled() gates the AIV IO path and is true for both backends.
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mmap import alloc_mmap
from sglang.srt.runtime_context import get_memory, get_parallel
from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)

_MEMFABRIC_GB = 1024**3
_MEMFABRIC_RESERVE_SLACK = 0.05


def host_memory_backend() -> str | None:
    """Return 'memfabric' | 'acl' | None from SGLANG_HICACHE_HOST_MEM_BACKEND."""
    return os.environ.get("SGLANG_HICACHE_HOST_MEM_BACKEND", "").lower() or None


def memfabric_host_memory_enabled() -> bool:
    """MemFabric offload entity backs the host pool (offload.empty blocks)."""
    return host_memory_backend() == "memfabric"


def acl_host_memory_enabled() -> bool:
    """aclrtHostRegister backs the host pool (mmap-backed host blocks)."""
    return host_memory_backend() == "acl"


@lru_cache(maxsize=1)
def _log_acl_mmap_host_pool() -> None:
    configured = (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip().upper()
    pages = configured if configured in ("2MB", "1GB") else "4KB"
    logger.info(
        "[HiCache] acl host pool via mmap allocator, page size %s "
        "(hugetlb blocks fall back to 4KB on failure)",
        pages,
    )


def ascendc_io_enabled() -> bool:
    """Use the AIV kv_exchange kernel for HiCache L2<->L1 IO.

    Both registered-host-memory backends map host DRAM into the device VA
    space, which is what the AIV kernel de-references, so either one
    enables this IO path.
    """
    return host_memory_backend() in ("memfabric", "acl")


class NpuHostMemoryBackend:
    """Base class for host-DRAM backends mapped into the device VA space.

    A backend hands out host blocks that the AIV kv_exchange kernel can
    de-reference directly.  Host pools declare their combined block budget
    with ensure_capacity() before the first alloc(); MemFabric needs the
    declaration (the entity reservation is fixed by the first one) while
    the aclrt path registers each block at allocation time and keeps the
    inherited no-op.
    """

    def ensure_capacity(self, total_bytes: int, device_id: int) -> None:
        """Declare the combined host-pool budget before allocation."""

    def alloc(
        self,
        dims: tuple,
        dtype: torch.dtype,
        device: str,
        pin_memory: bool,
        allocator: None,
    ) -> torch.Tensor:
        """Allocate one host block visible to the AIV kv_exchange kernel."""
        raise NotImplementedError


class MemfabricHostMemoryBackend(NpuHostMemoryBackend):
    """Host blocks from a MemFabric Offload instance (pool.empty).

    One ``memfabric_hybrid.offload.Offload`` instance owns the host pool:
    ``pool.initialize`` reserves and pre-registers the whole DRAM range, so
    every ``pool.malloc``/``pool.empty`` block is device-visible immediately
    (16-byte-bump allocation, no per-block GB rounding), and
    ``pool.host_register`` on a block is an idempotent in-pool check.  The
    instance is published via the kernel-side ``set_memfabric_pool``
    so its register/get-device-pointer calls resolve through the same pool.
    """

    def __init__(self):
        self._pool = None
        self._initialized = False
        self._reserved_bytes = 0
        self._allocated_bytes = 0
        self._device_id = None

    def _get_offload(self):
        if self._pool is None:
            try:
                from memfabric_hybrid import offload
            except ImportError as exc:
                raise ImportError(
                    "SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric requires "
                    "the memfabric_hybrid package (provides the Offload "
                    "host-memory pool). Install it or unset the env var."
                ) from exc
            self._pool = offload.Offload()
            self._offload_module = offload
        return self._offload_module

    def _register_block(self, tensor: torch.Tensor) -> None:
        """Enter an MF-allocated block into the kernel DVA table.

        The pool itself is already device-mapped at initialize time; this
        registration makes the block (and its interior views) resolvable by
        the AIV kv_exchange kernel through the host_memory table. Registration
        happens once, at allocation time; interior views resolve by byte
        offset at transfer time.
        """
        try:
            from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
                register_host_memory,
            )
        except ImportError as exc:
            raise ImportError(
                "SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric requires sgl-kernel-npu "
                "with host_memory.register_host_memory (AIV kv_exchange needs "
                "registered MF blocks)"
            ) from exc
        register_host_memory(tensor)

    def ensure_capacity(self, total_bytes: int, device_id: int) -> None:
        """Initialize the host pool, sized by total_bytes.

        total_bytes is the combined byte size of all buffers the calling
        host pool is about to allocate (ultimately derived from
        --hicache-size / --hicache-ratio).  The entity is sized by the first
        declaration; later host pools in the same process must fit into
        what is left.
        """
        if not self._initialized:
            try:
                from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
                    set_memfabric_pool,
                )
            except ImportError as exc:
                raise ImportError(
                    "SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric requires sgl-kernel-npu "
                    "with host_memory.set_memfabric_pool"
                ) from exc

            offload = self._get_offload()
            config = offload.OffloadConfig()
            config.device_id = device_id
            config.reserve_size = total_bytes
            # LOCAL scene requires alloc_size == reserve_size.
            config.alloc_size = total_bytes
            config.scene = offload.Scene.LOCAL
            config.executor = offload.Executor.AICPU
            ret = self._pool.initialize(config)
            if ret != 0:
                self._pool = None
                raise RuntimeError(
                    f"Offload.initialize failed: ret={ret}, "
                    f"device_id={device_id}, reserve_size={total_bytes}"
                )
            set_memfabric_pool(self._pool)
            self._initialized = True
            self._reserved_bytes = total_bytes
            self._device_id = device_id
            logger.info(
                "[HiCache] memfabric host pool initialized (Offload entry): "
                "reserve=%.2fGB device=%d (exact combined budget)",
                total_bytes / _MEMFABRIC_GB,
                device_id,
            )
        elif self._device_id != device_id:
            raise RuntimeError(
                f"memfabric host pool is already initialized for device "
                f"{self._device_id}, got device {device_id}"
            )
        remaining = self._reserved_bytes - self._allocated_bytes
        if total_bytes > remaining:
            raise RuntimeError(
                f"memfabric host memory exhausted: need "
                f"{total_bytes / _MEMFABRIC_GB:.2f}GB, "
                f"only {remaining / _MEMFABRIC_GB:.2f}GB left of the "
                f"{self._reserved_bytes / _MEMFABRIC_GB:.2f}GB reserve "
                "(sized automatically from the first L2 host pool, i.e. from "
                "--hicache-size / --hicache-ratio). All host pools of the "
                "process share this reserve."
            )

    def alloc(
        self,
        dims: tuple,
        dtype: torch.dtype,
        device: str,
        pin_memory: bool,
        allocator: None,
    ) -> torch.Tensor:
        """Allocate one host-pool block and enter it into the DVA table.

        The Python-side accounting tracks the exact block byte size.
        """
        if not self._initialized:
            raise RuntimeError("MemFabric ensure_capacity must run before alloc")
        nbytes = math.prod(dims) * dtype.itemsize
        remaining = self._reserved_bytes - self._allocated_bytes
        if nbytes > remaining:
            raise RuntimeError(
                f"memfabric host memory exhausted: block needs "
                f"{nbytes / _MEMFABRIC_GB:.2f}GB, "
                f"only {remaining / _MEMFABRIC_GB:.2f}GB left of the "
                f"{self._reserved_bytes / _MEMFABRIC_GB:.2f}GB reserve"
            )
        tensor = self._pool.empty(list(dims), dtype=dtype)
        self._register_block(tensor)
        self._allocated_bytes += nbytes
        return tensor


class AclHostMemoryBackend(NpuHostMemoryBackend):
    """mmap-backed host blocks registered with aclrtHostRegister.

    Standard CANN runtime registration: host blocks allocated by the
    shared mmap allocator (alloc_mmap, honoring SGLANG_HUGEPAGE_SIZE
    hugetlb pages with a prefaulted plain-mmap fallback) are mapped into
    the device VA space by aclrtHostRegister(ACL_HOST_REGISTER_MAPPED)
    inside the kernel-side ``host_memory`` module.  There is no
    pre-reserved entity and nothing to declare up front, so ensure_capacity
    stays the inherited no-op; the kernel-side host_memory table resolves
    interior views by byte offset at transfer time.
    """

    def _register_block(self, tensor: torch.Tensor) -> None:
        try:
            from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
                register_acl_host_memory,
            )
        except ImportError as exc:
            raise ImportError(
                "SGLANG_HICACHE_HOST_MEM_BACKEND=acl requires sgl-kernel-npu "
                "with host_memory.register_acl_host_memory (AIV kv_exchange "
                "needs aclrt-registered host blocks)"
            ) from exc
        register_acl_host_memory(tensor)

    def alloc(
        self,
        dims: tuple,
        dtype: torch.dtype,
        device: str,
        pin_memory: bool,
        allocator: None,
    ) -> torch.Tensor:
        """Allocate one mmap-backed host block and map it for NPU access.

        Memory comes from alloc_mmap, so SGLANG_HUGEPAGE_SIZE selects
        2MB/1GB hugetlb pages (prefaulted 4KB mmap when unset or on
        allocation failure); the registration path is identical for
        every page size.
        """
        _log_acl_mmap_host_pool()
        tensor = alloc_mmap(tuple(dims), dtype)
        self._register_block(tensor)
        return tensor


_backend: NpuHostMemoryBackend | None = None
_memfabric_hicache_budget: ContextVar[tuple[object, int] | None] = ContextVar(
    "memfabric_hicache_budget", default=None
)


def get_host_memory_backend() -> NpuHostMemoryBackend | None:
    """Return the backend selected by SGLANG_HICACHE_HOST_MEM_BACKEND.

    Returns None when no registered-host-memory backend is configured;
    callers then fall back to plain pinned memory.  The instance is cached,
    so the selection and the MemFabric entity state are process-wide
    singletons.
    """
    global _backend
    if _backend is None:
        if memfabric_host_memory_enabled():
            _backend = MemfabricHostMemoryBackend()
        elif acl_host_memory_enabled():
            _backend = AclHostMemoryBackend()
    return _backend


@contextmanager
def memfabric_host_memory_budget(
    kv_pool: object, *, enabled: bool = True, sync_snapshot: bool = False
):
    """Save the host budget before the first pool books memory."""
    if (
        not enabled
        or not is_npu()
        or not memfabric_host_memory_enabled()
        or _memfabric_hicache_budget.get() is not None
    ):
        yield
        return

    from sglang.srt.mem_cache.pool_host.base import (
        host_memory_budget_scope,
        host_memory_preflight,
    )

    host_bytes = host_memory_preflight(book=False).available_bytes
    # The auto-size path already synchronizes ranks after sampling host memory.
    if (
        sync_snapshot
        and get_memory().hicache_host_memory_fraction is None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.barrier(group=get_parallel().world_group.cpu_group)
    token = _memfabric_hicache_budget.set((kv_pool, host_bytes))
    try:
        # Later pools continue booking from this snapshot after MF claims DRAM.
        with host_memory_budget_scope(host_bytes):
            yield
    finally:
        _memfabric_hicache_budget.reset(token)


def build_memfabric_kv_host_pool(kv_pool, build, /, *args, **kwargs):
    """Prepare the same budget for a standalone decode host pool."""
    with memfabric_host_memory_budget(kv_pool):
        return build(*args, **kwargs)


def reserve_memfabric_hicache(
    backend: MemfabricHostMemoryBackend, dims: tuple, dtype: torch.dtype
) -> None:
    """Initialize MemFabric on its first nonempty host allocation."""
    if backend._initialized:
        return
    budget = _memfabric_hicache_budget.get()
    if budget is None:
        raise RuntimeError(
            "MemFabric HiCache allocation requires a host memory budget."
        )
    kv_pool, host_bytes = budget
    memory = get_memory()
    fixed_size = memory.hicache_size > 0
    if fixed_size:
        nominal_bytes = memory.hicache_size * 1_000_000_000
    else:
        device_budget_bytes = getattr(kv_pool, "memfabric_device_budget_bytes", None)
        if device_budget_bytes is None:
            raise ValueError(
                "MemFabric HiCache ratio requires the pre-split device memory budget."
            )
        nominal_bytes = math.ceil(memory.hicache_ratio * device_budget_bytes)
    # Leave room for host pool padding. LOCAL commits whole GiB of DRAM,
    # so pass the aligned capacity to the allocator's byte accounting.
    available = host_bytes // _MEMFABRIC_GB * _MEMFABRIC_GB
    first_block_bytes = math.prod(dims) * dtype.itemsize
    minimum_bytes = (
        max(first_block_bytes, nominal_bytes) if fixed_size else first_block_bytes
    )
    if available < minimum_bytes:
        raise ValueError(
            f"Not enough host memory for MemFabric HiCache: need "
            f"{minimum_bytes / _MEMFABRIC_GB:.2f} GiB, but only "
            f"{available / _MEMFABRIC_GB:.2f} GiB is available after alignment."
        )
    target_bytes = max(
        first_block_bytes, math.ceil(nominal_bytes * (1 + _MEMFABRIC_RESERVE_SLACK))
    )
    target = -(-target_bytes // _MEMFABRIC_GB) * _MEMFABRIC_GB
    # The ratio estimate can include capacity later removed by device pool caps.
    total_bytes = min(target, available)
    backend.ensure_capacity(total_bytes, torch.npu.current_device())


# ---------------------------------------------------------------------------
# Sync-free H2D upload (NPU)
#
# A pageable .to(device) / torch.tensor(..., device=npu) completes with an
# aclrtStreamSynchronize that drains EVERYTHING queued on the current stream
# (profiler-confirmed on the HiCache load path).  When called on the default
# stream before entering the load/write stream, that sync stalls all queued
# compute; on the load stream it serializes the layer-group pipeline.  Stage
# through pinned memory instead: pinned + non_blocking=True is a genuinely
# async enqueue.  The pinned staging tensors are kept alive until their
# consumer copy retires, tracked via events (Event.query() is host-side and
# never synchronizes).
# ---------------------------------------------------------------------------
_pinned_inflight: list = []


def track_pinned_staging(pinned: torch.Tensor) -> None:
    """Keep a pinned staging tensor alive until its async consumer retires.

    Completed entries are dropped on each call so the list stays small.
    """
    done = torch.npu.Event()
    done.record()
    _pinned_inflight.append((pinned, done))
    _pinned_inflight[:] = [e for e in _pinned_inflight if not e[1].query()]


def to_device_no_sync(cpu_tensor: torch.Tensor, device) -> torch.Tensor:
    """Upload a CPU tensor to the NPU without synchronizing the stream.

    NPU-only helper (torch.npu.Event); callers are on the AscendC IO path.
    """
    pinned = cpu_tensor.pin_memory()
    out = torch.empty(pinned.shape, dtype=pinned.dtype, device=device)
    out.copy_(pinned, non_blocking=True)
    track_pinned_staging(pinned)
    return out
