# SPDX-License-Identifier: Apache-2.0
"""Run the software NVFP4 byte and pool tests on a real NPU."""

import runpy
import unittest
from pathlib import Path

import torch
import torch_npu  # noqa: F401

if __name__ == "__main__":
    torch.npu.set_device(0)
    test_file = Path(__file__).with_name("test_npu_nvfp4_kv_cache.py")
    case = runpy.run_path(str(test_file))["TestNPUNVFP4KVCache"]
    case.device = "npu"
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(case)
    )
    raise SystemExit(not result.wasSuccessful())
