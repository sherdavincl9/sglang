"""PLE GSM8K: 5-shot completion, 1314 scored questions; run with python3."""

import json
import os
import resource
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import MODEL_WEIGHTS_DIR
from sglang.test.run_eval import run_eval
from sglang.test.simple_eval_mixed_prefix_gsm8k import GSM8K_URL
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.utils import download_and_cache_file, read_jsonl

MODEL = os.path.join(MODEL_WEIGHTS_DIR, "Qwen3.8-Flash-Next")
BACKEND = "pinned"  # Change to "file" to evaluate that backend.
MIN_ACCURACY = 0.95  # Reference CI gate, not a Qwen model-card score.
MAX_TOKENS = 512
# Repository layout: <work>/code-mine/sglang/test/manual/ascend/this_file.py.
SCRIPT_DIR = Path(__file__).resolve().parents[5] / "script"


class TestNpuPleOffloadAccuracy(CustomTestCase):
    def test_gsm8k(self):
        self.assertIn(BACKEND, ("pinned", "file"))
        self.assertTrue(os.path.isdir(MODEL), f"Model weights not found: {MODEL}")
        data_dir, log_dir = SCRIPT_DIR / "datasets/gsm8k", SCRIPT_DIR / "logs"
        data_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        data_path = download_and_cache_file(GSM8K_URL, str(data_dir / "test.jsonl"))
        rows = list(read_jsonl(data_path))
        self.assertEqual(len(rows), 1319, "Incomplete GSM8K data; check test.jsonl")
        self.assertTrue(all(row.get("question") and row.get("answer") for row in rows))
        report_dir = Path(tempfile.mkdtemp(prefix=f"gsm8k_{BACKEND}_", dir=log_dir))
        print(
            f"GSM8K: backend={BACKEND}, scored=1314, reports={report_dir}", flush=True
        )
        if BACKEND == "pinned":
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))
        with tempfile.TemporaryDirectory(prefix="ple_gsm8k_", dir=SCRIPT_DIR) as tables:
            process = popen_launch_server(
                MODEL,
                DEFAULT_URL_FOR_TEST,
                timeout=1800,
                other_args=(
                    "--trust-remote-code --attention-backend ascend --tp-size 8 "
                    "--mamba-ssm-dtype bfloat16 --mem-fraction-static 0.85 "
                    "--cuda-graph-max-bs-decode 4 --ple-offload-embedding"
                ).split()
                + ["--ple-offload-backend", BACKEND, "--ple-offload-dir", tables],
            )
            try:
                args = SimpleNamespace(
                    model=MODEL,
                    base_url=DEFAULT_URL_FOR_TEST,
                    eval_name="gsm8k",
                    api="completion",
                    gsm8k_data_path=data_path,
                    num_examples=1319,
                    num_shots=5,
                    num_threads=4,
                    max_tokens=MAX_TOKENS,
                    temperature=0.0,
                )
                metrics = run_eval(args)
                for suffix in ("html", "json"):
                    source = Path(f"/tmp/gsm8k_{MODEL.replace('/', '_')}.{suffix}")
                    shutil.move(str(source), str(report_dir / f"gsm8k.{suffix}"))
                (report_dir / "config.json").write_text(
                    json.dumps(
                        vars(args) | {"backend": BACKEND, "min_accuracy": MIN_ACCURACY},
                        indent=2,
                    )
                )
                self.assertGreaterEqual(metrics["score"], MIN_ACCURACY, str(report_dir))
            finally:
                kill_process_tree(process.pid)
                process.wait(timeout=30)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
