# SPDX-License-Identifier: Apache-2.0
"""Validate software NVFP4 against unquantized NPU inference.

Run inside the NPU container with this checkout on PYTHONPATH. Logs and JSON
responses are retained in --output-dir. Device visibility is set by the caller.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


def wait_ready(process, url, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with {process.returncode}")
        try:
            if requests.get(url + "/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("Server startup timed out")


def generate(url, text, *, tokens=16, logprobs=False):
    payload = {
        "text": text,
        "sampling_params": {"temperature": 0, "max_new_tokens": tokens},
        "return_logprob": logprobs,
        "logprob_start_len": 0,
    }
    response = requests.post(url + "/generate", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def run_case(args, *, name, dtype, kv_dtype, workspace):
    url = f"http://127.0.0.1:{args.port}"
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--device",
        "npu",
        "--dtype",
        dtype,
        "--kv-cache-dtype",
        kv_dtype,
        "--prefill-kv-cache-dequant-dtype",
        workspace,
        "--context-length",
        "1024",
        "--max-total-tokens",
        "2048",
        "--max-running-requests",
        "4",
        "--chunked-prefill-size",
        "256",
        "--mem-fraction-static",
        "0.25",
        "--disable-overlap-schedule",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--random-seed",
        "42",
    ]
    output = Path(args.output_dir)
    (output / f"{name}.command.json").write_text(json.dumps(command, indent=2))
    with (output / f"{name}.server.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            wait_ready(process, url)
            print(f"READY {name}", flush=True)
            info = requests.get(url + "/server_info", timeout=10).json()
            prompt = "The capital city of France is"
            first = generate(url, prompt, tokens=16)
            repeat = generate(url, prompt, tokens=16)
            assert first["text"] == repeat["text"], (first, repeat)
            assert first["meta_info"]["completion_tokens"] >= 8, first
            assert "Paris" in first["text"], first
            batch = generate(
                url, ["2 + 2 =", "The capital city of Japan is"], tokens=16
            )
            assert len(batch) == 2 and all(item["text"] for item in batch), batch
            # More than one prefill chunk, plus a cached-prefix extend request.
            long_prompt = (
                "The garden has trees, flowers, birds and a fountain. " * 48
            ) + "The capital city of France is"
            cold = generate(url, long_prompt, tokens=16, logprobs=True)
            warm = generate(
                url, long_prompt + " Paris, and the capital city of Japan is", tokens=16
            )
            assert cold["meta_info"]["prompt_tokens"] > 256, cold
            assert warm["meta_info"]["cached_tokens"] >= 256, warm
            evidence = {
                "server_info": info,
                "first": first,
                "repeat": repeat,
                "batch": batch,
                "cold": cold,
                "warm": warm,
            }
            (output / f"{name}.responses.json").write_text(
                json.dumps(evidence, indent=2, ensure_ascii=False)
            )
            return evidence
        finally:
            stop(process)


def input_logprobs(evidence):
    return [
        row[0]
        for row in evidence["cold"]["meta_info"]["input_token_logprobs"]
        if row[0] is not None
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/weights/Qwen3-0.6B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, default=31884)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "baseline-bf16",
            "nvfp4-auto",
            "nvfp4-bf16",
            "baseline-fp16",
            "nvfp4-fp16",
        ],
    )
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cases = {
        "baseline-bf16": ("bfloat16", "auto", "auto"),
        "nvfp4-auto": ("bfloat16", "nvfp4", "auto"),
        "nvfp4-bf16": ("bfloat16", "nvfp4", "bf16"),
        "baseline-fp16": ("float16", "auto", "auto"),
        "nvfp4-fp16": ("float16", "nvfp4", "fp16"),
    }
    results = {}
    for name in args.cases:
        dtype, kv_dtype, workspace = cases[name]
        results[name] = run_case(
            args, name=name, dtype=dtype, kv_dtype=kv_dtype, workspace=workspace
        )
        print(f"PASS {name}", flush=True)
    summary = {}
    for name, baseline in (
        ("nvfp4-auto", "baseline-bf16"),
        ("nvfp4-bf16", "baseline-bf16"),
        ("nvfp4-fp16", "baseline-fp16"),
    ):
        if name in results and baseline in results:
            actual, expected = (
                input_logprobs(results[name]),
                input_logprobs(results[baseline]),
            )
            assert len(actual) == len(expected) and actual
            differences = [abs(a - b) for a, b in zip(actual, expected)]
            assert all(math.isfinite(x) for x in differences)
            mean_error = sum(differences) / len(differences)
            assert mean_error < 0.25, (name, mean_error)
            summary[name] = {
                "input_logprob_mean_absolute_error": mean_error,
                "tokens_compared": len(actual),
                "max_error": max(differences),
            }
    Path(args.output_dir, "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
