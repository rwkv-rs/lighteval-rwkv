# MIT License

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from io import BufferedReader
from pathlib import Path
from typing import Sequence

from lighteval.models.rwkv.http_pool import PoolError, PoolManifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/eval/lighteval-full.toml"
DEFAULT_MANIFESTS = {
    "1.5b": PROJECT_ROOT / "temp/vllm_pool.1.5b.json",
    "2.9b": PROJECT_ROOT / "temp/vllm_pool.2.9b.json",
    "7.2b": PROJECT_ROOT / "temp/vllm_pool.7.2b.json",
    "13.3b": PROJECT_ROOT / "temp/vllm_pool.13.3b.json",
}
G1J_MANIFESTS = {
    "1.5b": PROJECT_ROOT / "temp/vllm_pool.g1j.1.5b.json",
    "2.9b": PROJECT_ROOT / "temp/vllm_pool.g1j.2.9b.json",
    "7.2b": PROJECT_ROOT / "temp/vllm_pool.g1j.7.2b.json",
    "13.3b": PROJECT_ROOT / "temp/vllm_pool.g1j.13.3b.json",
}
DEFAULT_CAPACITIES = {"1.5b": 1024, "2.9b": 1024, "7.2b": 960, "13.3b": 320}
G1J_CAPACITIES = {"1.5b": 1024, "2.9b": 512, "7.2b": 256, "13.3b": 248}
DATASET_RATE_WINDOW_SECONDS = 300
DATASET_STARTS_PER_WINDOW = 2


@dataclass(frozen=True)
class ModelEvaluation:
    size: str
    capacity: int
    manifest: Path


def _parse_args(
    argv: Sequence[str] | None = None,
    *,
    default_manifests: dict[str, Path] = DEFAULT_MANIFESTS,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the four RWKV LightEval evaluations in parallel.")
    parser.add_argument(
        "--models",
        default="1.5b,2.9b,7.2b,13.3b",
        help="Comma-separated model sizes to evaluate (1.5b,2.9b,7.2b,13.3b).",
    )
    parser.add_argument(
        "--manifest-1.5b", dest="manifest_1_5b", type=Path, default=default_manifests["1.5b"], metavar="PATH"
    )
    parser.add_argument(
        "--manifest-2.9b", dest="manifest_2_9b", type=Path, default=default_manifests["2.9b"], metavar="PATH"
    )
    parser.add_argument(
        "--manifest-7.2b", dest="manifest_7_2b", type=Path, default=default_manifests["7.2b"], metavar="PATH"
    )
    parser.add_argument(
        "--manifest-13.3b",
        dest="manifest_13_3b",
        type=Path,
        default=default_manifests["13.3b"],
        metavar="PATH",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _evaluations(
    args: argparse.Namespace,
    expected_capacities: dict[str, int] = DEFAULT_CAPACITIES,
) -> tuple[ModelEvaluation, ...]:
    available = (
        ModelEvaluation("1.5B", expected_capacities["1.5b"], args.manifest_1_5b),
        ModelEvaluation("2.9B", expected_capacities["2.9b"], args.manifest_2_9b),
        ModelEvaluation("7.2B", expected_capacities["7.2b"], args.manifest_7_2b),
        ModelEvaluation("13.3B", expected_capacities["13.3b"], args.manifest_13_3b),
    )
    selected = [value.strip().lower() for value in args.models.split(",")]
    supported = {evaluation.size.lower(): evaluation for evaluation in available}
    if not selected or any(not value for value in selected):
        raise ValueError("--models must contain at least one model size")
    unknown = [value for value in selected if value not in supported]
    if unknown:
        raise ValueError("unknown --models values: " + ", ".join(unknown))
    if len(selected) != len(set(selected)):
        raise ValueError("--models must not contain duplicates")
    return tuple(supported[value] for value in selected)


def _validate(evaluations: Sequence[ModelEvaluation]) -> None:
    for evaluation in evaluations:
        manifest = PoolManifest.read(evaluation.manifest)
        if manifest.aggregate_capacity != evaluation.capacity:
            raise ValueError(
                f"{evaluation.size} pool capacity must be {evaluation.capacity}, "
                f"found {manifest.aggregate_capacity}: {evaluation.manifest}"
            )


def _command(args: argparse.Namespace) -> list[str]:
    command = ["uv", "run", "--no-sync", "lighteval", "rwkv", "--config", str(args.config)]
    if args.dry_run:
        command.append("--dry-run")
    return command


def _forward_output(stream: BufferedReader, *, stop_at_evaluation_started: bool = False) -> bool:
    for line in stream:
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()
        if stop_at_evaluation_started and b"RWKV evaluation started:" in line:
            return True
    return False


def main(  # noqa: C901
    argv: Sequence[str] | None = None,
    *,
    default_manifests: dict[str, Path] = DEFAULT_MANIFESTS,
    expected_capacities: dict[str, int] = DEFAULT_CAPACITIES,
) -> int:
    args = _parse_args(argv, default_manifests=default_manifests)
    try:
        evaluations = _evaluations(args, expected_capacities)
        _validate(evaluations)
    except (PoolError, ValueError) as error:
        print(f"Invalid RWKV evaluation manifests: {error}", file=sys.stderr, flush=True)
        return 2

    command = _command(args)
    processes: list[tuple[ModelEvaluation, subprocess.Popen[bytes]]] = []
    output_threads: list[threading.Thread] = []
    received_signal: int | None = None
    interrupted = threading.Event()
    dataset_start_times: deque[float] = deque()

    def forward_signal(signum, _frame) -> None:
        nonlocal received_signal
        received_signal = signum
        interrupted.set()
        for _, process in processes:
            if process.poll() is None:
                process.send_signal(signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)

    for evaluation in evaluations:
        if not args.dry_run:
            now = time.monotonic()
            while dataset_start_times and now - dataset_start_times[0] >= DATASET_RATE_WINDOW_SECONDS:
                dataset_start_times.popleft()
            if len(dataset_start_times) == DATASET_STARTS_PER_WINDOW:
                remaining = DATASET_RATE_WINDOW_SECONDS - (now - dataset_start_times[0])
                print(f"Waiting {remaining:.0f}s for the Hugging Face metadata rate window", flush=True)
                interrupted.wait(remaining)
                if received_signal is not None:
                    break
                now = time.monotonic()
                while dataset_start_times and now - dataset_start_times[0] >= DATASET_RATE_WINDOW_SECONDS:
                    dataset_start_times.popleft()
            dataset_start_times.append(time.monotonic())
        environment = os.environ.copy()
        environment["RWKV_EVAL_POOL_MANIFEST"] = str(evaluation.manifest.resolve())
        print(f"Starting {evaluation.size}: {evaluation.manifest}", flush=True)
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        processes.append((evaluation, process))
        assert process.stdout is not None
        evaluation_started = _forward_output(process.stdout, stop_at_evaluation_started=True)
        if evaluation_started:
            output_thread = threading.Thread(target=_forward_output, args=(process.stdout,), daemon=True)
            output_thread.start()
            output_threads.append(output_thread)
        elif (return_code := process.poll()) is not None and return_code < 0:
            received_signal = -return_code
            interrupted.set()
            break

    failed = [
        (evaluation.size, return_code) for evaluation, process in processes if (return_code := process.wait()) != 0
    ]
    for output_thread in output_threads:
        output_thread.join()
    if received_signal is not None:
        return 128 + received_signal
    if failed:
        summary = ", ".join(f"{size}={return_code}" for size, return_code in failed)
        print(f"RWKV evaluations failed: {summary}", flush=True)
        return 1
    print("All RWKV evaluations completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
