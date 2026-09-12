# MIT License

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
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
DEFAULT_MAX_RESTARTS = 5
DEFAULT_RESTART_DELAY_SECONDS = 15


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
    parser.add_argument("--run-id", default=os.environ.get("RWKV_EVAL_RUN_ID", "default"))
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("RWKV_EVAL_OUTPUT_ROOT", "results")))
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=int(os.environ.get("RWKV_EVAL_MAX_RESTARTS", DEFAULT_MAX_RESTARTS)),
        help="Maximum retries for one model process after an unsuccessful exit.",
    )
    parser.add_argument(
        "--restart-delay",
        type=float,
        default=float(os.environ.get("RWKV_EVAL_RESTART_DELAY", DEFAULT_RESTART_DELAY_SECONDS)),
        help="Seconds to wait before retrying one model process.",
    )
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


def _model_config(args: argparse.Namespace, evaluation: ModelEvaluation) -> Path:
    root = getattr(args, "output_root", Path("results")) / getattr(args, "run_id", "default") / evaluation.size.lower()
    root.mkdir(parents=True, exist_ok=True)
    text = args.config.read_text()
    lines = [f'output_dir = "{root.resolve()}"' if line.strip().startswith("output_dir") else line for line in text.splitlines()]
    path = root / ".runner-config.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def _command(args: argparse.Namespace, config: Path) -> list[str]:
    command = ["uv", "run", "--no-sync", "lighteval", "rwkv", "--config", str(config)]
    if args.dry_run:
        command.append("--dry-run")
    return command


def _forward_output(stream: BufferedReader, *, label: str | None = None) -> None:
    for line in stream:
        if label is not None:
            line = f"[{label}] ".encode() + line
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()


def _start_process(
    args: argparse.Namespace,
    evaluation: ModelEvaluation,
    config: Path,
) -> subprocess.Popen[bytes]:
    environment = os.environ.copy()
    environment["RWKV_EVAL_POOL_MANIFEST"] = str(evaluation.manifest.resolve())
    print(f"Starting {evaluation.size}: {evaluation.manifest}", flush=True)
    return subprocess.Popen(
        _command(args, config),
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _run_model(
    args: argparse.Namespace,
    evaluation: ModelEvaluation,
    config: Path,
    processes: dict[str, subprocess.Popen[bytes]],
    process_lock: threading.Lock,
    interrupted: threading.Event,
    received_signal: list[int | None],
) -> int:
    restarts = 0
    while not interrupted.is_set():
        process = _start_process(args, evaluation, config)
        with process_lock:
            processes[evaluation.size] = process
        assert process.stdout is not None
        output_thread = threading.Thread(
            target=_forward_output,
            args=(process.stdout,),
            kwargs={"label": evaluation.size},
            daemon=True,
        )
        output_thread.start()
        return_code = process.wait()
        output_thread.join()
        with process_lock:
            if processes.get(evaluation.size) is process:
                del processes[evaluation.size]

        if interrupted.is_set():
            return 128 + (received_signal[0] or signal.SIGTERM)
        if return_code == 0:
            print(f"Completed {evaluation.size} successfully.", flush=True)
            return 0

        restarts += 1
        max_restarts = getattr(args, "max_restarts", DEFAULT_MAX_RESTARTS)
        restart_delay = getattr(args, "restart_delay", DEFAULT_RESTART_DELAY_SECONDS)
        if restarts > max_restarts:
            print(
                f"{evaluation.size} failed with exit code {return_code} after {restarts - 1} retries.",
                file=sys.stderr,
                flush=True,
            )
            return return_code
        print(
            f"{evaluation.size} exited with code {return_code}; retry {restarts}/{max_restarts} "
            f"in {restart_delay:g}s.",
            file=sys.stderr,
            flush=True,
        )
        if interrupted.wait(restart_delay):
            return 128 + (received_signal[0] or signal.SIGTERM)

    return 128 + (received_signal[0] or signal.SIGTERM)


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

    run_id = getattr(args, "run_id", "default")
    output_root = getattr(args, "output_root", Path("results"))
    if not args.dry_run:
        metadata_dir = output_root / run_id
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / "run-manifest.json"
        config_bytes = args.config.read_bytes()
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        metadata = {
            "run_id": run_id,
            "config_sha256": config_hash,
            "models": [e.size for e in evaluations],
            "manifests": {e.size: str(e.manifest.resolve()) for e in evaluations},
        }
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text())
            if existing != metadata:
                print(f"run metadata mismatch: {metadata_path}; choose a new --run-id", file=sys.stderr)
                return 2
        else:
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    process_configs = {
        evaluation.size: args.config if args.dry_run else _model_config(args, evaluation)
        for evaluation in evaluations
    }
    processes: dict[str, subprocess.Popen[bytes]] = {}
    process_lock = threading.Lock()
    received_signal: list[int | None] = [None]
    interrupted = threading.Event()

    def forward_signal(signum, _frame) -> None:
        received_signal[0] = signum
        interrupted.set()
        with process_lock:
            active_processes = tuple(processes.values())
        for process in active_processes:
            if process.poll() is None:
                process.send_signal(signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)

    workers = []
    worker_results: dict[str, int] = {}
    result_lock = threading.Lock()

    def run_and_record(evaluation: ModelEvaluation) -> None:
        result = _run_model(
            args,
            evaluation,
            process_configs[evaluation.size],
            processes,
            process_lock,
            interrupted,
            received_signal,
        )
        with result_lock:
            worker_results[evaluation.size] = result

    if received_signal[0] is None:
        for evaluation in evaluations:
            worker = threading.Thread(target=run_and_record, args=(evaluation,), name=f"rwkv-{evaluation.size}")
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()

    if received_signal[0] is not None:
        return 128 + (received_signal[0] or signal.SIGTERM)
    failed = [(evaluation.size, worker_results[evaluation.size]) for evaluation in evaluations if worker_results[evaluation.size]]
    if failed:
        summary = ", ".join(f"{size}={return_code}" for size, return_code in failed)
        print(f"RWKV evaluations failed: {summary}", flush=True)
        return 1
    print("All RWKV evaluations completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
