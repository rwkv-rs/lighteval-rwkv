# MIT License

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from lighteval.models.rwkv.http_pool import PoolError, PoolManifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/eval/lighteval.toml"


@dataclass(frozen=True)
class ModelEvaluation:
    size: str
    capacity: int
    manifest: Path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the four RWKV LightEval evaluations in parallel.")
    parser.add_argument(
        "--models",
        default="1.5b,2.9b,7.2b,13.3b",
        help="Comma-separated model sizes to evaluate (1.5b,2.9b,7.2b,13.3b).",
    )
    parser.add_argument("--manifest-1.5b", dest="manifest_1_5b", type=Path, required=True, metavar="PATH")
    parser.add_argument("--manifest-2.9b", dest="manifest_2_9b", type=Path, required=True, metavar="PATH")
    parser.add_argument("--manifest-7.2b", dest="manifest_7_2b", type=Path, required=True, metavar="PATH")
    parser.add_argument("--manifest-13.3b", dest="manifest_13_3b", type=Path, required=True, metavar="PATH")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=10)
    return parser.parse_args(argv)


def _evaluations(args: argparse.Namespace) -> tuple[ModelEvaluation, ...]:
    available = (
        ModelEvaluation("1.5B", 1024, args.manifest_1_5b),
        ModelEvaluation("2.9B", 1024, args.manifest_2_9b),
        ModelEvaluation("7.2B", 960, args.manifest_7_2b),
        ModelEvaluation("13.3B", 320, args.manifest_13_3b),
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
    command = ["uv", "run", "lighteval", "rwkv", "--config", str(args.config)]
    if args.dry_run:
        command.append("--dry-run")
    if args.max_samples is not None:
        command.extend(("--max-samples", str(args.max_samples)))
    return command


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    args = _parse_args(argv)
    try:
        evaluations = _evaluations(args)
        _validate(evaluations)
    except (PoolError, ValueError) as error:
        print(f"Invalid RWKV evaluation manifests: {error}", file=sys.stderr, flush=True)
        return 2

    command = _command(args)
    processes: list[tuple[ModelEvaluation, subprocess.Popen[bytes]]] = []
    received_signal: int | None = None

    def forward_signal(signum, _frame) -> None:
        nonlocal received_signal
        received_signal = signum
        for _, process in processes:
            if process.poll() is None:
                process.send_signal(signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)

    for evaluation in evaluations:
        environment = os.environ.copy()
        environment["RWKV_EVAL_POOL_MANIFEST"] = str(evaluation.manifest.resolve())
        print(f"Starting {evaluation.size}: {evaluation.manifest}", flush=True)
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment)
        processes.append((evaluation, process))

    failed = [
        (evaluation.size, return_code) for evaluation, process in processes if (return_code := process.wait()) != 0
    ]
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
