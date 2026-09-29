"""Register the benchmarks configured in ``configs/benchmarks.toml``."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import httpx

from src.rwkv_eval.configs import read_benchmarks


LOGGER = logging.getLogger(__name__)
DEFAULT_API_URL = "https://eval.rwkv.rs/test/api"


def add_benchmarks(
    config_path: str | Path,
    *,
    token: str,
    api_url: str = DEFAULT_API_URL,
) -> list[dict[str, Any]]:
    """Register every benchmark in a TOML configuration.

    The Scoreboard API registers one benchmark per request.  The function keeps
    the requests in configuration order and returns the API responses so that
    callers can use the assigned benchmark IDs if needed.
    """
    benchmarks = read_benchmarks(config_path)
    if not token:
        raise ValueError("an API token is required")

    responses: list[dict[str, Any]] = []
    with httpx.Client(
        base_url=f"{api_url.rstrip('/')}/",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    ) as client:
        for benchmark in benchmarks:
            response = client.post(
                "add_benchmarks",
                json={"name": benchmark.selector, "field": benchmark.field},
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError(f"unexpected response for benchmark {benchmark.selector}: {result!r}")
            responses.append(result)
            LOGGER.info("registered benchmark %s", benchmark.selector)
    return responses


def _argument_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Register RWKV benchmarks with the Scoreboard API.")
    parser.add_argument("--benchmarks", type=Path, default=root / "configs/benchmarks.toml")
    parser.add_argument("--token", default=os.environ.get("SCOREBOARD_API_TOKEN"))
    parser.add_argument("--api-url", default=os.environ.get("SCOREBOARD_API_URL", DEFAULT_API_URL))
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    add_benchmarks(args.benchmarks, token=args.token or "", api_url=args.api_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
