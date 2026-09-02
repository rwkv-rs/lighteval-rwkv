# MIT License

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Mapping


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

import typer

from lighteval.models.rwkv.http_model import PROMPT_TEMPLATES, SAMPLING_PARAMETERS, RWKVHttpModel
from lighteval.models.rwkv.http_pool import PoolError, PoolManifest, RWKVHttpPool
from lighteval.models.rwkv.pipeline import RWKVPipeline


_CONFIG_FIELDS = {
    "schema_version",
    "pool_manifest",
    "output_dir",
    "prompt_template",
    "cot_mode",
    "benchmarks",
}
_ENV_REFERENCE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")


class ConfigError(ValueError):
    """Raised when the RWKV one-click evaluation contract is invalid."""


@dataclass(frozen=True)
class RWKVEvaluationConfig:
    pool_manifest: Path
    output_dir: Path
    prompt_template: str
    cot_mode: str
    benchmarks: tuple[str, ...]

    @classmethod
    def read(  # noqa: C901
        cls,
        path: Path,
        env: Mapping[str, str] = os.environ,
    ) -> RWKVEvaluationConfig:
        try:
            with path.open("rb") as stream:
                raw = tomllib.load(stream)
        except FileNotFoundError as error:
            raise ConfigError(f"RWKV eval config not found: {path}") from error
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise ConfigError(f"invalid RWKV eval TOML: {error}") from error
        if not isinstance(raw, dict):
            raise ConfigError("RWKV eval config must be a TOML table")

        unknown = sorted(set(raw) - _CONFIG_FIELDS)
        if unknown:
            raise ConfigError("unknown RWKV eval config fields: " + ", ".join(unknown))
        missing = sorted(_CONFIG_FIELDS - set(raw))
        if missing:
            raise ConfigError("missing RWKV eval config fields: " + ", ".join(missing))
        if raw["schema_version"] != 1 or isinstance(raw["schema_version"], bool):
            raise ConfigError("RWKV eval schema_version must be 1")

        pool_value = cls._expand_environment(raw["pool_manifest"], env, "pool_manifest")
        pool_manifest = Path(pool_value).expanduser()
        if not pool_manifest.is_absolute():
            pool_manifest = path.parent / pool_manifest
        if not pool_manifest.is_file() or pool_manifest.is_symlink():
            raise ConfigError(f"RWKV pool manifest must be a regular file: {pool_manifest}")

        output_value = raw["output_dir"]
        if not isinstance(output_value, str) or not output_value.strip():
            raise ConfigError("output_dir must be a non-empty string")
        output_dir = Path(output_value).expanduser()

        prompt_template = raw["prompt_template"]
        if prompt_template not in PROMPT_TEMPLATES:
            raise ConfigError("prompt_template must be one of: " + ", ".join(PROMPT_TEMPLATES))
        cot_mode = raw["cot_mode"]
        if cot_mode not in SAMPLING_PARAMETERS:
            raise ConfigError("cot_mode must be one of: " + ", ".join(SAMPLING_PARAMETERS))

        configured_benchmarks = raw["benchmarks"]
        if not isinstance(configured_benchmarks, list) or not configured_benchmarks:
            raise ConfigError("benchmarks must be a non-empty array")
        if any(
            not isinstance(value, str) or not value or value != value.strip() or "," in value
            for value in configured_benchmarks
        ):
            raise ConfigError("benchmarks must contain non-empty selector strings without commas")
        benchmarks = tuple(configured_benchmarks)
        duplicates = sorted(selector for selector in set(benchmarks) if benchmarks.count(selector) > 1)
        if duplicates:
            raise ConfigError("duplicate benchmark selectors: " + ", ".join(duplicates))

        return cls(
            pool_manifest=pool_manifest,
            output_dir=output_dir,
            prompt_template=prompt_template,
            cot_mode=cot_mode,
            benchmarks=benchmarks,
        )

    @staticmethod
    def _expand_environment(value: object, env: Mapping[str, str], field_name: str) -> str:
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{field_name} must be a non-empty string")
        match = _ENV_REFERENCE.fullmatch(value)
        if match is None:
            return value
        variable = match.group(1)
        expanded = env.get(variable)
        if not expanded:
            raise ConfigError(f"missing environment variable referenced by {field_name}: {variable}")
        return expanded


@dataclass(frozen=True)
class ResolvedBenchmarks:
    selector_count: int
    leaf_tasks: tuple[str, ...]
    selector_tasks: tuple[tuple[str, tuple[str, ...]], ...] = ()


def _selector_sample_budgets(
    resolved: ResolvedBenchmarks, max_samples: int | None
) -> tuple[dict[str, tuple[str, ...]], dict[str, int] | None]:
    selector_tasks = dict(resolved.selector_tasks)
    if max_samples is None:
        return selector_tasks, None
    budgets: dict[str, int] = {}
    for leaves in selector_tasks.values():
        if len(leaves) <= max_samples:
            base, remainder = divmod(max_samples, len(leaves))
            for index, leaf in enumerate(leaves):
                budgets[leaf] = base + (index < remainder)
            continue
        selected = {leaves[index * len(leaves) // max_samples] for index in range(max_samples)}
        for leaf in selected:
            budgets[leaf] = 1
    return selector_tasks, budgets


def resolve_benchmarks(selectors: tuple[str, ...]) -> ResolvedBenchmarks:
    """Resolve selectors without loading benchmark datasets or running inference."""
    from lighteval.tasks.registry import Registry
    from lighteval.tasks.requests import SamplingMethod

    registry = Registry(tasks=None, load_multilingual=True)
    missing: list[str] = []
    incompatible: list[str] = []
    owners: dict[str, str] = {}
    selector_tasks: list[tuple[str, tuple[str, ...]]] = []

    for selector in selectors:
        expanded = registry._expand_task_definition(selector)
        unavailable = [leaf for leaf in expanded if leaf not in registry._task_registry]
        if unavailable:
            missing.append(selector)
            continue
        selector_tasks.append((selector, tuple(expanded)))
        for leaf in expanded:
            if leaf in owners:
                raise ConfigError(f"benchmark selectors overlap on {leaf}: {owners[leaf]}, {selector}")
            owners[leaf] = selector
            config = registry._task_registry[leaf]
            categories = {metric.category for metric in config.metrics}
            if not categories or not categories.issubset({SamplingMethod.GENERATIVE, SamplingMethod.LOGPROBS}):
                incompatible.append(leaf)

    if missing:
        raise ConfigError(
            "benchmark selectors are not registered in LightEval: "
            + ", ".join(missing)
            + "; evaluate them with another framework"
        )
    if incompatible:
        raise ConfigError(
            "benchmark leaf tasks are incompatible with the generative RWKV adapter: "
            + ", ".join(sorted(incompatible))
        )
    if not owners:
        raise ConfigError("no LightEval benchmark leaf tasks were resolved")
    return ResolvedBenchmarks(
        selector_count=len(selectors),
        leaf_tasks=tuple(sorted(owners)),
        selector_tasks=tuple(selector_tasks),
    )


def _preflight(config: RWKVEvaluationConfig) -> tuple[PoolManifest, RWKVHttpPool, ResolvedBenchmarks]:
    resolved = resolve_benchmarks(config.benchmarks)
    manifest = PoolManifest.read(config.pool_manifest)
    pool = RWKVHttpPool(manifest, api_key=os.environ.get("RWKV_EVAL_API_KEY"))
    try:
        pool.preflight()
    except Exception:
        pool.close()
        raise
    return manifest, pool, resolved


def _print_preflight(
    config: RWKVEvaluationConfig,
    manifest: PoolManifest,
    resolved: ResolvedBenchmarks,
    max_samples: int | None,
) -> None:
    typer.echo(f"selectors: {resolved.selector_count}")
    typer.echo(f"leaf tasks: {len(resolved.leaf_tasks)}")
    typer.echo(f"model: {manifest.model_name}")
    typer.echo(f"served model: {manifest.served_model_name}")
    typer.echo(f"model revision: {manifest.model_revision}")
    typer.echo(f"wkv_mode: {manifest.wkv_mode}")
    typer.echo(f"vLLM version: {manifest.vllm_version}")
    typer.echo(f"pool capacity: {manifest.aggregate_capacity}")
    typer.echo(f"pool fingerprint: {manifest.fingerprint}")
    typer.echo(f"prompt template: {config.prompt_template}")
    typer.echo(f"CoT mode: {config.cot_mode}")
    typer.echo(f"output directory: {config.output_dir}")
    typer.echo("run mode: full" if max_samples is None else f"run mode: partial (max_samples={max_samples})")


def rwkv(
    config: Annotated[
        Path,
        typer.Option("--config", dir_okay=False, help="Path to the RWKV LightEval TOML configuration."),
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate tasks and every pool endpoint without evaluating datasets."),
    ] = False,
    max_samples: Annotated[
        int | None,
        typer.Option("--max-samples", min=1, help="Partial smoke-test limit; omit for full benchmark splits."),
    ] = None,
) -> None:
    """Evaluate the configured native LightEval benchmarks on an existing RWKV endpoint pool."""
    pool: RWKVHttpPool | None = None
    model: RWKVHttpModel | None = None
    try:
        eval_config = RWKVEvaluationConfig.read(config)
        manifest, pool, resolved = _preflight(eval_config)
        _print_preflight(eval_config, manifest, resolved, max_samples)
        if dry_run:
            pool.close()
            return

        from lighteval.logging.evaluation_tracker import EvaluationTracker
        from lighteval.pipeline import ParallelismManager, PipelineParameters

        model = RWKVHttpModel(
            manifest=manifest,
            prompt_template=eval_config.prompt_template,
            cot_mode=eval_config.cot_mode,
            cache_dir=eval_config.output_dir / ".cache",
            max_samples=max_samples,
            pool=pool,
        )
        tracker = EvaluationTracker(output_dir=str(eval_config.output_dir), save_details=True)
        parameters = PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            max_samples=max_samples,
            load_tasks_multilingual=True,
        )
        selector_tasks, task_max_samples = _selector_sample_budgets(resolved, max_samples)
        pipeline = RWKVPipeline(
            tasks=",".join(eval_config.benchmarks),
            pipeline_parameters=parameters,
            evaluation_tracker=tracker,
            model=model,
            selector_tasks=selector_tasks,
            task_max_samples=task_max_samples,
        )
        from lighteval.logging.scoreboard import ScoreboardCallback

        scoreboard = ScoreboardCallback.from_environment(
            config_path=config,
            pipeline=pipeline,
            tracker=tracker,
            model=model,
        )
        if scoreboard is not None:
            pipeline.task_callback = scoreboard
        pipeline.evaluate()
        pipeline.show_results()
        pipeline.save_and_push_results()
    except (ConfigError, PoolError, ValueError) as error:
        typer.echo(f"RWKV evaluation failed: {error}", err=True)
        raise typer.Exit(code=2) from error
    finally:
        if model is not None:
            model.cleanup()
        elif pool is not None:
            pool.close()
