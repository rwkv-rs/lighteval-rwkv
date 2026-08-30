# MIT License

from __future__ import annotations

import os
import re
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Mapping


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

import typer

from lighteval.metrics.metrics_sample import SampleLevelComputation, SamplingMetric
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.rwkv.http_model import PROMPT_TEMPLATES, SAMPLING_PARAMETERS, RWKVHttpModel
from lighteval.models.rwkv.http_pool import PoolError, PoolManifest, RWKVHttpPool
from lighteval.tasks.requests import Doc, SamplingMethod


_CONFIG_FIELDS = {
    "schema_version",
    "pool_manifest",
    "output_dir",
    "prompt_template",
    "cot_mode",
    "benchmarks",
}
_ENV_REFERENCE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")
_MIN_COMPLETIONS = 5000
_LARGE_BENCHMARK_SIZE = 50_000
_LARGE_BENCHMARK_FRACTION = 0.2


class RWKVAvgAtK(SampleLevelComputation):
    """Average one native task scorer over exactly k independent completions."""

    def __init__(self, k: int, metric) -> None:
        self.k = k
        self.metric = metric

    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        return sum(self._score(doc, model_response[index]) for index in range(self.k)) / self.k

    def _score(self, doc: Doc, model_response) -> float:
        scorer = self.metric.sample_level_fn
        if isinstance(scorer, SamplingMetric):
            return float(scorer.compute_score(doc, model_response))
        value = next(iter(self.metric.compute_sample(doc=doc, model_response=model_response).values()))
        if isinstance(value, (list, tuple)):
            return sum(float(item) for item in value) / len(value)
        return float(value)


def _evaluation_plan(num_docs: int) -> tuple[int, int, str]:
    """Return evaluated documents, completions per document, and the only metric name."""
    if num_docs > _LARGE_BENCHMARK_SIZE:
        return int(num_docs * _LARGE_BENCHMARK_FRACTION), 1, "avg@0.2"
    k = 1
    while k * num_docs <= _MIN_COMPLETIONS:
        k *= 2
    return num_docs, k, f"avg@{k}"


def _configure_evaluation_plan(pipeline) -> None:
    """Apply the RWKV sampling plan after native task preparation."""
    for task in pipeline.tasks_dict.values():
        original_num_docs = len(task.eval_docs())
        effective_num_docs, k, metric_name = _evaluation_plan(original_num_docs)
        docs = pipeline.documents_dict[task.full_name][:effective_num_docs]
        pipeline.documents_dict[task.full_name] = docs
        source_metric = task.metrics[0]
        source_name = source_metric.metric_name[0] if isinstance(source_metric.metric_name, (list, tuple)) else None
        corpus_level_fn = source_metric.corpus_level_fn[source_name] if source_name else source_metric.corpus_level_fn
        higher_is_better = source_metric.higher_is_better[source_name] if source_name else source_metric.higher_is_better
        task.metrics = (
            SampleLevelMetric(
                metric_name=metric_name,
                sample_level_fn=RWKVAvgAtK(k, source_metric),
                category=SamplingMethod.GENERATIVE,
                corpus_level_fn=corpus_level_fn,
                higher_is_better=higher_is_better,
            ),
        )
        task.num_samples = [1, k]
        for doc in docs:
            doc.num_samples = k
        task.config = copy(task.config)
        task.config.metrics = task.metrics
        task.config.original_num_docs = original_num_docs
        task.config.effective_num_docs = len(docs)
        task.sampling_methods = [SamplingMethod.GENERATIVE]

    pipeline.sampling_docs = defaultdict(list)
    for docs in pipeline.documents_dict.values():
        for doc in docs:
            for sampling in doc.sampling_methods:
                pipeline.sampling_docs[sampling].append(doc)
    pipeline.evaluation_tracker.task_config_logger.log(pipeline.tasks_dict)


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


def resolve_benchmarks(selectors: tuple[str, ...]) -> ResolvedBenchmarks:
    """Resolve selectors without loading benchmark datasets or running inference."""
    from lighteval.tasks.registry import Registry
    from lighteval.tasks.requests import SamplingMethod

    registry = Registry(tasks=None, load_multilingual=True)
    missing: list[str] = []
    incompatible: list[str] = []
    owners: dict[str, str] = {}

    for selector in selectors:
        expanded = registry._expand_task_definition(selector)
        unavailable = [leaf for leaf in expanded if leaf not in registry._task_registry]
        if unavailable:
            missing.append(selector)
            continue
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
    return ResolvedBenchmarks(selector_count=len(selectors), leaf_tasks=tuple(sorted(owners)))


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
        from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

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
            convert_logprob_choices_to_generation=True,
        )
        pipeline = Pipeline(
            tasks=",".join(eval_config.benchmarks),
            pipeline_parameters=parameters,
            evaluation_tracker=tracker,
            model=model,
        )
        _configure_evaluation_plan(pipeline)
        from lighteval.logging.scoreboard import ScoreboardCallback

        scoreboard = ScoreboardCallback.from_environment(
            config_path=config,
            pipeline=pipeline,
            tracker=tracker,
            model=model,
        )
        if scoreboard is not None:
            tracker.details_logger = scoreboard.details_logger()
        pipeline.evaluate()
        pipeline.show_results()
        pipeline.save_and_push_results()
        if scoreboard is not None:
            scoreboard.finalize()
    except (ConfigError, PoolError, ValueError) as error:
        typer.echo(f"RWKV evaluation failed: {error}", err=True)
        raise typer.Exit(code=2) from error
    finally:
        if model is not None:
            model.cleanup()
        elif pool is not None:
            pool.close()
