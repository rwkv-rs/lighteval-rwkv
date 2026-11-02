"""RWKV evaluation entry point.

The entry point only translates the RWKV TOML manifests into LightEval's
native model, pipeline, metrics, and tracker interfaces.  Backend scheduling,
prompt preparation, task loading, sampling counts, and scoring belong to
LightEval itself.
"""

from __future__ import annotations

import argparse
import collections
import copy
import fcntl
import hashlib
import json
import logging
import math
import os
import pwd
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Literal, Sequence

import httpx

from lighteval.metrics.metrics_sample import AvgAtN, SamplingMetric
from lighteval.pipeline import Pipeline
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import SamplingMethod
from src.rwkv_eval.answer_extract.free_response import (
    answer_extractor_for_benchmark,
    extract_answer,
)
from src.rwkv_eval.configs import (
    BenchmarkSpec,
    ModelEndpoint,
    RWKVAvgAtN,
    RWKVLightevalTaskConfig,
    RwkvModel,
    SamplingConfig,
    read_benchmarks,
    read_models,
)
from src.rwkv_eval.rwkv_model import RWKVClient, _litelm_model, _sampling_config
from src.rwkv_eval.upload import Detail, Score
from src.rwkv_eval.upload import upload as upload_score


LOGGER = logging.getLogger(__name__)
REQUEST_RETRIES = 6
RETRY_DELAY = 1.0
TEST_MODE_MAX_SAMPLES = 10


CotMode = Literal["NoCoT", "FakeCoT", "CoT"]


class RWKVLightevalTask(LightevalTask):
    """Prototype: LightevalTask._get_docs_from_split; split-local cache IDs."""

    @property
    def cot_mode(self) -> CotMode:
        if self.sampling_methods == [SamplingMethod.LOGPROBS]:
            return "NoCoT"
        return self.config.cot_mode

    def _get_docs_from_split(self, splits: list[str], few_shots=False):
        if few_shots or len(self.evaluation_split) <= 1:
            return super()._get_docs_from_split(splits, few_shots=few_shots)
        docs = []
        for split in splits:
            split_docs = super()._get_docs_from_split([split], few_shots=few_shots)
            for doc in split_docs:
                doc.id = f"{split}:{doc.id}"
            docs.extend(split_docs)
        return docs


class RWKVRegistry(Registry):
    """Prototype: Registry; task and output policy stay private to RWKV."""

    @staticmethod
    def _extract_configs(module):
        configs = Registry._extract_configs(module)
        configs.update({config.name: config for config in getattr(module, "RWKV_TASKS_TABLE", ())})
        return configs

    def load_tasks(self) -> dict[str, LightevalTask]:
        if len(self.task_to_configs) == 0:
            return {config.full_name: RWKVLightevalTask(config=config) for config in self._task_registry.values()}
        return {
            config.full_name: RWKVLightevalTask(config=config)
            for configs in self.task_to_configs.values()
            for config in configs
        }

    def _update_task_configs(self):
        configs = super()._update_task_configs()
        for task_configs in configs.values():
            for index, original in enumerate(task_configs):
                config = (
                    original if isinstance(original, RWKVLightevalTaskConfig) else RWKVLightevalTaskConfig(original)
                )
                task_configs[index] = config
        return configs


class RWKVPipeline(Pipeline):
    """Prototype: native Pipeline; RWKV task/output policy is isolated here."""

    def __init__(self, *args, avg_k: int | None = None, registry: RWKVRegistry | None = None, **kwargs):
        self.rwkv_avg_k = avg_k
        self.registry = registry
        super().__init__(*args, **kwargs)

    def _configure_sampling_counts(self, tasks=None):
        tasks = list(tasks or self.tasks_dict.values())
        if self.rwkv_avg_k is None:
            super()._configure_sampling_counts(tasks)
        for task in tasks:
            count = self.rwkv_avg_k or max(count for count in task.num_samples if count is not None)
            if task.sampling_methods[0] != SamplingMethod.GENERATIVE and count != 1:
                raise ValueError(f"{task.name}: non-generative evaluation requires avg_k=1")
            task.num_samples = [count]
            for metric in task.metrics:
                sample_fn = metric.sample_level_fn
                if isinstance(sample_fn, SamplingMetric) and hasattr(sample_fn, "n"):
                    sample_fn.n = count
                    if isinstance(metric.metric_name, list):
                        metric.metric_name = [re.sub(r"n=\d+", f"n={count}", name) for name in metric.metric_name]
                    else:
                        metric.metric_name = re.sub(r"n=\d+", f"n={count}", metric.metric_name)
            if task.sampling_methods != [SamplingMethod.GENERATIVE] or task.name.startswith("lcb:codegeneration"):
                continue
            if any(isinstance(metric.sample_level_fn, RWKVAvgAtN) for metric in task.metrics):
                continue
            from lighteval.metrics.metrics import Metrics

            primary = task.metrics[0]
            average = copy.deepcopy(Metrics.avg_at_n.value)
            average.sample_level_fn = RWKVAvgAtN(primary, count)
            if isinstance(primary.sample_level_fn, AvgAtN):
                average.metric_name = primary.metric_name
                task.metrics = (average, *task.metrics[1:])
            else:
                average.metric_name = "rwkv_avg@n"
                task.metrics = (*task.metrics, average)
            task.config.metrics = task.metrics

    def _init_tasks_and_requests(self, tasks: str):
        # Follow the native initialization order without monkeypatching Registry
        # or altering TASKS_TABLE/metric singleton objects for other models.
        self.registry = self.registry or RWKVRegistry(
            tasks=tasks,
            load_multilingual=self.pipeline_parameters.load_tasks_multilingual,
            custom_tasks=self.pipeline_parameters.custom_tasks_directory,
        )
        self.tasks_dict = self.registry.load_tasks()
        self.documents_dict = {}
        self.sampling_docs = collections.defaultdict(list)
        if not self.pipeline_parameters.streaming_evaluation:
            LightevalTask.load_datasets(self.tasks_dict, self.pipeline_parameters.dataset_loading_processes)
            self._configure_sampling_counts()
            self.documents_dict = {
                task.full_name: task.get_docs(self.pipeline_parameters.max_samples)
                for task in self.tasks_dict.values()
            }
            for docs in self.documents_dict.values():
                for doc in docs:
                    for sampling in doc.sampling_methods:
                        self.sampling_docs[sampling].append(doc)
        if self._metric_options:
            self._update_num_samples(list(self.tasks_dict.values()))
        self.evaluation_tracker.task_config_logger.log(self.tasks_dict)

    def _run_model_for_docs(self, sampling_docs):
        outputs = super()._run_model_for_docs(sampling_docs)
        for method, responses in outputs.items():
            if method != SamplingMethod.GENERATIVE:
                continue
            # Use this batch's documents, not self.sampling_docs: streaming
            # evaluation intentionally leaves the latter empty.
            for doc, response in zip(sampling_docs[method], responses, strict=True):
                extractor = answer_extractor_for_benchmark(doc.task_name)
                response.text_post_processed = [
                    ""
                    if response.finish_reasons and response.finish_reasons[index].lower() in {"length", "max_tokens"}
                    else extract_answer(
                        text,
                        extractor,
                        len(doc.choices),
                        rendered_prompt=response.rendered_prompt,
                    )
                    or ""
                    for index, text in enumerate(response.text)
                ]
        return outputs

    def _retain_streaming_detail(self, task_name, doc, response, output):
        task = self.tasks_dict[task_name]
        rollout_scores = (doc.specific or {}).get("rwkv_rollout_scores")
        if rollout_scores is None:
            metric = task.metrics[0]
            name = metric.metric_name[0] if isinstance(metric.metric_name, list) else metric.metric_name
            rollout_scores = [float(output[name])]
        if len(rollout_scores) != doc.num_samples:
            raise RuntimeError(f"{task_name}: missing per-rollout scores")
        counts = self.streaming_detail_counts[task_name]
        buckets = []
        for index, score in enumerate(rollout_scores):
            if task.sampling_methods == [SamplingMethod.GENERATIVE]:
                finish = response.finish_reasons[index] if index < len(response.finish_reasons) else "stop"
                failed = finish.lower() in {"length", "max_tokens"} or not response.final_text[index]
            else:
                failed = False
            buckets.append(2 if failed else 0 if score == 1.0 else 1)
        if all(counts[bucket] >= 20 for bucket in buckets):
            return
        for bucket in buckets:
            counts[bucket] = min(counts[bucket] + 1, 20)
        self.streaming_details[task_name].append(
            self.evaluation_tracker.details_logger.Detail(
                copy.deepcopy(doc), copy.deepcopy(response), copy.deepcopy(output)
            )
        )


def evaluate(  # noqa: C901
    models: Sequence[ModelEndpoint],
    benchmarks: Sequence[BenchmarkSpec],
    *,
    prompt_template: str | None = None,
    max_samples: int | None = None,
    test_mode: bool = False,
    seed: int = 42,
    scoreboard_token: str | None = None,
    scoreboard_url: str = "https://eval.rwkv.rs/test/api",
    no_upload: bool = False,
    output_dir: str = "results",
) -> tuple[list[Score], list[Score]]:
    """Run native LightEval independently for each configured model pool."""
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.pipeline import ParallelismManager, PipelineParameters

    if not models or not benchmarks:
        raise ValueError("models and benchmarks must not be empty")
    if prompt_template not in {None, "bot", "assistant", "function_calling"}:
        raise ValueError(f"unsupported prompt_template: {prompt_template!r}")
    if not no_upload and not scoreboard_token:
        raise ValueError("scoreboard token is required unless --no-upload is used")
    groups = collections.defaultdict(list)
    for endpoint in models:
        groups[(endpoint.model_name, endpoint.ctx_len, endpoint.is_preview)].append(endpoint)
    if len(groups) > 1:
        results, errors = [], []
        with ProcessPoolExecutor(
            max_workers=len(groups),
            mp_context=get_context("spawn"),
            initializer=partial(
                logging.basicConfig,
                level=logging.getLogger().getEffectiveLevel(),
                format="%(asctime)s %(levelname)s %(message)s",
            ),
        ) as executor:
            jobs = [
                executor.submit(
                    evaluate,
                    replicas,
                    benchmarks,
                    prompt_template=prompt_template,
                    max_samples=max_samples,
                    test_mode=test_mode,
                    seed=seed,
                    scoreboard_token=scoreboard_token,
                    scoreboard_url=scoreboard_url,
                    no_upload=no_upload,
                    output_dir=output_dir,
                )
                for replicas in groups.values()
            ]
            for job in jobs:
                try:
                    results.append(job.result())
                except Exception as error:
                    errors.append(error)
        if errors:
            raise errors[0]
        return [score for scores, _ in results for score in scores], [
            score for _, failures in results for score in failures
        ]

    endpoint = models[0]
    model = RwkvModel.from_model_name(endpoint.model_name, endpoint.ctx_len, endpoint.is_preview)
    directory = (
        Path(output_dir)
        / "scores"
        / (
            re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(endpoint.model_name).name)
            + f"-ctx{endpoint.ctx_len}-{'preview' if endpoint.is_preview else 'release'}"
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BlockingIOError(f"another evaluation producer owns {directory}") from error
        effective_max_samples = max_samples
        if test_mode:
            effective_max_samples = (
                TEST_MODE_MAX_SAMPLES if max_samples is None else min(max_samples, TEST_MODE_MAX_SAMPLES)
            )
        configured = {(spec.selector, spec.field): spec for spec in benchmarks}
        registered_k = dict.fromkeys(configured)
        pending_pairs = set(configured)
        if not no_upload:
            with httpx.Client(
                base_url=f"{scoreboard_url.rstrip('/')}/",
                headers={"Authorization": f"Bearer {scoreboard_token}"},
                timeout=30.0,
            ) as client:
                for attempt in range(REQUEST_RETRIES):
                    try:
                        response = client.post("get_tasks", json={"frameworks": ["lighteval"]})
                        response.raise_for_status()
                        payload = response.json()
                        response = client.get("scores")
                        response.raise_for_status()
                        score_rows = response.json()
                        if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
                            raise ValueError("Scoreboard get_tasks response must contain a tasks list")
                        if not isinstance(score_rows, list):
                            raise ValueError("Scoreboard scores response must be a list")
                        break
                    except (httpx.HTTPStatusError, httpx.TransportError) as error:
                        if (
                            isinstance(error, httpx.HTTPStatusError)
                            and error.response.status_code != 429
                            and error.response.status_code < 500
                        ) or attempt + 1 == REQUEST_RETRIES:
                            raise
                        time.sleep(RETRY_DELAY * 2**attempt)
            registered, known, pending_pairs = set(), set(), set()
            for rows, pending in ((score_rows, False), (payload["tasks"], True)):
                for row in rows:
                    if not isinstance(row, dict) or not isinstance(row.get("model"), dict):
                        raise ValueError("invalid model in Scoreboard manifest")
                    remote_model = row["model"]
                    if not isinstance(remote_model["is_preview"], bool):
                        raise ValueError("Scoreboard model is_preview must be a boolean")
                    key = tuple(
                        str(remote_model[name]).casefold() for name in ("arch_version", "data_version", "param_size")
                    ) + (int(remote_model["ctx_len"]), remote_model["is_preview"])
                    metadata = (
                        row.get("benchmark")
                        if pending
                        else {
                            "name": row.get("benchmark_name"),
                            "field": row.get("field"),
                            "avg_k": row.get("avg_k"),
                        }
                    )
                    if not isinstance(metadata, dict):
                        raise ValueError("invalid benchmark in Scoreboard manifest")
                    if pending and metadata.get("framework", "lighteval") != "lighteval":
                        continue
                    name, field = metadata.get("name"), metadata.get("field")
                    if not isinstance(name, str) or not name or not isinstance(field, str) or not field:
                        raise ValueError("invalid benchmark in Scoreboard manifest")
                    pair = name, field
                    registered.add(key)
                    known.add(pair)
                    if pending and key == model.key and pair in configured:
                        pending_pairs.add(pair)
                    if pair in configured:
                        count = metadata.get("avg_k")
                        if (
                            isinstance(count, bool)
                            or not isinstance(count, (int, float))
                            or not math.isfinite(count)
                            or count < 1
                            or not float(count).is_integer()
                        ):
                            raise ValueError(f"benchmark {pair} has no valid registered integer avg_k")
                        if registered_k[pair] is not None and registered_k[pair] != int(count):
                            raise ValueError(f"benchmark {pair} has conflicting registered avg_k values")
                        registered_k[pair] = int(count)
            if model.key not in registered:
                if missing := configured.keys() - known:
                    raise ValueError(
                        f"configured benchmarks are absent from the Scoreboard manifest: {sorted(missing)}"
                    )
                pending_pairs.update(configured)
            if missing := {pair for pair, count in registered_k.items() if count is None}:
                raise ValueError(f"Scoreboard manifest lacks registered avg_k for {sorted(missing)}")

        work = []
        if not no_upload:
            for path in sorted(directory.glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("upload_status") in {"pending_upload", "upload_failed"}:
                    spec = configured.get((payload.get("benchmark_name"), payload.get("field")))
                    if spec is not None:
                        work.append((spec, payload, path))
        registries = {
            spec: RWKVRegistry(tasks=spec.selector, load_multilingual=True)
            for spec in benchmarks
            if (spec.selector, spec.field) in pending_pairs or any(item[0] == spec for item in work)
        }
        tasks = {spec: registry.load_tasks() for spec, registry in registries.items()}
        ordered = sorted(
            (spec for spec in benchmarks if (spec.selector, spec.field) in pending_pairs),
            key=lambda spec: any(SamplingMethod.GENERATIVE in task.sampling_methods for task in tasks[spec].values()),
        )
        LOGGER.info("benchmark order: %s", [spec.selector for spec in ordered])
        work.extend(
            (spec, None, directory / (re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.selector) + ".json")) for spec in ordered
        )
        successful, failures, pending_uploads = [], [], []
        # Restored and newly computed scores share the same publication path.
        for final_attempt in (False, True):
            for item in pending_uploads if final_attempt else work:
                if final_attempt:
                    score, spec, payload, path, restored = item
                else:
                    spec, payload, path = item
                    pair = spec.selector, spec.field
                    restored = payload is not None
                    if not restored and pair not in pending_pairs:
                        continue
                    task = next(iter(tasks[spec].values()))
                    mode, budget = task.cot_mode, task.generation_size
                    template = None if mode == "NoCoT" else prompt_template or task.config.prompt_template
                    LOGGER.info(
                        "benchmark mode: selector=%s cot_mode=%s max_generated_tokens=%d", spec.selector, mode, budget
                    )
                    config = _litelm_model(
                        endpoint,
                        models,
                        mode,
                        template,
                        budget,
                        seed,
                        cache_dir=str(
                            Path(output_dir) / ".lighteval_cache" / ("preview" if endpoint.is_preview else "release")
                        ),
                    )
                    pipeline = RWKVPipeline(
                        spec.selector,
                        PipelineParameters(
                            launcher_type=ParallelismManager.NONE,
                            max_samples=effective_max_samples,
                            max_total_samples=effective_max_samples if test_mode else None,
                            load_tasks_multilingual=True,
                            streaming_evaluation=True,
                            remove_reasoning_tags=False,
                        ),
                        EvaluationTracker(output_dir=output_dir, save_details=False, save_streaming_completions=True),
                        model=RWKVClient(config),
                        avg_k=registered_k[pair],
                        registry=registries[spec],
                    )
                    if registered_k[pair] is not None:
                        pipeline._configure_sampling_counts()
                    if not restored:
                        pipeline.evaluate()
                        pipeline.show_results()
                    identity = {
                        "model_hash": pipeline.model._cache.model_hash,
                        "task_hashes": {
                            name: pipeline.model._cache._get_task_hash(name) for name in sorted(pipeline.tasks_dict)
                        },
                        "max_samples": effective_max_samples,
                        "test_mode": test_mode,
                    }
                    if restored:
                        if (
                            payload.get("score_contract") != "rwkv_avg_at_k_v1"
                            or payload.get("metric_type") != "average"
                            or payload.get("metric_k") != payload.get("avg_k")
                        ):
                            raise ValueError(f"stored score requires explicit replay: {path}")
                        score = Score(
                            **{
                                **{field.name: payload[field.name] for field in fields(Score)},
                                "model": RwkvModel(**payload["model"]),
                                **{
                                    name: [
                                        Detail(
                                            **{
                                                **detail,
                                                "sampling_config": SamplingConfig(**detail["sampling_config"]),
                                            }
                                        )
                                        for detail in payload[name]
                                    ]
                                    for name in ("passed_details", "wrong_details", "failed_details")
                                },
                            }
                        )
                        if payload.get("model_name") != endpoint.model_name or score.model.key != model.key:
                            raise ValueError(f"stored score has a different model identity: {path}")
                        if payload.get("cot_mode") != mode or score.avg_k != registered_k[pair]:
                            raise ValueError(f"stored score has a different evaluation configuration: {path}")
                        if payload.get("run_identity") != identity:
                            raise ValueError(f"stored score has a different run identity: {path}")
                    else:
                        score = Score.from_pipeline(
                            pipeline,
                            spec.selector,
                            _sampling_config(mode, budget, seed),
                            cot_mode=mode,
                            is_preview=endpoint.is_preview,
                        )
                        if registered_k[pair] is not None and score.avg_k != registered_k[pair]:
                            raise RuntimeError(f"{spec.selector}: actual avg_k differs from registered avg_k")
                        payload = {
                            "field": spec.field,
                            "upload_status": "local_only" if no_upload else "pending_upload",
                            "model_name": endpoint.model_name,
                            "metric_k": score.avg_k,
                            "score_contract": "rwkv_avg_at_k_v1",
                            "cot_mode": mode,
                            "run_identity": identity,
                        }
                        if template is not None:
                            payload["prompt_template"] = template
                        score.save(path, payload)
                        successful.append(score)
                        pipeline.save_and_push_results()
                    pending_pairs.discard(pair)
                    if no_upload:
                        continue
                try:
                    upload_score(
                        score,
                        field=spec.field,
                        cot_mode=payload["cot_mode"],
                        token=scoreboard_token or "",
                        api_url=scoreboard_url,
                    )
                except (httpx.HTTPStatusError, httpx.TransportError) as error:
                    rejected = (
                        isinstance(error, httpx.HTTPStatusError)
                        and error.response.status_code != 429
                        and error.response.status_code < 500
                    )
                    if rejected or final_attempt:
                        payload["upload_status"] = "upload_failed"
                        score.save(path, payload)
                        if rejected:
                            raise
                        failures.append(score)
                    else:
                        pending_uploads.append((score, spec, payload, path, restored))
                else:
                    payload["upload_status"] = "uploaded"
                    score.save(path, payload)
                    if restored and not final_attempt:
                        successful.append(score)
                    score.passed_details.clear()
                    score.wrong_details.clear()
                    score.failed_details.clear()
        return successful, failures
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run RWKV through the LightEval pipeline.")
    parser.add_argument("--models", type=Path, default=root / "configs/models.toml")
    parser.add_argument("--benchmarks", type=Path, default=root / "configs/benchmarks.toml")
    parser.add_argument("--prompt-template", choices=("bot", "assistant", "function_calling"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--test",
        "--test-mode",
        dest="test_mode",
        action="store_true",
        help=f"test mode: evaluate at most {TEST_MODE_MAX_SAMPLES} questions per benchmark",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--scoreboard-url")
    parser.add_argument("--scoreboard", choices=("production", "test"), help="select credentials from the environment")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--model-name", help="evaluate only this configured weight name")
    parser.add_argument("--model-context", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--model-release", choices=("preview", "release"), help=argparse.SUPPRESS)
    parser.add_argument("--launch", action="store_true", help="start independently restarting systemd services")
    parser.add_argument("--environment-file", type=Path, default=Path("/etc/rwkv-eval/.env"))
    parser.add_argument("--service-user", default=os.environ.get("SUDO_USER") or pwd.getpwuid(os.getuid()).pw_name)
    parser.add_argument("--unit-prefix", default="rwkv-eval")
    parser.add_argument("--offline", action="store_true", help="launch services with Hugging Face offline mode")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    service = bool(os.environ.get("INVOCATION_ID"))
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if args.offline and not args.launch:
        raise SystemExit("--offline requires --launch; set HF_HUB_OFFLINE=1 for foreground runs")
    try:
        models = read_models(args.models, resolve_api_keys=not args.launch)
        benchmarks = read_benchmarks(args.benchmarks)
        models = tuple(
            model
            for model in models
            if (args.model_name is None or model.model_name == args.model_name)
            and (args.model_context is None or model.ctx_len == args.model_context)
            and (args.model_release is None or model.is_preview == (args.model_release == "preview"))
        )
        if not models:
            raise ValueError("no configured model matches the requested identity")
        if args.scoreboard and args.scoreboard_url is not None:
            raise ValueError("--scoreboard cannot be combined with --scoreboard-url")
        if args.launch:
            root = Path(__file__).resolve().parents[2]
            python = root / ".venv/bin/python"
            if not python.is_file():
                raise ValueError(f"project Python is missing: {python}")
            if not args.environment_file.is_absolute():
                raise ValueError("--environment-file must be an absolute path")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.unit_prefix):
                raise ValueError("--unit-prefix may contain only letters, digits, dots, underscores, and hyphens")
            groups = dict.fromkeys((model.model_name, model.ctx_len, model.is_preview) for model in models)
            failures = []
            for model_name, ctx_len, is_preview in groups:
                output_dir = Path(
                    args.output_dir.replace(
                        "{param_size}", RwkvModel.from_model_name(model_name, ctx_len, is_preview).param_size
                    )
                ).resolve()
                identity = f"{model_name}|{ctx_len}|{is_preview}"
                suffix = hashlib.sha256(identity.encode()).hexdigest()[:8]
                name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(model_name).name)[:100]
                unit = f"{args.unit_prefix}-{name}-{ctx_len}-{'preview' if is_preview else 'release'}-{suffix}"
                command = ([] if os.geteuid() == 0 else ["sudo", "-n"]) + [
                    "/usr/bin/systemd-run",
                    f"--unit={unit}",
                    f"--property=User={args.service_user}",
                    f"--property=WorkingDirectory={root}",
                    f"--property=EnvironmentFile={args.environment_file}",
                    "--property=Restart=on-failure",
                    "--property=RestartSec=30s",
                    "--property=RestartPreventExitStatus=78",
                    "--property=LimitNOFILE=65536",
                    "--property=NoNewPrivileges=yes",
                    "/usr/bin/timeout",
                    "--kill-after=60s",
                    "604800s",
                    "/usr/bin/env",
                    "PYTHONUNBUFFERED=1",
                    f"PYTHONPATH={root / 'src'}",
                ]
                if args.offline:
                    command.append("HF_HUB_OFFLINE=1")
                command.extend(
                    [
                        str(python),
                        "-m",
                        "src.rwkv_eval.main",
                        "--models",
                        str(args.models.resolve()),
                        "--benchmarks",
                        str(args.benchmarks.resolve()),
                        "--model-name",
                        model_name,
                        "--model-context",
                        str(ctx_len),
                        "--model-release",
                        "preview" if is_preview else "release",
                        "--output-dir",
                        str(output_dir),
                        "--seed",
                        str(args.seed),
                        "--verbose",
                    ]
                )
                if args.prompt_template is not None:
                    command.extend(["--prompt-template", args.prompt_template])
                if args.max_samples is not None:
                    command.extend(["--max-samples", str(args.max_samples)])
                if args.test_mode:
                    command.append("--test")
                if args.no_upload:
                    command.append("--no-upload")
                if args.scoreboard_url is not None:
                    command.extend(["--scoreboard-url", args.scoreboard_url])
                if args.scoreboard is not None:
                    command.extend(["--scoreboard", args.scoreboard])
                try:
                    description = (
                        "rwkv-eval:"
                        + hashlib.sha256(
                            json.dumps(command[command.index("/usr/bin/systemd-run") :]).encode()
                        ).hexdigest()
                    )
                    command.insert(command.index("/usr/bin/systemd-run") + 1, f"--description={description}")
                    status = subprocess.run(
                        [
                            "/usr/bin/systemctl",
                            "show",
                            f"{unit}.service",
                            "--property=LoadState",
                            "--property=ActiveState",
                            "--property=Description",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    properties = dict(line.split("=", 1) for line in status.stdout.splitlines() if "=" in line)
                    if properties.get("LoadState") != "not-found":
                        if properties.get("Description") != description:
                            raise RuntimeError(f"{unit}.service already exists with different launch settings")
                        if properties.get("ActiveState") in {"active", "activating", "reloading"}:
                            print(f"Already running {unit}.service", flush=True)
                            continue
                    if properties.get("LoadState") != "not-found":
                        start = ([] if os.geteuid() == 0 else ["sudo", "-n"]) + [
                            "/usr/bin/systemctl",
                            "start",
                            f"{unit}.service",
                        ]
                        subprocess.run(start, check=True, timeout=30)
                        print(f"Resumed {unit}.service", flush=True)
                        continue
                    subprocess.run(command, check=True, timeout=30)
                except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                    failures.append(f"{unit}: {error}")
                    continue
                print(f"Submitted {unit}.service", flush=True)
            if failures:
                raise RuntimeError("could not start all model services: " + "; ".join(failures))
            return 0
        if args.scoreboard:
            suffix = "_TEST" if args.scoreboard == "test" else ""
            scoreboard_token = os.environ.get(f"SCOREBOARD_PUBLICATION_TOKEN{suffix}")
            scoreboard_url = os.environ.get(f"SCOREBOARD_API_BASE_URL{suffix}")
            if not args.no_upload and (not scoreboard_token or not scoreboard_url):
                raise ValueError(f"missing {args.scoreboard} Scoreboard environment variables")
            if scoreboard_url:
                scoreboard_url = scoreboard_url.rstrip("/")
                if not scoreboard_url.endswith("/api"):
                    scoreboard_url += "/api"
        else:
            scoreboard_token = os.environ.get("SCOREBOARD_API_TOKEN")
            scoreboard_url = args.scoreboard_url or os.environ.get(
                "SCOREBOARD_API_URL", "https://eval.rwkv.rs/test/api"
            )
    except (OSError, ValueError) as error:
        if service:
            LOGGER.error("evaluation configuration failed: %s", error)
            return 78
        if args.launch:
            LOGGER.error("%s", error)
            return 1
        raise
    except RuntimeError as error:
        if args.launch:
            LOGGER.error("%s", error)
            return 1
        raise
    try:
        scores, failures = evaluate(
            models,
            benchmarks,
            prompt_template=args.prompt_template,
            max_samples=args.max_samples,
            test_mode=args.test_mode,
            seed=args.seed,
            scoreboard_token=scoreboard_token,
            scoreboard_url=scoreboard_url or "https://eval.rwkv.rs/test/api",
            no_upload=args.no_upload,
            output_dir=args.output_dir,
        )
    except Exception as error:
        if service:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None) or getattr(error, "status_code", None)
            if status in {401, 403}:
                LOGGER.error("evaluation authentication failed; check inference and Scoreboard credentials")
                return 78
            if isinstance(error, (ValueError, BlockingIOError)) or (
                isinstance(error, httpx.HTTPStatusError) and status != 429 and status < 500
            ):
                LOGGER.error("%s", error)
                return 78
            from huggingface_hub import constants
            from huggingface_hub.errors import OfflineModeIsEnabled

            missing_file = isinstance(error, FileNotFoundError) or (
                isinstance(error, ConnectionError)
                and isinstance(error.__context__, (FileNotFoundError, OfflineModeIsEnabled))
            )
            if constants.HF_HUB_OFFLINE and missing_file:
                LOGGER.error("required file is missing in offline mode: %s", error)
                return 78
        raise
    if failures:
        LOGGER.error("%d score uploads remain pending", len(failures))
        return 1
    LOGGER.info("completed %d scores", len(scores))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
