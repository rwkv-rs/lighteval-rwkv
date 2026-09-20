# MIT License
"""Publish completed RWKV benchmark tasks to Scoreboard."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from urllib.parse import quote, urlsplit

import httpx
import pyarrow.parquet as pq

from lighteval.metrics.normalizations import normalize_log_probs
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod


MAX_SAMPLES_PER_OUTCOME = 20
MAX_COMPRESSED_BYTES = 128 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_TASK_CONFIG_FIELDS = ("num_fewshots", "generation_size", "stop_sequence", "original_num_docs", "effective_num_docs")
_ENVIRONMENT_FIELDS = (
    "served_model_name",
    "model_revision",
    "vllm_version",
    "pool_fingerprint",
    "max_model_length",
    "prompt_template",
)
_FIELD_MARKER = re.compile(r"field:([a-z][a-z0-9_-]{0,63})")
_SCOREBOARD_COT_MODES = {"no_cot": "NoCoT", "fake_think": "FakeCoT", "open_think": "CoT"}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Rollout:
    doc: Doc
    task_name: str
    document_index: int
    repeat_id: int
    response: ModelResponse
    extracted_answer: str
    score: float
    outcome: str
    is_logprob: bool = False


@dataclass
class _TaskAccumulator:
    score_sum: float = 0.0
    count: int = 0
    truncated: int = 0
    outcome_totals: dict[str, int] = field(default_factory=lambda: {"correct": 0, "incorrect": 0, "unanswered": 0})
    selected: dict[str, list[_Rollout]] = field(
        default_factory=lambda: {"correct": [], "incorrect": [], "unanswered": []}
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        default=lambda item: item.item(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _finalized_task_matches(receipt: dict, identity: str, publication_sha256: str) -> bool:
    """Check that an idempotent retry describes the already finalized task."""
    return receipt.get("task_hashes", {}).get(identity) == publication_sha256


def _campaign_run_key(campaign: dict) -> str:
    normalized = dict(campaign)
    normalized.pop("run_key", None)
    for name in ("configured_benchmarks", "resolved_benchmarks", "skipped_benchmarks"):
        normalized[name] = sorted(normalized[name])
    tasks = []
    for value in normalized["expected_tasks"]:
        task = dict(value)
        for name in ("evaluation_splits", "languages", "tags"):
            task[name] = sorted(task[name])
        tasks.append(task)
    normalized["expected_tasks"] = sorted(tasks, key=lambda task: (task["identity"], _canonical_json(task)))
    return _sha256(normalized)


class ScoreboardCallback:
    """Publish one configured benchmark selector as one finalized campaign.

    Reads each leaf task's rollout facts back from its already-closed, streamed parquet file
    (`tracker.task_details_path`) one row group at a time, rather than holding a whole task's
    details in memory - a task's row groups already cap at `STREAMING_FLUSH_BATCH_SIZE` docs.
    """

    def __init__(self, *, base_url, token, config_path, pipeline, tracker, model, run_mode, rerun_reason=None) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("SCOREBOARD_API_BASE_URL must be an HTTP(S) URL without query or fragment")
        revision = model.config.model_revision
        if len(revision) != 64 or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError("RWKV model_revision must be the lowercase weight SHA-256 for Scoreboard publication")
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._pipeline = pipeline
        self._tracker = tracker
        self._model = model
        self._run_mode = run_mode
        self._cot_mode = self._scoreboard_cot_mode(model.config.cot_mode)
        self._validate_model_name(model.config.model_name)
        self._task_registry_by_name = {
            task["name"]: {"module": module["module"], "docstring": module["docstring"]}
            for module in pipeline.registry.get_tasks_dump()
            for task in module["tasks"]
        }
        self._task_metadata_by_name = {name: entry["docstring"] for name, entry in self._task_registry_by_name.items()}
        self._field_by_selector = self._resolve_task_fields()
        self._config_digest = _sha256(
            {
                "file_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                "max_samples": model.config.max_samples,
            }
        )
        self._rerun_reason = rerun_reason
        self._selector_completed: dict[str, set[str]] = {}
        self.publication_errors: list[tuple[str, str]] = []
        self._publication_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rwkv-scoreboard")
        self._publication_futures = []
        self._publication_errors_lock = Lock()
        preflight = self._request("GET", "/api/v1/evaluation-publication-preflight")
        expected_schema = "scoreboard-v2"
        if preflight.get("schema_version") != expected_schema:
            raise ValueError(
                f"Scoreboard {run_mode} endpoint requires {preflight.get('schema_version')}, expected {expected_schema}"
            )
        self.completed_selectors = self._load_completed_selectors()

    def _completed_campaign_ids(self) -> dict[str, list[tuple[str, int | None]]]:
        """Return completed campaign candidates grouped by task identity."""
        evaluations = self._fetch_completed_evaluations()
        self._canonical_k = self._canonical_k_by_benchmark(evaluations)
        return self._completed_campaign_candidates(evaluations)

    def _fetch_completed_evaluations(self) -> list[dict]:
        try:
            evaluations = self._request("GET", "/api/evaluations?limit=5000").get("evaluations", [])
        except ValueError as error:
            logger.warning("Scoreboard completed-task lookup unavailable: %s", error)
            return []
        if not isinstance(evaluations, list):
            logger.warning("Scoreboard completed-task lookup returned an invalid evaluations list")
            return []
        return [evaluation for evaluation in evaluations if isinstance(evaluation, dict)]

    def _canonical_k_by_benchmark(self, evaluations: list[dict]) -> dict[str, int]:
        benchmark_k: dict[str, list[int]] = {}
        for evaluation in evaluations:
            task = evaluation.get("task")
            benchmark = task.get("benchmark") if isinstance(task, dict) else None
            cot_mode = task.get("cot_mode") if isinstance(task, dict) else None
            k_metric = evaluation.get("k_metric")
            k = k_metric.get("avg@k") if isinstance(k_metric, dict) else None
            if benchmark is not None and cot_mode == self._cot_mode and isinstance(k, int) and k > 0:
                benchmark_k.setdefault(benchmark, []).append(k)
        return {benchmark: min(values) for benchmark, values in benchmark_k.items()}

    def _completed_campaign_candidates(self, evaluations: list[dict]) -> dict[str, list[tuple[str, int | None]]]:
        revision = self._model.config.model_revision
        prefix = f"{revision}:{self._model.config.wkv_mode}:"
        candidates: dict[str, list[tuple[str, int | None]]] = {}
        for evaluation in evaluations:
            candidate = self._completed_campaign_candidate(evaluation, prefix)
            if candidate is None:
                continue
            identity, campaign_id, k = candidate
            candidates.setdefault(identity, []).append((campaign_id, k))
        return candidates

    def _completed_campaign_candidate(self, evaluation: dict, prefix: str) -> tuple[str, str, int | None] | None:
        if evaluation.get("completed_at") is None:
            return None
        task = evaluation.get("task")
        campaign_id = evaluation.get("campaign_id")
        identity = task.get("identity") if isinstance(task, dict) else None
        if not isinstance(campaign_id, str) or not isinstance(identity, str) or not identity.startswith(prefix):
            return None
        selector = identity.removeprefix(prefix)
        if selector not in self._pipeline._selector_tasks:
            return None
        if not isinstance(task, dict) or task.get("cot_mode") != self._cot_mode:
            return None
        k_metric = evaluation.get("k_metric")
        k = k_metric.get("avg@k") if isinstance(k_metric, dict) else None
        return identity, campaign_id, k if isinstance(k, int) else None

    def _load_completed_selectors(self) -> frozenset[str]:
        """Find selectors already finalized for this exact model and evaluation config."""
        campaign_by_identity = self._completed_campaign_ids()
        revision = self._model.config.model_revision
        prefix = f"{revision}:{self._model.config.wkv_mode}:"

        completed = set()
        for identity, candidates in campaign_by_identity.items():
            selector = identity.removeprefix(prefix)
            expected_k = self._canonical_k.get(selector)
            for campaign_id, published_k in candidates:
                if expected_k is None:
                    expected_k = published_k
                if published_k != expected_k:
                    logger.info(
                        "Republishing selector with the current k metric: selector=%s published=%s expected=%s",
                        selector,
                        published_k,
                        expected_k,
                    )
                    continue
                try:
                    campaign = self._request("GET", f"/api/v1/evaluation-campaigns/{quote(campaign_id, safe='')}")
                except ValueError as error:
                    logger.warning("Scoreboard campaign lookup unavailable: campaign=%s error=%s", campaign_id, error)
                    continue
                task_hashes = campaign.get("task_hashes", {})
                if campaign.get("status") == "complete" and isinstance(task_hashes, dict) and identity in task_hashes:
                    completed.add(selector)
                    break
        if completed:
            logger.info("Skipping already published selectors: %s", ", ".join(sorted(completed)))
        return frozenset(completed)

    @staticmethod
    def _scoreboard_cot_mode(cot_mode: str) -> str:
        try:
            return _SCOREBOARD_COT_MODES[cot_mode]
        except KeyError as error:
            raise ValueError(f"unknown RWKV CoT mode: {cot_mode}") from error

    def _expected_k(self, tasks) -> int | None:
        tasks = tuple(tasks)
        documents_dict = getattr(self._pipeline, "documents_dict", {})
        values = {
            doc.num_samples
            for task in tasks
            for doc in documents_dict.get(getattr(task, "full_name", ""), [])
            if isinstance(doc.num_samples, int) and not isinstance(doc.num_samples, bool) and doc.num_samples > 0
        }
        if values:
            return next(iter(values)) if len(values) == 1 else None
        # Keep completed-campaign lookup useful for lightweight callers that do
        # not materialize documents, while requiring the canonical avg@k form.
        metric_values = set()
        for task in tasks:
            metric = task.metrics[0].metric_name
            names = metric if isinstance(metric, (list, tuple)) else (metric,)
            match = re.fullmatch(r"avg@([1-9][0-9]*)", str(names[0])) if names else None
            if match is None:
                return None
            metric_values.add(int(match.group(1)))
        return next(iter(metric_values)) if len(metric_values) == 1 else None

    @staticmethod
    def _extract_task_field(task_name: str, tags: list[str]) -> str:
        markers = [tag for tag in tags if tag.startswith("field:")]
        if len(markers) != 1:
            raise ValueError(f"Scoreboard task {task_name} must have exactly one field:<id> marker")
        match = _FIELD_MARKER.fullmatch(markers[0])
        if match is None:
            raise ValueError(f"Scoreboard task {task_name} has invalid field marker: {markers[0]}")
        return match.group(1)

    def _resolve_task_fields(self) -> dict[str, str]:
        fields = {}
        for selector, task_names in self._pipeline._selector_tasks.items():
            leaf_fields = {
                self._extract_task_field(
                    task.config.name,
                    self._task_metadata_by_name[task.config.name].get("tags", []),
                )
                for task in (self._pipeline.tasks_dict[task_name] for task_name in task_names)
            }
            if len(leaf_fields) != 1:
                raise ValueError(f"Scoreboard selector {selector} has inconsistent field markers")
            fields[selector] = next(iter(leaf_fields))
        return fields

    def _campaign(self, task_metadata: dict) -> dict:
        omitted = {"identity", "weight_sha256", "weight_display_name", "wkv_mode"}
        registry = [{key: value for key, value in task_metadata.items() if key not in omitted}]
        campaign = {
            "schema_version": "scoreboard-v2",
            "source": "lighteval",
            "config_sha256": self._config_digest,
            "registry_sha256": _sha256(registry),
            "contract_sha256": _sha256(
                "scoreboard-v2:lighteval:selector,field,cot_mode,document,rollout,k_metric,score,model_response"
            ),
            "configured_benchmarks": [task_metadata["benchmark"]],
            "resolved_benchmarks": [task_metadata["benchmark"]],
            "skipped_benchmarks": [],
            "expected_tasks": [task_metadata],
            "rerun_reason": self._rerun_reason,
        }
        campaign["run_key"] = _campaign_run_key(campaign)
        return campaign

    @classmethod
    def from_environment(cls, *, variable_suffix: str = "", **kwargs) -> "ScoreboardCallback | None":
        base_url_name = f"SCOREBOARD_API_BASE_URL{variable_suffix}"
        token_name = f"SCOREBOARD_PUBLICATION_TOKEN{variable_suffix}"
        base_url = os.environ.get(base_url_name)
        token = os.environ.get(token_name)
        if base_url is None and token is None:
            return None
        if not base_url or not token:
            raise ValueError(f"{base_url_name} and {token_name} must be set together")
        return cls(base_url=base_url, token=token, rerun_reason=os.environ.get("SCOREBOARD_RERUN_REASON"), **kwargs)

    def __call__(self, task_name: str) -> None:
        selector = self._pipeline._task_selectors[task_name]
        completed = self._selector_completed.setdefault(selector, set())
        completed.add(task_name)
        expected = self._pipeline._selector_tasks[selector]
        if any(expected_task not in completed for expected_task in expected):
            return
        try:
            executor = getattr(self, "_publication_executor", None)
            if executor is None:
                self._publish_selector(selector, expected)
            else:
                future = executor.submit(self._publish_selector, selector, expected)
                self._publication_futures.append((selector, future))
        except ValueError as error:
            self._record_publication_error(selector, error)
        finally:
            del self._selector_completed[selector]

    def _record_publication_error(self, selector: str, error: ValueError) -> None:
        lock = getattr(self, "_publication_errors_lock", None)
        if lock is None:
            lock = Lock()
        with lock:
            self.publication_errors.append((selector, str(error)))
        logger.error("Scoreboard publication deferred: selector=%s error=%s", selector, error)

    def wait(self) -> None:
        """Wait for all queued publications without blocking generation or scoring."""
        executor = getattr(self, "_publication_executor", None)
        if executor is None:
            return
        self._publication_executor = None
        executor.shutdown(wait=True)
        unexpected_error = None
        for selector, future in self._publication_futures:
            try:
                future.result()
            except ValueError as error:
                self._record_publication_error(selector, error)
            except BaseException as error:  # pragma: no cover - preserves unexpected worker failures
                unexpected_error = unexpected_error or error
        self._publication_futures.clear()
        if unexpected_error is not None:
            raise unexpected_error

    close = wait

    def _publish_selector(self, selector, task_names) -> None:
        tasks = [self._pipeline.tasks_dict[task_name] for task_name in task_names]
        task_metadata = self._task_metadata(selector, tasks)
        accumulator = _TaskAccumulator()
        document_offset = 0
        for task, task_name in zip(tasks, task_names):
            path = self._tracker.task_details_path(task_name)
            task_accumulator, document_offset = self._accumulate_task(task, path, document_offset)
            self._merge_accumulators(accumulator, task_accumulator)
        k = self._expected_k(tasks)
        if k is None:
            raise ValueError(f"Scoreboard selector {selector} must use one avg@k metric")
        metric_name = f"avg@{k}"
        score = accumulator.score_sum / accumulator.count if accumulator.count else 0.0
        score_source = self._score_source(tasks)
        selected = [rollout for bucket in accumulator.selected.values() for rollout in bucket]
        samples = [self._sample(index, rollout, metric_name) for index, rollout in enumerate(selected)]
        outcome_uploaded = {outcome: len(bucket) for outcome, bucket in accumulator.selected.items()}
        completions = accumulator.count
        campaign = self._campaign(task_metadata)
        receipt = self._request(
            "POST",
            "/api/v1/evaluation-campaigns",
            campaign,
            f"campaign:{campaign['run_key']}",
        )
        campaign_id = receipt["campaign_id"]
        publication = {
            "schema_version": "scoreboard-v2",
            "campaign_id": campaign_id,
            "task": task_metadata,
            "result_files": [],
            "task_config": self._task_config(tasks),
            "environment": {
                "framework": "lighteval",
                "lighteval_sha": self._tracker.general_config_logger.lighteval_sha,
                **{name: getattr(self._model.config, name) for name in _ENVIRONMENT_FIELDS},
            },
            "sampling_config": self._sampling_config(tasks),
            "k_metric": {"avg@k": k},
            "score": score,
            "diagnostics": {
                "documents_total": sum(len(self._pipeline.documents_dict[task.full_name]) for task in tasks),
                "completions_total": completions,
                "samples_total": completions,
                "samples_uploaded": len(samples),
                "outcome_totals": accumulator.outcome_totals,
                "outcome_uploaded": outcome_uploaded,
                "truncated_completions": accumulator.truncated,
                "truncation_rate": accumulator.truncated / completions if completions else 0.0,
                "score_metric": score_source,
            },
            "samples": samples,
        }
        identity = task_metadata["identity"]
        publication_sha256 = _sha256(publication)
        if receipt.get("status") == "complete":
            if not _finalized_task_matches(receipt, identity, publication_sha256):
                raise ValueError(
                    f"Scoreboard task is finalized with different content: campaign={campaign_id} task={selector}"
                )
            logger.info(
                "Scoreboard task already finalized: campaign=%s task=%s",
                campaign_id,
                selector,
            )
            return
        path = f"/api/v1/evaluation-campaigns/{campaign_id}/tasks/{quote(identity, safe='')}"
        self._request("PUT", path, publication, f"publish:{publication_sha256}")
        self._finalize_campaign(campaign_id)

    def _finalize_campaign(self, campaign_id: str) -> None:
        self._request(
            "POST",
            f"/api/v1/evaluation-campaigns/{campaign_id}/finalize",
            idempotency_key=f"finalize:{campaign_id}",
        )

    def _task_metadata(self, selector, tasks) -> dict:
        configs = [task.config for task in tasks]
        revision = self._model.config.model_revision
        versions = sorted({str(config.version) for config in configs})
        repositories = {config.hf_repo for config in configs}
        subsets = {config.hf_subset for config in configs}
        metadata = [self._task_metadata_by_name[config.name] for config in configs]
        tags = {tag for values in metadata for tag in values.get("tags", [])}
        task_metadata = {
            "identity": f"{revision}:{self._model.config.wkv_mode}:{selector}",
            "weight_sha256": revision,
            "weight_display_name": self._model.config.model_name,
            "wkv_mode": self._model.config.wkv_mode,
            "cot_mode": self._cot_mode,
            "benchmark": selector,
            "task_name": selector,
            "field": self._field_by_selector[selector],
            "task_version": ",".join(versions),
            "dataset": next(iter(repositories)) if len(repositories) == 1 else None,
            "subset": next(iter(subsets)) if len(subsets) == 1 else None,
            "evaluation_splits": sorted({split for config in configs for split in config.evaluation_splits}),
            "languages": sorted({language for values in metadata for language in values.get("languages", [])}),
            "tags": sorted(tag for tag in tags if not tag.startswith("field:")),
        }
        return task_metadata

    def _task_config(self, tasks) -> dict:
        configs = [task.config for task in tasks]
        values = {}
        for name in _TASK_CONFIG_FIELDS:
            field_values = [getattr(config, name) for config in configs]
            if name in {"original_num_docs", "effective_num_docs"}:
                values[name] = sum(field_values)
            elif name == "generation_size":
                configured_sizes = [value for value in field_values if value is not None]
                values[name] = max(configured_sizes) if configured_sizes else None
            elif name == "stop_sequence":
                values[name] = list(dict.fromkeys(stop for value in field_values for stop in value))
            else:
                unique = list(dict.fromkeys(field_values))
                values[name] = unique[0] if len(unique) == 1 else unique
        values["skipped_multiselect_docs"] = values["original_num_docs"] - values["effective_num_docs"]
        return values

    @staticmethod
    def _score_source(tasks):
        names = []
        for task in tasks:
            for metric in task.metrics:
                if getattr(metric, "category", None) == SamplingMethod.GENERATIVE:
                    name = metric.metric_name
                    names.extend(name if isinstance(name, (list, tuple)) else [name])
        if not names:
            for task in tasks:
                for metric in task.metrics:
                    if getattr(metric, "category", None) == SamplingMethod.LOGPROBS:
                        name = metric.metric_name
                        names.extend(name if isinstance(name, (list, tuple)) else [name])
        unique = list(dict.fromkeys(str(name) for name in names))
        return unique[0] if len(unique) == 1 else unique

    def _sampling_config(self, tasks) -> dict:
        parameters = dict(self._model._generation_parameters)
        # CoT intensity is a backend decoding option, not a prompt template.
        # Publish it explicitly so NoCoT/FakeCoT/CoT runs remain distinguishable.
        parameters["cot_mode"] = self._model.config.cot_mode
        parameters["chat_template_kwargs"] = {
            "rwkv_prompt_template": self._model.config.prompt_template,
            "rwkv_generation_prompt": self._model.config.cot_mode,
        }
        documents = [doc for task in tasks for doc in self._pipeline.documents_dict[task.full_name]]
        max_new_tokens = max(self._model._completion_limit(doc) for doc in documents)
        parameters.update(seed=42)
        parameters.update(
            max_completion_tokens=max_new_tokens,
            stop=list(dict.fromkeys(stop for doc in documents for stop in self._model._stop_sequences(doc))),
        )
        return parameters

    @staticmethod
    def _validate_model_name(model_name: str) -> None:
        generation_match = re.search(r"-(g\d+[a-z]*)-", model_name, re.IGNORECASE)
        parameter_match = re.search(r"(?:^|-)(\d+(?:\.\d+)?b)(?:-|$)", model_name, re.IGNORECASE)
        if generation_match is None or parameter_match is None:
            raise ValueError("Scoreboard publication requires model_name to contain generation and parameter size")

    @staticmethod
    def _merge_accumulators(target: _TaskAccumulator, source: _TaskAccumulator) -> None:
        target.score_sum += source.score_sum
        target.count += source.count
        target.truncated += source.truncated
        for outcome, total in source.outcome_totals.items():
            target.outcome_totals[outcome] += total
        for outcome, bucket in source.selected.items():
            remaining = MAX_SAMPLES_PER_OUTCOME - len(target.selected[outcome])
            if remaining > 0:
                target.selected[outcome].extend(bucket[:remaining])

    @classmethod
    def _accumulate_task(cls, task, path: Path, document_offset: int) -> tuple[_TaskAccumulator, int]:
        """Stream a task's rollout facts back from its closed parquet file."""
        accumulator = _TaskAccumulator()
        has_generative_metric = cls._has_metric_category(task, SamplingMethod.GENERATIVE)
        is_logprob_task = cls._has_metric_category(task, SamplingMethod.LOGPROBS) and not has_generative_metric
        document_indexes: dict[str, int] = {}
        parquet_file = pq.ParquetFile(path)
        for row_group_index in range(parquet_file.num_row_groups):
            for row in parquet_file.read_row_group(row_group_index).to_pylist():
                doc = Doc(**row["doc"])
                document_key = str(doc.id)
                document_index = document_indexes.setdefault(document_key, document_offset + len(document_indexes))
                response = ModelResponse(**row["model_response"])
                cls._accumulate_detail(
                    task,
                    row,
                    doc,
                    response,
                    document_index,
                    accumulator,
                    has_generative_metric,
                    is_logprob_task,
                )
        return accumulator, document_offset + len(document_indexes)

    @staticmethod
    def _has_metric_category(task, category: SamplingMethod) -> bool:
        return any(getattr(metric, "category", None) == category for metric in task.metrics)

    @classmethod
    def _accumulate_detail(
        cls,
        task,
        row: dict,
        doc: Doc,
        response: ModelResponse,
        document_index: int,
        accumulator: _TaskAccumulator,
        has_generative_metric: bool,
        is_logprob_task: bool,
    ) -> None:
        # Mixed tasks reuse the Doc for both requests; the logprob response has
        # no generated text and must not be counted as the primary generation.
        if has_generative_metric and not response.text:
            return
        specific = row["doc"]["specific"] or {}
        scores = specific.get("rwkv_rollout_scores")
        extracted_answers = specific.get("rwkv_rollout_extracted_answers")
        if scores is None or extracted_answers is None:
            cls._accumulate_missing_facts(
                task, row, doc, response, document_index, accumulator, has_generative_metric, is_logprob_task
            )
            return
        cls._accumulate_generated_rollouts(
            task,
            doc,
            response,
            document_index,
            accumulator,
            scores,
            extracted_answers,
        )

    @classmethod
    def _accumulate_missing_facts(
        cls,
        task,
        row: dict,
        doc: Doc,
        response: ModelResponse,
        document_index: int,
        accumulator: _TaskAccumulator,
        has_generative_metric: bool,
        is_logprob_task: bool,
    ) -> None:
        if has_generative_metric:
            raise ValueError(
                f"details for {task.full_name} are missing producer rollout facts "
                "(rwkv_rollout_scores/rwkv_rollout_extracted_answers)"
            )
        if not is_logprob_task:
            raise ValueError(
                f"details for {task.full_name} are missing producer rollout facts "
                "(rwkv_rollout_scores/rwkv_rollout_extracted_answers)"
            )
        predicted_index, extracted_answer = cls._logprob_prediction(task, doc, response)
        score = cls._logprob_score(task, row.get("metric"))
        gold_indices = doc.gold_index if isinstance(doc.gold_index, (list, tuple)) else (doc.gold_index,)
        outcome = "correct" if predicted_index in gold_indices else "incorrect"
        cls._record_rollout(
            accumulator,
            _Rollout(
                doc=doc,
                task_name=task.full_name,
                document_index=document_index,
                repeat_id=0,
                response=response,
                extracted_answer=extracted_answer,
                score=score,
                outcome=outcome,
                is_logprob=True,
            ),
        )

    @classmethod
    def _accumulate_generated_rollouts(
        cls,
        task,
        doc: Doc,
        response: ModelResponse,
        document_index: int,
        accumulator: _TaskAccumulator,
        scores,
        extracted_answers,
    ) -> None:
        if len(scores) != len(response.text) or len(extracted_answers) != len(scores):
            raise ValueError(f"details for {task.full_name} have inconsistent producer rollout facts")
        for repeat_id in range(len(response.text)):
            rollout_response = response[repeat_id]
            rollout_response.truncated_tokens_count = int(rollout_response.finish_reasons == ["length"])
            score = float(scores[repeat_id])
            extracted_answer = str(extracted_answers[repeat_id])
            cls._record_rollout(
                accumulator,
                _Rollout(
                    doc=doc,
                    task_name=task.full_name,
                    document_index=document_index,
                    repeat_id=repeat_id,
                    response=rollout_response,
                    extracted_answer=extracted_answer,
                    score=score,
                    outcome=cls._outcome(rollout_response, score, extracted_answer),
                ),
            )

    @staticmethod
    def _record_rollout(accumulator: _TaskAccumulator, rollout: _Rollout) -> None:
        accumulator.count += 1
        accumulator.score_sum += rollout.score
        accumulator.truncated += rollout.response.finish_reasons == ["length"]
        accumulator.outcome_totals[rollout.outcome] += 1
        bucket = accumulator.selected[rollout.outcome]
        if len(bucket) < MAX_SAMPLES_PER_OUTCOME:
            bucket.append(rollout)

    @staticmethod
    def _logprob_metric_options(task) -> tuple[object | None, bool]:
        for metric in task.metrics:
            if getattr(metric, "category", None) != SamplingMethod.LOGPROBS:
                continue
            sample_level_fn = getattr(metric, "sample_level_fn", None)
            normalization = getattr(sample_level_fn, "logprob_normalization", None)
            if normalization is None:
                normalization = getattr(sample_level_fn, "log_prob_normalization", None)
            return normalization, bool(getattr(sample_level_fn, "length_normalization", False))
        return None, False

    @classmethod
    def _logprob_choice_scores(cls, task, doc: Doc, response: ModelResponse) -> list[float]:
        choice_count = len(doc.choices)
        if choice_count == 0:
            raise ValueError(f"details for {task.full_name} have no logprob answer choices")
        if len(response.logprobs) < choice_count:
            raise ValueError(f"details for {task.full_name} have fewer logprob scores than answer choices")

        choice_scores = list(response.logprobs[:choice_count])
        unconditioned_scores = response.unconditioned_logprobs
        if unconditioned_scores is None and len(response.logprobs) == choice_count * 2:
            unconditioned_scores = response.logprobs[choice_count:]
        choice_tokens = response.output_tokens[:choice_count]
        normalization, length_normalization = cls._logprob_metric_options(task)

        if normalization is not None:
            if len(choice_tokens) != choice_count:
                raise ValueError(f"details for {task.full_name} have incomplete logprob choice tokens")
            try:
                return normalize_log_probs(
                    normalization,
                    choice_scores,
                    unconditioned_scores,
                    doc.choices,
                    choice_tokens,
                )
            except (AssertionError, IndexError, TypeError, ValueError, ZeroDivisionError) as error:
                raise ValueError(f"details for {task.full_name} have invalid logprob normalization data") from error
        if length_normalization:
            try:
                return [score / len(choice) for score, choice in zip(choice_scores, doc.choices, strict=True)]
            except (TypeError, ValueError, ZeroDivisionError) as error:
                raise ValueError(f"details for {task.full_name} have invalid logprob choice text") from error
        return choice_scores

    @classmethod
    def _logprob_prediction(cls, task, doc: Doc, response: ModelResponse) -> tuple[int, str]:
        choice_scores = cls._logprob_choice_scores(task, doc, response)
        predicted_index = max(range(len(choice_scores)), key=choice_scores.__getitem__)
        return predicted_index, doc.choices[predicted_index]

    @staticmethod
    def _logprob_score(task, metrics) -> float:
        if not isinstance(metrics, dict):
            raise ValueError(f"details for {task.full_name} are missing logprob metric facts")
        names = []
        for metric in task.metrics:
            if getattr(metric, "category", None) == SamplingMethod.LOGPROBS:
                name = metric.metric_name
                names.extend(name if isinstance(name, (list, tuple)) else [name])
        values = [metrics[name] for name in names if name in metrics]
        if not values:
            values = list(metrics.values())
        if not values:
            raise ValueError(f"details for {task.full_name} are missing logprob metric facts")
        value = values[0]
        try:
            if isinstance(value, (list, tuple)):
                if not value:
                    raise ValueError("empty logprob metric")
                value = sum(float(item) for item in value) / len(value)
            score = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"details for {task.full_name} contain an invalid logprob metric") from error
        if not math.isfinite(score):
            raise ValueError(f"details for {task.full_name} contain a non-finite logprob metric")
        return score

    @staticmethod
    def _outcome(response: ModelResponse, score: float, extracted_answer: str) -> str:
        if response.finish_reasons == ["length"]:
            return "unanswered"
        if not extracted_answer.strip():
            return "unanswered"
        if score == 1.0:
            return "correct"
        return "incorrect"

    @staticmethod
    def _sample(index: int, rollout: _Rollout, primary_metric: str) -> dict:
        doc = rollout.doc
        response = rollout.response
        outcome = rollout.outcome
        document_id = str(doc.id)
        problem_id = f"{rollout.task_name}:{document_id}"
        ground_truth = doc.get_golds()
        fail_reason = "answer_mismatch" if outcome == "incorrect" else None
        if outcome == "unanswered":
            fail_reason = (
                "max_tokens_before_final_answer"
                if response.truncated_tokens_count
                else "empty_or_unextractable_answer"
            )
        model_response = {"text": response.text}
        if rollout.is_logprob:
            model_response.update(
                logprobs=response.logprobs,
                argmax_logits_eq_gold=response.argmax_logits_eq_gold,
            )
        sample = {
            "sample_index": index,
            "document_index": rollout.document_index,
            # Answer metadata below contains the UI fields; keep source records minimal.
            "document": {"id": problem_id, "query": doc.query},
            "metrics": {"scoreboard_outcome": outcome, primary_metric: rollout.score},
            "model_response": model_response,
            "answer": {
                "outcome": outcome,
                "problem_id": problem_id,
                "repeat_id": rollout.repeat_id,
                "ground_truth": ground_truth[0]
                if len(ground_truth) == 1
                else json.dumps(ground_truth, ensure_ascii=False),
                "extracted_answer": rollout.extracted_answer,
                "assembled_prompt": response.input
                if isinstance(response.input, str)
                else json.dumps(response.input, ensure_ascii=False),
                # Native logprob scoring selects a supplied continuation; it does not generate
                # a completion, so raw_completion and generated_tokens are intentionally empty.
                "raw_completion": "" if rollout.is_logprob else response.text[0],
                "fail_reason": fail_reason,
                "generated_tokens": 0 if rollout.is_logprob else sum(len(tokens) for tokens in response.output_tokens),
                "latency_ms": None,
            },
        }
        return sample

    def _request(
        self, method: str, path: str, payload: dict | None = None, idempotency_key: str | None = None
    ) -> dict:
        body = _canonical_json(payload) if payload is not None else None
        compressed = gzip.compress(body) if body is not None else None
        if body is not None and (len(body) > MAX_UNCOMPRESSED_BYTES or len(compressed) > MAX_COMPRESSED_BYTES):
            raise ValueError("Scoreboard publication exceeds the server payload limits")
        headers = {"Authorization": f"Bearer {self._token}"}
        if compressed is not None:
            headers.update({"Content-Type": "application/json", "Content-Encoding": "gzip"})
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        try:
            response = httpx.request(
                method, f"{self._base_url}{path}", content=compressed, headers=headers, timeout=60
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as error:
            raise ValueError(f"Scoreboard HTTP {error.response.status_code}: {error.response.text[:65536]}") from error
        except httpx.HTTPError as error:
            raise ValueError(f"Scoreboard request failed: {error}") from error
