# MIT License
"""Publish completed RWKV benchmark tasks to Scoreboard."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx

from lighteval.logging.info_loggers import DetailsLogger
from lighteval.models.model_output import ModelResponse


MAX_SAMPLES_PER_OUTCOME = 20
MAX_COMPRESSED_BYTES = 128 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_TASK_CONFIG_FIELDS = ("num_fewshots", "generation_size", "stop_sequence", "original_num_docs", "effective_num_docs")
_ENVIRONMENT_FIELDS = ("served_model_name", "model_revision", "vllm_version", "pool_fingerprint", "max_model_length")
_FIELD_MARKER = re.compile(r"field:([a-z][a-z0-9_-]{0,63})")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Rollout:
    detail: DetailsLogger.Detail
    task_name: str
    document_index: int
    repeat_id: int
    response: ModelResponse
    extracted_answer: str
    score: float
    outcome: str


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
    """Publish one configured benchmark selector as one finalized campaign."""

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
        self._selector_details: dict[str, dict[str, list[DetailsLogger.Detail]]] = {}
        self.publication_errors: list[tuple[str, str]] = []
        preflight = self._request("GET", "/api/v1/evaluation-publication-preflight")
        expected_schema = "scoreboard-v1"
        if preflight.get("schema_version") != expected_schema:
            raise ValueError(
                f"Scoreboard {run_mode} endpoint requires {preflight.get('schema_version')}, expected {expected_schema}"
            )

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
            "schema_version": "scoreboard-v1",
            "source": "lighteval",
            "config_sha256": self._config_digest,
            "registry_sha256": _sha256(registry),
            "contract_sha256": _sha256(
                "scoreboard-v1:lighteval:selector,field,document,rollout,metrics,model_response"
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

    def __call__(self, task_name: str, details: list[DetailsLogger.Detail]) -> None:
        selector = self._pipeline._task_selectors[task_name]
        selector_details = self._selector_details.setdefault(selector, {})
        selector_details[task_name] = details
        expected = self._pipeline._selector_tasks[selector]
        if any(expected_task not in selector_details for expected_task in expected):
            return
        try:
            self._publish_selector(selector, expected, selector_details)
        except ValueError as error:
            self.publication_errors.append((selector, str(error)))
            logger.error("Scoreboard publication deferred: selector=%s error=%s", selector, error)
        finally:
            del self._selector_details[selector]

    def _publish_selector(self, selector, task_names, details_by_task) -> None:
        tasks = [self._pipeline.tasks_dict[task_name] for task_name in task_names]
        task_metadata = self._task_metadata(selector, tasks)
        rollouts = []
        document_offset = 0
        for task, task_name in zip(tasks, task_names):
            details = details_by_task[task_name]
            rollouts.extend(self._rollouts(task, details, document_offset))
            document_offset += len(details)
        primary_metric = self._primary_metric(tasks)
        metrics = {primary_metric: sum(rollout.score for rollout in rollouts) / len(rollouts)}
        selected, outcome_totals = self._select_samples(rollouts)
        samples = [self._sample(index, rollout, primary_metric) for index, rollout in enumerate(selected)]
        outcome_uploaded = {
            outcome: sum(rollout.outcome == outcome for rollout in selected) for outcome in outcome_totals
        }
        truncated = sum(rollout.response.finish_reasons == ["length"] for rollout in rollouts)
        completions = len(rollouts)
        campaign = self._campaign(task_metadata)
        receipt = self._request(
            "POST",
            "/api/v1/evaluation-campaigns",
            campaign,
            f"campaign:{campaign['run_key']}",
        )
        campaign_id = receipt["campaign_id"]
        publication = {
            "schema_version": "scoreboard-v1",
            "campaign_id": campaign_id,
            "task": task_metadata,
            "result_files": [],
            "task_config": self._task_config(tasks, primary_metric),
            "environment": {
                "framework": "lighteval",
                "lighteval_sha": self._tracker.general_config_logger.lighteval_sha,
                **{name: getattr(self._model.config, name) for name in _ENVIRONMENT_FIELDS},
            },
            "sampling_config": self._sampling_config(tasks),
            "primary_metric": primary_metric,
            "metrics": metrics,
            "diagnostics": {
                "documents_total": sum(len(details_by_task[task_name]) for task_name in task_names),
                "completions_total": completions,
                "samples_total": completions,
                "samples_uploaded": len(samples),
                "outcome_totals": outcome_totals,
                "outcome_uploaded": outcome_uploaded,
                "truncated_completions": truncated,
                "truncation_rate": truncated / completions if completions else 0.0,
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

    def _task_config(self, tasks, primary_metric) -> dict:
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
        values["k_metrics"] = primary_metric
        values["skipped_multiselect_docs"] = values["original_num_docs"] - values["effective_num_docs"]
        return values

    @staticmethod
    def _primary_metric(tasks) -> str:
        names = {
            task.metrics[0].metric_name
            if isinstance(task.metrics[0].metric_name, str)
            else task.metrics[0].metric_name[0]
            for task in tasks
        }
        return next(iter(names)) if len(names) == 1 else "mean"

    def _sampling_config(self, tasks) -> dict:
        config = self._model.config
        parameters = dict(self._model._generation_parameters)
        chat_kwargs = {"rwkv_prompt_template": config.prompt_template, "rwkv_generation_prompt": config.cot_mode}
        documents = [doc for task in tasks for doc in self._pipeline.documents_dict[task.full_name]]
        max_new_tokens = max(self._model._completion_limit(doc) for doc in documents)
        parameters.update(
            num_samples=max(doc.num_samples for doc in documents), seed=42, chat_template_kwargs=chat_kwargs
        )
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
    def _select_samples(rollouts: list[_Rollout]):
        buckets = {"correct": [], "incorrect": [], "unanswered": []}
        for rollout in rollouts:
            buckets[rollout.outcome].append(rollout)
        selected = [rollout for bucket in buckets.values() for rollout in bucket[:MAX_SAMPLES_PER_OUTCOME]]
        return selected, {outcome: len(bucket) for outcome, bucket in buckets.items()}

    @classmethod
    def _rollouts(cls, task, details: list[DetailsLogger.Detail], document_offset: int = 0) -> list[_Rollout]:
        scorer = task.metrics[0].sample_level_fn
        rollouts = []
        for document_index, detail in enumerate(details, start=document_offset):
            for repeat_id in range(len(detail.model_response.text)):
                response = detail.model_response[repeat_id]
                response.truncated_tokens_count = int(response.finish_reasons == ["length"])
                score = float(scorer.score_rollout(detail.doc, response))
                extracted_answer = scorer.extract_rollout_answer(detail.doc, response)
                rollouts.append(
                    _Rollout(
                        detail=detail,
                        task_name=task.full_name,
                        document_index=document_index,
                        repeat_id=repeat_id,
                        response=response,
                        extracted_answer=extracted_answer,
                        score=score,
                        outcome=cls._outcome(response, score, extracted_answer),
                    )
                )
        return rollouts

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
        detail = rollout.detail
        doc = detail.doc
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
        return {
            "sample_index": index,
            "document_index": rollout.document_index,
            # Answer metadata below contains the UI fields; keep source records minimal.
            "document": {"id": problem_id, "query": doc.query},
            "metrics": {"scoreboard_outcome": outcome, primary_metric: rollout.score},
            "model_response": {"text": response.text},
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
                "raw_completion": response.text[0],
                "fail_reason": fail_reason,
                "generated_tokens": sum(len(tokens) for tokens in response.output_tokens),
                "latency_ms": None,
            },
        }

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
