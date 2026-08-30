# MIT License
"""Publish completed RWKV benchmark tasks to Scoreboard."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Mapping
from urllib.parse import quote, urlsplit

import httpx

from lighteval.logging.info_loggers import DetailsLogger, MetricsLogger
from lighteval.models.rwkv.http_model import MAX_NEW_TOKENS, PROMPT_TEMPLATES


MAX_SAMPLES_PER_OUTCOME = 20
MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_TASK_CONFIG_FIELDS = ("num_fewshots", "generation_size", "stop_sequence", "original_num_docs", "effective_num_docs")
_ENVIRONMENT_FIELDS = ("served_model_name", "model_revision", "vllm_version", "pool_fingerprint", "max_model_length")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=lambda item: item.item(),
                      sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


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


class TaskCallbackDetailsLogger(DetailsLogger):
    """Call Scoreboard as soon as the last detail for one task is stored."""

    def __init__(self, expected_samples: Mapping[str, int], callback: "ScoreboardCallback") -> None:
        super().__init__()
        self._expected_samples = dict(expected_samples)
        self._callback = callback

    def log(self, task_name, doc, model_response, metrics) -> None:
        super().log(task_name, doc, model_response, metrics)
        # DetailsLogger receives the metric immediately after MetricsLogger does.
        if len(self.details[task_name]) == self._expected_samples[task_name]:
            self._callback(task_name, self.details[task_name])


class ScoreboardCallback:
    """Create one campaign and publish each completed LightEval task."""

    def __init__(self, *, base_url, token, config_path, pipeline, tracker, model, rerun_reason=None) -> None:
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
        self._model_variant = self._model_metadata(model.config.model_name)
        self._tasks = {name: self._task_metadata(task) for name, task in pipeline.tasks_dict.items()}
        benchmarks = sorted({task["benchmark"] for task in self._tasks.values()})
        config_digest = _sha256({"file_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                                 "max_samples": model.config.max_samples})
        omitted = {"identity", "weight_sha256", "weight_display_name", "wkv_mode"}
        registry = [{key: value for key, value in task.items() if key not in omitted} for task in self._tasks.values()]
        registry.sort(key=lambda task: task["task_name"])
        campaign = {
            "schema_version": "scoreboard-v1",
            "source": "lighteval",
            "config_sha256": config_digest,
            "registry_sha256": _sha256(registry),
            "contract_sha256": _sha256("scoreboard-v1:lighteval:document,metrics,model_response"),
            "configured_benchmarks": benchmarks,
            "resolved_benchmarks": benchmarks,
            "skipped_benchmarks": [],
            "expected_tasks": list(self._tasks.values()),
            "rerun_reason": rerun_reason,
        }
        campaign["run_key"] = _campaign_run_key(campaign)
        self._request("GET", "/api/v1/evaluation-publication-preflight")
        receipt = self._request("POST", "/api/v1/evaluation-campaigns", campaign,
                                f"campaign:{campaign['run_key']}")
        self.campaign_id = receipt["campaign_id"]

    @classmethod
    def from_environment(cls, **kwargs) -> "ScoreboardCallback | None":
        base_url = os.environ.get("SCOREBOARD_API_BASE_URL")
        token = os.environ.get("SCOREBOARD_PUBLICATION_TOKEN")
        if base_url is None and token is None:
            return None
        if not base_url or not token:
            raise ValueError("SCOREBOARD_API_BASE_URL and SCOREBOARD_PUBLICATION_TOKEN must be set together")
        return cls(base_url=base_url, token=token, rerun_reason=os.environ.get("SCOREBOARD_RERUN_REASON"),
                   **kwargs)

    def details_logger(self) -> TaskCallbackDetailsLogger:
        expected = {name: len(documents) for name, documents in self._pipeline.documents_dict.items()}
        return TaskCallbackDetailsLogger(expected, self)

    def __call__(self, task_name: str, details: list[DetailsLogger.Detail]) -> None:
        task = self._pipeline.tasks_dict[task_name]
        aggregate = MetricsLogger()
        aggregate.metrics_values[task_name] = self._tracker.metrics_logger.metrics_values[task_name]
        aggregate.aggregate({task_name: task}, bootstrap_iters=0)
        metrics = {name: float(value) for name, value in aggregate.metric_aggregated[task_name].items()}
        selected, outcome_totals = self._select_samples(details, next(iter(metrics)))
        samples = [self._sample(index, detail, outcome) for index, (detail, outcome) in enumerate(selected)]
        outcome_uploaded = {outcome: sum(value == outcome for _, value in selected) for outcome in outcome_totals}
        truncated = sum(detail.model_response.truncated_tokens_count for detail in details)
        completions = sum(len(detail.model_response.text) for detail in details)
        publication = {
            "schema_version": "scoreboard-v1",
            "campaign_id": self.campaign_id,
            "task": self._tasks[task_name],
            "result_files": [],
            "task_config": {**{name: getattr(task.config, name) for name in _TASK_CONFIG_FIELDS}, "k_metrics": next(iter(metrics))},
            "environment": {
                "framework": "lighteval",
                "lighteval_sha": self._tracker.general_config_logger.lighteval_sha,
                **{name: getattr(self._model.config, name) for name in _ENVIRONMENT_FIELDS},
            },
            "sampling_config": self._sampling_config(task),
            "primary_metric": next(iter(metrics)),
            "metrics": metrics,
            "diagnostics": {
                "samples_total": len(details),
                "samples_uploaded": len(samples),
                "outcome_totals": outcome_totals,
                "outcome_uploaded": outcome_uploaded,
                "completions": completions,
                "truncated_completions": truncated,
                "truncation_rate": truncated / completions if completions else 0.0,
            },
            "samples": samples,
            "comparison": self._comparison(task, metrics, len(samples), truncated / completions if completions else 0.0),
        }
        identity = self._tasks[task_name]["identity"]
        path = f"/api/v1/evaluation-campaigns/{self.campaign_id}/tasks/{quote(identity, safe='')}"
        self._request("PUT", path, publication, f"publish:{_sha256(publication)}")

    def finalize(self) -> None:
        self._request("POST", f"/api/v1/evaluation-campaigns/{self.campaign_id}/finalize", idempotency_key=f"finalize:{self.campaign_id}")

    def _task_metadata(self, task) -> dict:
        config = task.config
        task_name = task.full_name
        revision = self._model.config.model_revision
        return {
            "identity": f"{revision}:{self._model.config.wkv_mode}:{task_name}",
            "weight_sha256": revision,
            "weight_display_name": self._model.config.model_name,
            "wkv_mode": self._model.config.wkv_mode,
            "benchmark": config.name,
            "task_name": task_name,
            "task_version": str(config.version),
            "dataset": config.hf_repo or None,
            "subset": config.hf_subset or None,
            "evaluation_splits": sorted(config.evaluation_splits),
            "languages": [],
            "tags": [],
        }

    def _sampling_config(self, task) -> dict:
        config = self._model.config
        parameters = dict(self._model._generation_parameters)
        chat_kwargs = {"rwkv_prompt_template": config.prompt_template, "rwkv_generation_prompt": config.cot_mode}
        parameters.update(max_completion_tokens=MAX_NEW_TOKENS, num_samples=max(task.num_samples), stop=[PROMPT_TEMPLATES[config.prompt_template]], chat_template_kwargs=chat_kwargs)
        return parameters

    @staticmethod
    def _model_metadata(model_name: str) -> dict:
        architecture = "RWKV" if model_name.upper().startswith("RWKV") else "QWEN"
        generation_match = re.search(r"-(g\d+[a-z]*)-", model_name, re.IGNORECASE)
        parameter_match = re.search(r"(?:^|-)(\d+(?:\.\d+)?b)(?:-|$)", model_name, re.IGNORECASE)
        if generation_match is None or parameter_match is None:
            raise ValueError("Scoreboard publication requires model_name to contain generation and parameter size")
        return {"label": f"{architecture} {generation_match.group(1).upper()} {parameter_match.group(1).upper()}", "architecture": architecture, "generation": generation_match.group(1).upper(),
                "parameters": parameter_match.group(1).upper()}

    def _comparison(self, task, metrics: dict, samples: int, truncation_rate: float) -> dict:
        precision = self._model.config.wkv_mode
        parameter = self._model_variant["parameters"]
        option = {"id": "precision", "label": "fp16 vs fp32io16", "short_label": "精度",
                  "a_label": "fp16", "b_label": "fp32io16",
                  "contract": "同一 checkpoint 与 generation contract，仅改变 WKV precision。"}
        group = {"id": parameter.lower(), "label": parameter, "a_model": self._model_variant,
                 "b_model": self._model_variant, "parameter_delta_percent": 0.0, "comparable": True}
        return {
            "model": self._model_variant,
            "benchmark": {"label": task.config.name, "categories": [{"id": "benchmark", "label": "Benchmark"}],
                          "evaluation_method": next(iter(metrics)), "score_multiplier": 100.0},
            "evaluation": {
                "prompt_profile": self._model.config.cot_mode,
                "prompt_template": self._model.config.prompt_template,
                "precision": precision,
            },
            "coordinates": [{"comparison": option, "parameter_group": group,
                             "arm": "a" if precision == "fp16" else "b"}],
            "samples": samples,
            "truncation_rate": truncation_rate,
        }

    @classmethod
    def _select_samples(cls, details, primary_metric: str):
        buckets = {"correct": [], "incorrect": [], "unanswered": []}
        for detail in details:
            outcome = cls._outcome(detail, primary_metric)
            buckets[outcome].append(detail)
        selected = [
            (detail, outcome) for outcome, bucket in buckets.items() for detail in bucket[:MAX_SAMPLES_PER_OUTCOME]
        ]
        return selected, {outcome: len(bucket) for outcome, bucket in buckets.items()}

    @staticmethod
    def _outcome(detail: DetailsLogger.Detail, primary_metric: str) -> str:
        response = detail.model_response
        score = detail.metric.get(primary_metric, next(iter(detail.metric.values())))
        if float(score) == 1.0:
            return "correct"
        if not any(text.strip() for text in response.final_text) or response.truncated_tokens_count >= len(response.text):
            return "unanswered"
        if detail.doc.specific and detail.doc.specific.get("rwkv_choice"):
            if len(response.text) == 1 and float(score) == 1.0 / len(detail.doc.choices):
                return "unanswered"
        return "incorrect"

    @staticmethod
    def _sample(index: int, detail: DetailsLogger.Detail, outcome: str) -> dict:
        doc = detail.doc
        response = detail.model_response
        document_id = str(doc.id)
        completions = [text for text in response.text if isinstance(text, str)]
        final_answers = [text for text in response.final_text if isinstance(text, str)]
        ground_truth = doc.get_golds()
        fail_reason = "answer_mismatch" if outcome == "incorrect" else None
        if outcome == "unanswered":
            fail_reason = ("max_tokens_before_final_answer" if response.truncated_tokens_count
                           else "empty_or_unextractable_answer")
        return {
            "sample_index": index,
            "document_index": int(document_id) if document_id.isdigit() else index,
            # Answer metadata below contains the UI fields; keep source records minimal.
            "document": {"id": document_id, "query": doc.query},
            "metrics": {"scoreboard_outcome": outcome, **detail.metric},
            "model_response": {"text": response.text},
            "answer": {
                "outcome": outcome,
                "problem_id": document_id or str(index),
                "repeat_id": 0,
                "ground_truth": ground_truth[0] if len(ground_truth) == 1 else json.dumps(ground_truth, ensure_ascii=False),
                "extracted_answer": "\n\n".join(final_answers),
                "assembled_prompt": response.input if isinstance(response.input, str) else json.dumps(response.input, ensure_ascii=False),
                "raw_completion": "\n\n".join(completions),
                "fail_reason": fail_reason,
                "generated_tokens": sum(len(tokens) for tokens in response.output_tokens),
                "latency_ms": None,
            },
        }

    def _request(self, method: str, path: str, payload: dict | None = None,
                 idempotency_key: str | None = None) -> dict:
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
            response = httpx.request(method, f"{self._base_url}{path}", content=compressed, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as error:
            raise ValueError(f"Scoreboard HTTP {error.response.status_code}: {error.response.text[:65536]}") from error
        except httpx.HTTPError as error:
            raise ValueError(f"Scoreboard request failed: {error}") from error
