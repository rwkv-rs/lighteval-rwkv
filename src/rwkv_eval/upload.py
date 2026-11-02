from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import httpx
import pyarrow.parquet as pq

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from src.rwkv_eval.answer_extract.free_response import (
    answer_extractor_for_benchmark,
    extract_answer,
    extract_math_answer,
    verify_math_answer,
)
from src.rwkv_eval.configs import RWKVAvgAtN, RwkvModel, SamplingConfig


def _message_dict(message: Any) -> dict[str, Any] | None:
    if isinstance(message, dict):
        return dict(message)
    for name in ("model_dump", "dict"):
        method = getattr(message, name, None)
        if callable(method):
            value = method()
            if isinstance(value, dict):
                return value
    try:
        value = dict(message)
    except (TypeError, ValueError):
        value = None
    if isinstance(value, dict):
        return value
    role = getattr(message, "role", None)
    return {"role": role, "content": getattr(message, "content", "")} if role is not None else None


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item if isinstance(item, str) else str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
            if not isinstance(item, dict) or item.get("type") in (None, "text")
        )
    return str(content)


def rendered_prompt_from_response(response: Any) -> str | None:
    """Read the prompt rendered by the inference server.

    The producer asks vLLM/RWKV for ``prompt_text``.  Uploading that value is
    preferable to duplicating the server's chat-template implementation here.
    """
    rendered_prompt = getattr(response, "rendered_prompt", None)
    if isinstance(rendered_prompt, str):
        return rendered_prompt
    if isinstance(response, dict) and isinstance(response.get("rendered_prompt"), str):
        return response["rendered_prompt"]
    candidates = [response]
    if model_dump := getattr(response, "model_dump", None):
        dumped = model_dump()
        if isinstance(dumped, dict):
            candidates.append(dumped)
    if isinstance(hidden := getattr(response, "_hidden_params", None), dict):
        candidates.append(hidden)
    for candidate in candidates:
        for name in ("prompt_text", "rendered_prompt"):
            value = candidate.get(name) if isinstance(candidate, dict) else getattr(candidate, name, None)
            if isinstance(value, str):
                return value
    return None


def build_uploaded_messages(
    model_input: Any,
    answer: str,
    *,
    assistant_message: str | None = None,
    fallback_query: Any = "",
    rendered_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Return the actual request input followed by the model answer.

    CoT prompts come from the inference response.  This module deliberately
    does not reimplement the RWKV/vLLM chat template.
    """
    if rendered_prompt is not None:
        messages = [{"role": "user", "content": rendered_prompt}]
    elif isinstance(model_input, (list, tuple)):
        messages = [message for item in model_input if (message := _message_dict(item)) is not None]
        if not messages:
            messages = [{"role": "user", "content": _content_text(fallback_query)}]
    else:
        messages = [{"role": "user", "content": _content_text(model_input or fallback_query)}]
    messages.append({"role": "assistant", "content": str(answer if assistant_message is None else assistant_message)})
    return messages


@dataclass
class Score:
    model: RwkvModel
    benchmark_name: str
    num_samples: int
    avg_k: float
    score: float
    truncation_rate: float
    passed_details: list["Detail"]
    wrong_details: list["Detail"]
    failed_details: list["Detail"]
    metric_type: Literal["average"] = "average"

    @classmethod
    def from_pipeline(
        cls, pipeline, benchmark_name: str, sampling: SamplingConfig, *, cot_mode: str, is_preview: bool
    ):
        """Convert native task aggregates and retained details into one benchmark result."""
        rows = []
        buckets = ([], [], [])
        details = pipeline.get_details()
        for name, count in pipeline.task_sample_counts.items():
            if not count:
                continue
            task = pipeline.tasks_dict[name]
            metric = next(
                (metric for metric in task.metrics if isinstance(metric.sample_level_fn, RWKVAvgAtN)), task.metrics[0]
            )
            metric_name = metric.metric_name[0] if isinstance(metric.metric_name, list) else metric.metric_name
            score = float(pipeline.evaluation_tracker.metrics_logger.metric_aggregated[name][metric_name])
            truncated = total = 0
            task_buckets = ([], [], [])
            for native in details.get(name, []):
                response = native.model_response
                for index in range(1 if cot_mode == "NoCoT" else max(len(response.text), 1)):
                    detail, bucket = get_detail(
                        native.doc, response, sampling, native.metric, index, no_cot=cot_mode == "NoCoT"
                    )
                    finish = response.finish_reasons[index] if index < len(response.finish_reasons) else "stop"
                    truncated += int(finish.lower() in {"length", "max_tokens"})
                    total += 1
                    if len(task_buckets[bucket]) < 20:
                        task_buckets[bucket].append(detail)
            for target, source in zip(buckets, task_buckets):
                target.extend(source[: 20 - len(target)])
            completions = pipeline.task_completion_counts.get(name, total)
            truncation = pipeline.task_truncation_counts.get(name, truncated) / completions if completions else 0.0
            rows.append((count, pipeline.task_avg_k[name], score, truncation))
        if not rows:
            raise RuntimeError(f"{benchmark_name}: no evaluated subtasks")
        if any(row[1] != rows[0][1] for row in rows):
            raise RuntimeError(f"{benchmark_name}: inconsistent subtask rollout counts")
        total = sum(row[0] for row in rows)
        config = pipeline.model.config
        return cls(
            model=RwkvModel.from_model_name(
                config.model_name, config.scoreboard_ctx_len or config.max_model_length, is_preview
            ),
            benchmark_name=benchmark_name,
            num_samples=total,
            avg_k=float(rows[0][1]),
            score=sum(count * score for count, _, score, _ in rows) / total,
            truncation_rate=sum(count * rate for count, _, _, rate in rows) / total,
            passed_details=buckets[0],
            wrong_details=buckets[1],
            failed_details=buckets[2],
        )

    def save(self, path: Path, metadata: dict) -> None:
        """Serialize the result atomically, retaining its publication metadata."""
        payload = {**metadata, **asdict(self)}
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


@dataclass
class Detail:
    messages: list[dict[str, str]]
    sampling_config: SamplingConfig
    answer: str
    ground_truth: str
    is_passed: bool


def _stored_metric_score(metric: Any) -> float:
    # Native grouped metrics retain their declared order, including NumPy
    # boolean MC1 values before MC2 floats.
    return float(next(iter(metric.values()), 0.0)) if isinstance(metric, dict) else 0.0


def get_detail(  # noqa: C901
    doc: Doc,
    response: ModelResponse,
    sampling: SamplingConfig,
    metric: dict,
    index: int,
    *,
    no_cot: bool,
) -> tuple[Detail, int]:
    """Prototype: get_eg_details, shared by live and persisted publication.

    No prompt reconstruction and no aggregate-score-to-rollout conversion are
    allowed here.  Raw assistant output is kept separately from the answer.
    """
    gold_indices = doc.gold_index if isinstance(doc.gold_index, list) else [doc.gold_index]
    golds = [str(gold) for gold in doc.get_golds()]
    extractor = answer_extractor_for_benchmark(doc.task_name)
    if no_cot:
        scores = response.logprobs[: len(doc.choices)]
        if len(scores) != len(doc.choices) or len(response.candidate_inputs) != len(scores):
            raise ValueError("NoCoT detail is missing the exact candidate inputs or scores")
        # Prototype: TruthfulqaMCMetrics.compute/mc1. MC1 and MC2 are
        # separate candidate groups, not one multiple-answer question. The
        # published score is the first native metric (MC1); display its exact
        # candidate and gold, while the pipeline still computes both metrics.
        choice_count = (doc.specific or {}).get("len_mc1", len(scores))
        gold_indices = [gold for gold in gold_indices if gold < choice_count]
        selected = max(range(choice_count), key=scores.__getitem__)
        model_input = response.candidate_inputs[selected]
        answer = chr(ord("A") + selected)
        ground_truth = ", ".join(chr(ord("A") + gold) for gold in gold_indices)
        score = float(selected in gold_indices)
        raw_answer = answer
        rendered_prompt = None
        finish = "stop"
    else:
        sampling = replace(
            sampling,
            max_generated_tokens=response.effective_max_new_tokens
            if response.effective_max_new_tokens is not None
            else sampling.max_generated_tokens,
        )
        raw_answer = response.text[index] if response.text else ""
        answer = extract_answer(
            raw_answer,
            extractor,
            len(doc.choices),
            rendered_prompt=response.rendered_prompt,
        )
        if extractor in {"math", "short_answer"}:
            # rwkv_math_prompt already extracts MATH's balanced gold box.
            # Re-extracting that concise value can reject text/multiline LaTeX
            # or turn a complete expression into its last numeric token. Keep
            # the exact gold used by the native metric, also on parquet replay.
            if doc.task_name.split("|", 1)[0].split(":", 1)[0] not in {"math", "gsm8k", "gsm_plus"}:
                golds = [gold if extractor == "short_answer" else extract_math_answer(gold) for gold in golds]
            if any(gold is None for gold in golds):
                raise ValueError("RWKV detail contains an unextractable gold answer")
            score = float(
                any(verify_math_answer(gold, answer or "", allow_text=extractor == "short_answer") for gold in golds)
            )
        elif extractor == "choice":
            score = float(answer is not None and answer in golds)
        else:
            rollout_scores = (doc.specific or {}).get("rwkv_rollout_scores")
            if rollout_scores is not None:
                score = float(rollout_scores[index])
            elif len(response.text) <= 1:
                score = _stored_metric_score(metric)
            else:
                raise ValueError("RWKV detail is missing per-rollout scores")
        ground_truth = golds[0] if len(golds) == 1 else json.dumps(golds, ensure_ascii=False)
        rendered_prompt = rendered_prompt_from_response(response)
        if rendered_prompt is None:
            raise ValueError("generated detail is missing the server-rendered prompt")
        model_input = response.input
        finish = response.finish_reasons[index] if response.finish_reasons else "stop"
    is_truncated = finish.lower() in {"length", "max_tokens"}
    passed = not is_truncated and answer is not None and score == 1.0
    bucket = 2 if is_truncated or answer is None else 0 if passed else 1
    return Detail(
        messages=build_uploaded_messages(
            model_input,
            answer or "",
            assistant_message=raw_answer,
            rendered_prompt=rendered_prompt,
        ),
        sampling_config=sampling,
        answer=answer if answer is not None else "-",
        ground_truth=ground_truth,
        is_passed=passed,
    ), bucket


def get_eg_details(
    file_path: str | Path,
    *,
    no_cot: bool = False,
    task_name: str | None = None,
) -> tuple[list[Detail], list[Detail], list[Detail]]:
    """Read up to 20 details while preserving the stored model inputs.

    NoCoT candidate inputs are produced by the loglikelihood adapter.  Older
    parquet records without them cannot be safely reconstructed and are
    rejected instead of receiving a synthetic prompt.
    """
    path = Path(file_path)
    details_root = next(parent for parent in path.parents if parent.name == "details")
    sampling_path = (
        details_root.parent
        / "results"
        / path.parent.parent.relative_to(details_root)
        / f"sampling_config_{path.parent.name}.json"
    )
    sampling_config = get_sampling_config(sampling_path)
    buckets: tuple[list[Detail], list[Detail], list[Detail]] = ([], [], [])
    with pq.ParquetFile(path) as parquet_file:
        for batch in parquet_file.iter_batches(batch_size=64, columns=["doc", "model_response", "metric"]):
            for row in batch.to_pylist():
                doc = Doc(**row["doc"])
                if task_name is not None:
                    doc.task_name = task_name
                response = ModelResponse(**row["model_response"])
                count = 1 if no_cot else max(len(response.text), 1)
                for index in range(count):
                    detail, bucket = get_detail(doc, response, sampling_config, row["metric"], index, no_cot=no_cot)
                    if len(buckets[bucket]) < 20:
                        buckets[bucket].append(detail)
                if all(len(bucket) == 20 for bucket in buckets):
                    return buckets
    return buckets


def get_sampling_config(file_path: str | Path) -> SamplingConfig:
    sampling_config = SamplingConfig(**json.loads(Path(file_path).read_text(encoding="utf-8")))
    if any(value is None for value in vars(sampling_config).values()):
        raise ValueError(f"{file_path}: sampling configuration fields must not be null")
    return sampling_config


def upload(
    score: Score,
    *,
    field: Literal["knowledge", "reasoning", "maths", "coding", "instruction_following", "agentic", "vision"],
    cot_mode: Literal["NoCoT", "FakeCoT", "CoT"],
    token: str,
    api_url: str = "https://eval.rwkv.rs/test/api",
    sampling_config: SamplingConfig | None = None,
) -> dict[str, Any]:
    """Register a score and upload it to Scoreboard."""
    if (
        score.metric_type != "average"
        or isinstance(score.num_samples, bool)
        or not isinstance(score.num_samples, int)
        or score.num_samples < 1
        or isinstance(score.avg_k, bool)
        or not isinstance(score.avg_k, (int, float))
        or not math.isfinite(score.avg_k)
        or score.avg_k < 1
        or not float(score.avg_k).is_integer()
    ):
        raise ValueError("Scoreboard requires an average metric with positive integer sample and avg_k counts")
    if not math.isfinite(score.score) or not 0 <= score.score <= 1:
        raise ValueError("Scoreboard requires a finite average score in [0, 1]")
    details = score.passed_details + score.wrong_details + score.failed_details
    # Scoreboard has one sampling config per score. Its budget is the
    # ceiling across actual requests; each prompt-dependent value remains in
    # the producer's detail/cache record, without changing message evidence.
    sampling_config = sampling_config or (
        replace(
            details[0].sampling_config,
            max_generated_tokens=max(detail.sampling_config.max_generated_tokens for detail in details),
        )
        if details
        else None
    )
    if sampling_config is None:
        raise ValueError("sampling_config is required when the score has no details")
    if any(
        replace(detail.sampling_config, max_generated_tokens=sampling_config.max_generated_tokens) != sampling_config
        for detail in details
    ):
        raise ValueError("all details must share decoding parameters apart from prompt-dependent budgets")
    payload = {
        "model": {**asdict(score.model), "data_version": score.model.data_version.capitalize()},
        "cot_mode": cot_mode,
        "sampling_config": asdict(sampling_config),
        "score": score.score,
        "truncation_rate": score.truncation_rate,
        "benchmark_name": score.benchmark_name,
        "field": field,
        "num_samples": score.num_samples,
        "avg_k": int(score.avg_k),
        "metric_type": score.metric_type,
        "metric_k": int(score.avg_k),
        **{
            name: [
                {
                    "messages": detail.messages,
                    "answer": detail.answer,
                    "ground_truth": detail.ground_truth,
                    "is_passed": detail.is_passed,
                }
                for detail in getattr(score, name)
            ]
            for name in ("passed_details", "wrong_details", "failed_details")
        },
    }
    with httpx.Client(
        base_url=f"{api_url.rstrip('/')}/", headers={"Authorization": f"Bearer {token}"}, timeout=30.0
    ) as client:
        for attempt in range(6):
            try:
                response = client.post("upload", json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as error:
                if (
                    isinstance(error, httpx.HTTPStatusError)
                    and error.response.status_code != 429
                    and error.response.status_code < 500
                ) or attempt == 5:
                    raise
                time.sleep(2**attempt)
