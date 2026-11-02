from __future__ import annotations

import json
import math
import re
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
from src.rwkv_eval.configs import RwkvModel, SamplingConfig


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


def _stored_model_config(record: dict[str, Any]) -> Any:
    config = record.get("config_general")
    return config.get("model_config") if isinstance(config, dict) else None


def _stored_model_name(record: dict[str, Any]) -> str:
    config = _stored_model_config(record)
    if isinstance(config, dict) and isinstance(config.get("model_name"), str):
        return config["model_name"]
    if isinstance(config, str) and (
        match := re.search(r"(?:['\"])?model_name(?:['\"])?\s*[:=]\s*['\"]([^'\"]+)", config)
    ):
        return match.group(1)
    if isinstance(name := record.get("model_name"), str):
        return name
    raise ValueError("results metadata does not contain a model_name")


def _stored_context_length(record: dict[str, Any]) -> int:
    config = _stored_model_config(record)
    if isinstance(config, dict):
        for key in ("max_model_length", "ctx_len"):
            if isinstance(value := config.get(key), int) and not isinstance(value, bool) and value > 0:
                return value
    elif isinstance(config, str) and (
        match := re.search(r"(?:['\"])?(?:max_model_length|ctx_len)(?:['\"])?\s*[:=]\s*(\d+)", config)
    ):
        return int(match.group(1))
    return record.get("ctx_len", 4096) if isinstance(record.get("ctx_len", 4096), int) else 4096


def _parse_stored_model(record: dict[str, Any]) -> RwkvModel:
    model_name = Path(_stored_model_name(record)).name.removesuffix(".pth").lower()
    parts = model_name.split("-")
    if len(parts) < 3 or parts[0] not in {"rwkv7", "rwkv7a", "rwkv7b"}:
        raise ValueError(f"cannot parse RWKV model name: {model_name}")
    if not re.fullmatch(r"g1[a-z0-9]*", parts[1]) or parts[2] not in {"1.5b", "2.9b", "7.2b", "13.3b"}:
        raise ValueError(f"invalid RWKV model name: {model_name}")
    context_length = _stored_context_length(record)
    for part in parts[3:]:
        if match := re.fullmatch(r"ctx(\d+)", part):
            context_length = int(match.group(1))
            break
    return RwkvModel(parts[0], parts[1].capitalize(), parts[2], context_length, record["is_preview"])


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


@dataclass
class Detail:
    messages: list[dict[str, str]]
    sampling_config: SamplingConfig
    answer: str
    ground_truth: str
    is_passed: bool


def get_score(file_path: str | Path) -> Score:
    """Read one LightEval result and its representative details."""
    path = Path(file_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    task_names = [name for name in record["results"] if name != "all" and ":_average|" not in name]
    if not task_names or set(record["config_tasks"]) != set(task_names):
        raise ValueError(f"{path}: results and config_tasks must contain the same evaluated tasks")
    results_root = next(parent for parent in path.parents if parent.name == "results")
    date_id = path.stem.removeprefix("results_")
    rows, buckets = [], ([], [], [])
    for task_name in task_names:
        task_config, task_metrics = record["config_tasks"][task_name], record["results"][task_name]
        metric_names = [
            name
            for name in task_metrics
            if not name.endswith("_stderr")
            and name not in {"n_samples", "n_completions", "n_truncated", "truncation_rate"}
        ]
        if metric_names != ["avg@n"]:
            raise ValueError(f"{path}: {task_name} must contain only the avg@n metric")
        num_samples = task_config["effective_num_docs"]
        avg_k = record["summary_tasks"][task_name]["n_completions"] / num_samples
        if avg_k < 1 or not float(avg_k).is_integer():
            raise ValueError(f"{path}: avg_k must be a positive integer")
        truncation_rate = task_metrics["truncation_rate"]
        if not math.isfinite(truncation_rate) or not 0 <= truncation_rate <= 1:
            raise ValueError(f"{path}: results must contain a finite truncation_rate in [0, 1]")
        rows.append((num_samples, avg_k, float(task_metrics["avg@n"]), truncation_rate))
        detail_path = (
            results_root.parent
            / "details"
            / path.parent.relative_to(results_root)
            / date_id
            / f"details_{task_name}_{date_id}.parquet"
        )
        for target, source in zip(
            buckets,
            get_eg_details(detail_path, no_cot=record["cot_mode"] == "NoCoT", task_name=task_name),
        ):
            target.extend(source[: 20 - len(target)])
    if any(row[1] != rows[0][1] for row in rows):
        raise ValueError(f"{path}: subtasks must have the same avg_k")
    total = sum(row[0] for row in rows)
    return Score(
        _parse_stored_model(record),
        record["benchmark_name"],
        total,
        float(rows[0][1]),
        sum(count * score for count, _, score, _ in rows) / total,
        sum(count * rate for count, _, _, rate in rows) / total,
        *buckets,
    )


def _stored_metric_score(metric: Any) -> float:
    # Prototype: main._native_score. Native grouped metrics retain their
    # declared order, including NumPy boolean MC1 values before MC2 floats.
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
        answer = extract_answer(raw_answer, extractor, len(doc.choices), rendered_prompt=response.rendered_prompt)
        if extractor in {"math", "short_answer"}:
            # Prototype: rwkv_prompt already extracts MATH's balanced gold box.
            # Re-extracting that concise value can reject text/multiline LaTeX
            # or turn a complete expression into its last numeric token. Keep
            # the exact gold used by the native metric, also on parquet replay.
            if doc.task_name.split("|", 1)[0].split(":", 1)[0] != "math":
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
        "avg_k": score.avg_k,
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
        response = client.post("upload", json=payload)
        response.raise_for_status()
        return response.json()
