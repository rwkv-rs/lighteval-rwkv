from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import pyarrow.parquet as pq

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
    candidates = [response]
    if (model_dump := getattr(response, "model_dump", None)):
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
    messages.append({"role": "assistant", "content": str(answer)})
    return messages


def _stored_model_config(record: dict[str, Any]) -> Any:
    config = record.get("config_general")
    return config.get("model_config") if isinstance(config, dict) else None


def _stored_model_name(record: dict[str, Any]) -> str:
    config = _stored_model_config(record)
    if isinstance(config, dict) and isinstance(config.get("model_name"), str):
        return config["model_name"]
    if isinstance(config, str) and (match := re.search(r"(?:['\"])?model_name(?:['\"])?\s*[:=]\s*['\"]([^'\"]+)", config)):
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
    elif isinstance(config, str) and (match := re.search(r"(?:['\"])?(?:max_model_length|ctx_len)(?:['\"])?\s*[:=]\s*(\d+)", config)):
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
    return RwkvModel(parts[0], parts[1].capitalize(), parts[2], context_length)


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
    if len(task_names) != 1:
        raise ValueError(f"{path}: expected exactly one benchmark, found {len(task_names)}: {task_names}")
    (task_name,) = task_names
    if set(record["config_tasks"]) != {task_name}:
        raise ValueError(f"{path}: results and config_tasks must contain the same single benchmark")
    task_config, task_metrics = record["config_tasks"][task_name], record["results"][task_name]
    truncation_rate = task_metrics.get("truncation_rate")
    if isinstance(truncation_rate, bool) or not isinstance(truncation_rate, (int, float)) or not math.isfinite(truncation_rate) or not 0 <= truncation_rate <= 1:
        raise ValueError(f"{path}: results must contain a finite truncation_rate in [0, 1]")
    metric_name = next(name for name in task_metrics if not name.endswith("_stderr") and name not in {"n_samples", "n_completions", "n_truncated", "truncation_rate"})
    num_samples = task_config["effective_num_docs"]
    summary = record["summary_tasks"][task_name]
    avg_k = summary["n_completions"] / num_samples if summary["n_completions"] else 1
    results_root = next(parent for parent in path.parents if parent.name == "results")
    date_id = path.stem.removeprefix("results_")
    detail_path = results_root.parent / "details" / path.parent.relative_to(results_root) / date_id / f"details_{task_name}_{date_id}.parquet"
    passed, wrong, failed = get_eg_details(detail_path, no_cot=record.get("cot_mode") == "NoCoT")
    return Score(_parse_stored_model(record), task_config["name"], num_samples, float(avg_k), float(task_metrics[metric_name]), float(truncation_rate), passed, wrong, failed)


def get_eg_details(file_path: str | Path, *, no_cot: bool = False) -> tuple[list[Detail], list[Detail], list[Detail]]:
    """Read up to 20 passed, wrong, and failed examples from one parquet file."""
    path = Path(file_path)
    details_root = next(parent for parent in path.parents if parent.name == "details")
    sampling_path = details_root.parent / "results" / path.parent.parent.relative_to(details_root) / f"sampling_config_{path.parent.name}.json"
    sampling_config = get_sampling_config(sampling_path)
    buckets: tuple[list[Detail], list[Detail], list[Detail]] = ([], [], [])
    with pq.ParquetFile(path) as parquet_file:
        for batch in parquet_file.iter_batches(batch_size=64, columns=["doc", "model_response"]):
            for row in batch.to_pylist():
                doc, response, choices = row["doc"], row["model_response"], row["doc"]["choices"]
                gold_indices = doc["gold_index"] if isinstance(doc["gold_index"], list) else [doc["gold_index"]]
                golds = [gold for index in gold_indices for gold in (choices[index] if isinstance(choices[index], list) else [choices[index]])]
                ground_truth = str(golds[0]) if len(golds) == 1 else json.dumps(golds, ensure_ascii=False)
                model_input = doc.get("query", "") if no_cot and isinstance(response.get("input"), str) else response.get("input")
                if no_cot and response.get("logprobs"):
                    choice_scores = response["logprobs"][: len(choices)]
                    predicted = max(range(len(choice_scores)), key=choice_scores.__getitem__)
                    scores, answers, finishes = [float(predicted in gold_indices)], [chr(ord("A") + predicted)], ["stop"]
                elif response["text"]:
                    scores, answers, finishes = doc["specific"]["rwkv_rollout_scores"], doc["specific"]["rwkv_rollout_extracted_answers"], response["finish_reasons"] or ["stop"] * len(doc["specific"]["rwkv_rollout_extracted_answers"])
                else:
                    choice_scores = response["logprobs"][: len(choices)]
                    predicted = max(range(len(choice_scores)), key=choice_scores.__getitem__)
                    scores, answers, finishes = [float(predicted in gold_indices)], [choices[predicted]], ["stop"]
                for score, answer, finish in zip(scores, answers, finishes, strict=True):
                    answer = str(answer)
                    bucket = 2 if finish == "length" or not answer.strip() else int(score != 1)
                    if len(buckets[bucket]) < 20:
                        buckets[bucket].append(Detail(build_uploaded_messages(model_input, answer, fallback_query=doc.get("query", ""), rendered_prompt=rendered_prompt_from_response(response)), sampling_config, answer, ground_truth, bucket == 0))
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
    sampling_config = sampling_config or (details[0].sampling_config if details else None)
    if sampling_config is None:
        raise ValueError("sampling_config is required when the score has no details")
    if any(detail.sampling_config != sampling_config for detail in details):
        raise ValueError("all details must use the uploaded sampling_config")
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
    with httpx.Client(base_url=f"{api_url.rstrip('/')}/", headers={"Authorization": f"Bearer {token}"}, timeout=30.0) as client:
        response = client.post("upload", json=payload)
        response.raise_for_status()
        return response.json()
