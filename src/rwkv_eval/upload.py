from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import httpx
import pyarrow.parquet as pq

from src.rwkv_eval.configs import RwkvModel, SamplingConfig


PromptTemplate = Literal["bot", "assistant", "function_calling"]
GenerationPrompt = Literal["fake_think", "open_think"]

_PROMPT_TEMPLATE_ALIASES = {
    "bot": "bot",
    "assistant": "assistant",
    "function_calling": "function_calling",
    "\nBot✿": "bot",
    "\n\nAssistant: ": "assistant",
    "\n### Assistant": "function_calling",
}
# NoCoT is handled by LightEval's loglikelihood/greedy-choice path and is
# intentionally absent here.  These settings are only for generated answers.
_COT_GENERATION_PROMPTS = {
    "FakeCoT": "fake_think",
    "CoT": "open_think",
}


def _stringify_message_content(content: Any) -> str:
    """Match the RWKV chat template's text conversion for API messages."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _normalize_message_content(content: Any, *, collapse_blank: bool = False) -> str:
    """Apply the line normalization used by the deployed RWKV template.

    User turns collapse blank lines; system and assistant turns retain blank
    lines.  Every turn converts CRLF/CR to LF, removes trailing horizontal
    whitespace from each line, and strips trailing whitespace/newlines from
    the resulting content.
    """
    lines = _stringify_message_content(content).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized: list[str] = []
    for source_line in lines:
        line = source_line.rstrip(" \t")
        if collapse_blank and not line:
            continue
        normalized.append(line)
    return "\n".join(normalized).rstrip(" \t\n")


def _message_field(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def _as_message_dict(message: Any) -> dict[str, Any] | None:
    if isinstance(message, dict):
        return dict(message)
    for method_name in ("model_dump", "dict"):
        method = getattr(message, method_name, None)
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
    content = getattr(message, "content", None)
    if role is None:
        return None
    return {"role": role, "content": content}


def _tool_function(tool: Any) -> Any:
    if isinstance(tool, dict):
        return tool.get("function", tool)
    return getattr(tool, "function", tool)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _json_text(value: Any) -> str:
    """Serialize a value like the RWKV template's indented JSON blocks."""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return json.dumps(_stringify_message_content(value), ensure_ascii=False, indent=2)


def _tool_arguments_value(value: Any) -> Any:
    """Normalize OpenAI tool arguments as the deployed service does."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _render_tool_definitions(tools: Sequence[Any]) -> list[str]:
    rendered: list[str] = []
    for tool in tools:
        function = _tool_function(tool)
        name = _message_field(function, "name", "")
        description = _message_field(function, "description", "") or ""
        parameters = _message_field(function, "parameters", {}) or {}
        rendered.append(f"### `{name}`")
        if description:
            rendered.append(f"**Description:** {_stringify_message_content(description)}")
        rendered.extend(("**Parameters:**", "```json", _json_text(_json_value(parameters)), "```"))
    if rendered:
        rendered.extend(
            (
                "To call one of these tools, write exactly this format:",
                "**Tool Call:**",
                "```json",
                '{"name": "tool_name", "arguments": {"key": "value"}}',
                "```",
                "Do not invent tool call IDs or write tool outputs yourself.",
            )
        )
    return rendered


def _generation_suffix(generation_prompt: str | None) -> str:
    if generation_prompt in (None, "open_think", "CoT"):
        return "<think"
    if generation_prompt in ("fake_think", "FakeCoT"):
        return "<think></think"
    raise ValueError(f"unsupported RWKV generation prompt: {generation_prompt!r}")


def render_rwkv_prompt(  # noqa: C901
    messages: Sequence[dict[str, Any]],
    prompt_template: str | None = None,
    generation_prompt: str | None = None,
    *,
    tools: Sequence[Any] = (),
) -> str:
    """Render the textual prompt sent to the RWKV tokenizer.

    The endpoint receives OpenAI-style messages and applies this template on the
    server.  Scoreboard details have no access to ``extra_body``, so the same
    rendering must happen before the messages are uploaded.
    """
    try:
        style = _PROMPT_TEMPLATE_ALIASES[prompt_template or "bot"]
    except KeyError as error:
        raise ValueError(f"unsupported RWKV prompt template: {prompt_template!r}") from error

    messages = list(messages)
    has_tool_history = any(
        _message_field(message, "role", "") == "tool" or bool(_message_field(message, "tool_calls", None))
        for message in messages
    )
    use_tools = style == "function_calling" or bool(tools) or has_tool_history
    suffix = _generation_suffix(generation_prompt)

    if not use_tools:
        parts: list[str] = []
        for message in messages:
            role = _message_field(message, "role", "")
            content = _normalize_message_content(
                _message_field(message, "content", ""),
                collapse_blank=role == "user",
            )
            if style == "bot":
                label = {"system": "System", "user": "User", "assistant": "Bot"}.get(role)
                if label is None:
                    raise ValueError(f"unsupported RWKV chat role: {role!r}")
                parts.append(f"{label}✿{content}✿")
            else:
                label = {"system": "System", "user": "User", "assistant": "Assistant"}.get(role)
                if label is None:
                    raise ValueError(f"unsupported RWKV chat role: {role!r}")
                # The canonical template deliberately retains the space after
                # the colon even for an empty message.
                parts.append(f"{label}: {content}")
        if style == "bot":
            parts.append(f"Bot✿{suffix}")
            return "\n".join(parts)
        parts.append(f"Assistant: {suffix}")
        return "\n\n".join(parts)

    parts: list[str] = []
    pending_system: list[str] = []
    for message in messages:
        role = _message_field(message, "role", "")
        raw_content = _message_field(message, "content", "")
        if role == "system":
            pending_system.append(_normalize_message_content(raw_content))
            continue

        if pending_system or (tools and not parts):
            parts.extend(("### System", *pending_system))
            parts.extend(_render_tool_definitions(tools))
            pending_system.clear()

        if role == "user":
            parts.extend(("### User", _normalize_message_content(raw_content, collapse_blank=True)))
        elif role == "assistant":
            parts.append("### Assistant")
            content = _normalize_message_content(raw_content)
            if content:
                parts.append(content)
            for tool_call in _message_field(message, "tool_calls", []) or []:
                function = _tool_function(tool_call)
                payload = {
                    "name": _message_field(function, "name", ""),
                    "arguments": _tool_arguments_value(_message_field(function, "arguments", {})),
                }
                parts.extend(("**Tool Call:**", "```json", _json_text(payload), "```"))
        elif role == "tool":
            # The service serializes tool content as supplied.  In particular,
            # a JSON string is intentionally rendered as a JSON string rather
            # than parsed into an object; tool-call arguments are normalized
            # separately above.
            parts.extend(("### Tool Output", "```json", _json_text(raw_content), "```"))
        else:
            raise ValueError(f"unsupported RWKV chat role: {role!r}")

    if pending_system:
        # Match the canonical template's trailing-system branch: tool
        # definitions are emitted when a non-system turn flushes the pending
        # system block, but not when the conversation ends in system content.
        parts.extend(("### System", *pending_system))
    parts.extend(("### Assistant", suffix))
    return "\n".join(parts)


def rendered_prompt_from_response(response: Any) -> str | None:
    """Return a server-rendered prompt when an endpoint exposes one."""
    candidates: list[Any] = [response]
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            candidates.append(dumped)
    hidden_params = getattr(response, "_hidden_params", None)
    if isinstance(hidden_params, dict):
        candidates.append(hidden_params)
    for candidate in candidates:
        for name in ("prompt_text", "rendered_prompt"):
            value = candidate.get(name) if isinstance(candidate, dict) else getattr(candidate, name, None)
            if isinstance(value, str):
                return value
    return None


def _looks_like_rendered_rwkv_prompt(value: str) -> bool:
    """Recognize the generation prompt returned by the RWKV server.

    Requiring the think suffix avoids mistaking a raw user message such as
    ``"User: explain this"`` for an already-rendered conversation.
    """
    stripped = value.rstrip()
    if not stripped.endswith(("<think", "<think></think")):
        return False
    return stripped.startswith(
        (
            "System✿",
            "User✿",
            "Bot✿",
            "System: ",
            "User: ",
            "Assistant: ",
            "### ",
        )
    )


def build_uploaded_messages(
    model_input: Any,
    answer: str,
    *,
    prompt_template: str | None = None,
    generation_prompt: str | None = None,
    fallback_query: Any = "",
    rendered_prompt: str | None = None,
    tools: Sequence[Any] = (),
) -> list[dict[str, str]]:
    """Build scoreboard messages with the rendered model input and answer."""
    if rendered_prompt is not None:
        messages = [{"role": "user", "content": rendered_prompt}]
    elif isinstance(model_input, (list, tuple)):
        input_messages = [
            normalized for message in model_input if (normalized := _as_message_dict(message)) is not None
        ]
        if not input_messages:
            input_messages = [{"role": "user", "content": _stringify_message_content(fallback_query)}]
        if prompt_template is not None:
            prompt = render_rwkv_prompt(
                input_messages,
                prompt_template,
                generation_prompt,
                tools=tools,
            )
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = input_messages
    elif isinstance(model_input, str):
        prompt = model_input or _stringify_message_content(fallback_query)
        if prompt_template is not None and not _looks_like_rendered_rwkv_prompt(prompt):
            prompt = render_rwkv_prompt(
                [{"role": "user", "content": prompt}],
                prompt_template,
                generation_prompt,
                tools=tools,
            )
        messages = [{"role": "user", "content": prompt}]
    else:
        raw_prompt = _stringify_message_content(model_input or fallback_query)
        if prompt_template is not None:
            raw_prompt = render_rwkv_prompt(
                [{"role": "user", "content": raw_prompt}],
                prompt_template,
                generation_prompt,
                tools=tools,
            )
        messages = [{"role": "user", "content": raw_prompt}]

    messages.append({"role": "assistant", "content": str(answer)})
    return messages


def _stored_model_config(record: dict[str, Any]) -> Any:
    config_general = record.get("config_general")
    if not isinstance(config_general, dict):
        return None
    return config_general.get("model_config")


def _stored_model_name(record: dict[str, Any]) -> str:
    config = _stored_model_config(record)
    if isinstance(config, dict) and isinstance(config.get("model_name"), str):
        return config["model_name"]
    if isinstance(config, str):
        match = re.search(r"(?:['\"])?model_name(?:['\"])?\s*[:=]\s*['\"]([^'\"]+)", config)
        if match:
            return match.group(1)
    top_level_name = record.get("model_name")
    if isinstance(top_level_name, str):
        return top_level_name
    raise ValueError("results metadata does not contain a model_name")


def _stored_context_length(record: dict[str, Any]) -> int:
    config = _stored_model_config(record)
    if isinstance(config, dict):
        for key in ("max_model_length", "ctx_len"):
            value = config.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    elif isinstance(config, str):
        match = re.search(r"(?:['\"])?(?:max_model_length|ctx_len)(?:['\"])?\s*[:=]\s*(\d+)", config)
        if match:
            return int(match.group(1))
    value = record.get("ctx_len")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 4096


def _parse_stored_model(record: dict[str, Any]) -> RwkvModel:
    model_name = Path(_stored_model_name(record)).name.removesuffix(".pth").lower()
    parts = model_name.split("-")
    if len(parts) < 3 or parts[0] not in {"rwkv7", "rwkv7a", "rwkv7b"}:
        raise ValueError(f"cannot parse RWKV model name: {model_name}")
    if not re.fullmatch(r"g1[a-z0-9]*", parts[1]) or parts[2] not in {"1.5b", "2.9b", "7.2b", "13.3b"}:
        raise ValueError(f"invalid RWKV model name: {model_name}")
    context_length = _stored_context_length(record)
    for part in parts[3:]:
        match = re.fullmatch(r"ctx(\d+)", part)
        if match:
            context_length = int(match.group(1))
            break
    return RwkvModel(parts[0], parts[1].capitalize(), parts[2], context_length)


def _stored_prompt_settings(record: dict[str, Any]) -> tuple[str | None, str | None]:
    """Recover template settings from standard LightEval result metadata."""
    config = _stored_model_config(record)
    template = generation_prompt = None
    if isinstance(config, dict):
        extra_body = config.get("extra_body")
        if isinstance(extra_body, str):
            config = extra_body
        else:
            chat_kwargs = extra_body.get("chat_template_kwargs", {}) if isinstance(extra_body, dict) else {}
            if not chat_kwargs and isinstance(config.get("chat_template_kwargs"), dict):
                chat_kwargs = config["chat_template_kwargs"]
            if isinstance(chat_kwargs, dict):
                template = chat_kwargs.get("rwkv_prompt_template")
                generation_prompt = chat_kwargs.get("rwkv_generation_prompt")
    if isinstance(config, str):
        template_match = re.search(r"(?:['\"])?rwkv_prompt_template(?:['\"])?\s*[:=]\s*['\"]([^'\"]+)", config)
        generation_match = re.search(r"(?:['\"])?rwkv_generation_prompt(?:['\"])?\s*[:=]\s*['\"]([^'\"]+)", config)
        template = template_match.group(1) if template_match else None
        generation_prompt = generation_match.group(1) if generation_match else None
    if generation_prompt == "no_think":
        # Older runs used this marker for NoCoT.  Those details must retain
        # the raw user message rather than enter the generation renderer.
        template = None
        generation_prompt = None
    return (
        template if isinstance(template, str) else None,
        generation_prompt if isinstance(generation_prompt, str) else None,
    )


@dataclass
class Score:
    model: RwkvModel
    benchmark_name: str
    num_samples: int
    avg_k: float  # 整数 for 多次测量取平均; 小数 for 随机抽样
    score: float
    truncation_rate: float
    passed_details: list[Detail]
    wrong_details: list[Detail]
    failed_details: list[Detail]


@dataclass
class Detail:
    messages: list[dict[str, str]]  # openai 标准格式
    sampling_config: SamplingConfig
    answer: str
    ground_truth: str
    is_passed: bool


def get_score(
    file_path: str | Path,
    *,
    prompt_template: str | None = None,
    generation_prompt: str | None = None,
) -> Score:
    """读取一个 benchmark 的 results JSON 及其详情。"""
    path = Path(file_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    task_names = [name for name in record["results"] if name != "all" and ":_average|" not in name]
    if len(task_names) != 1:
        raise ValueError(f"{path}: expected exactly one benchmark, found {len(task_names)}: {task_names}")
    (task_name,) = task_names
    if set(record["config_tasks"]) != {task_name}:
        raise ValueError(f"{path}: results and config_tasks must contain the same single benchmark")
    task_config = record["config_tasks"][task_name]
    task_metrics = record["results"][task_name]
    truncation_rate = task_metrics.get("truncation_rate")
    if (
        isinstance(truncation_rate, bool)
        or not isinstance(truncation_rate, (int, float))
        or not math.isfinite(truncation_rate)
        or not 0 <= truncation_rate <= 1
    ):
        raise ValueError(f"{path}: results must contain a finite truncation_rate in [0, 1]")
    metric_name = next(
        name
        for name in task_metrics
        if not name.endswith("_stderr")
        and name not in {"n_samples", "n_completions", "n_truncated", "truncation_rate"}
    )
    summary = record["summary_tasks"][task_name]
    num_samples = task_config["effective_num_docs"]
    avg_k = summary["n_completions"] / num_samples if summary["n_completions"] else 1
    stored_model = _parse_stored_model(record)
    results_root = next(parent for parent in path.parents if parent.name == "results")
    date_id = path.stem.removeprefix("results_")
    detail_path = (
        results_root.parent
        / "details"
        / path.parent.relative_to(results_root)
        / date_id
        / f"details_{task_name}_{date_id}.parquet"
    )
    stored_template, stored_generation_prompt = _stored_prompt_settings(record)
    if prompt_template is None:
        prompt_template = (
            record.get("prompt_template") if isinstance(record.get("prompt_template"), str) else stored_template
        )
    if generation_prompt is None:
        generation_prompt = (
            record.get("generation_prompt")
            if isinstance(record.get("generation_prompt"), str)
            else stored_generation_prompt
        )
        if generation_prompt is None:
            generation_prompt = _COT_GENERATION_PROMPTS.get(record.get("cot_mode"))
        if generation_prompt == "no_think":
            prompt_template = None
            generation_prompt = None
    if record.get("cot_mode") == "NoCoT":
        # NoCoT details retain the raw user message and selected assistant
        # option; never reconstruct them with a generation template.
        prompt_template = None
        generation_prompt = None
    no_cot = record.get("cot_mode") == "NoCoT"
    passed_details, wrong_details, failed_details = get_eg_details(
        detail_path,
        prompt_template=prompt_template,
        generation_prompt=generation_prompt,
        no_cot=no_cot,
    )
    return Score(
        model=stored_model,
        benchmark_name=task_config["name"],
        num_samples=num_samples,
        avg_k=float(avg_k),
        score=float(task_metrics[metric_name]),
        truncation_rate=float(truncation_rate),
        passed_details=passed_details,
        wrong_details=wrong_details,
        failed_details=failed_details,
    )


def get_eg_details(
    file_path: str | Path,
    *,
    prompt_template: str | None = None,
    generation_prompt: str | None = None,
    no_cot: bool = False,
) -> tuple[list[Detail], list[Detail], list[Detail]]:
    """读取单 benchmark 的三类样例；采样配置位于对应 results JSON 的同目录。"""
    path = Path(file_path)
    details_root = next(parent for parent in path.parents if parent.name == "details")
    sampling_config = get_sampling_config(
        details_root.parent
        / "results"
        / path.parent.parent.relative_to(details_root)
        / f"sampling_config_{path.parent.name}.json"
    )
    passed_details, wrong_details, failed_details = [], [], []
    buckets = (passed_details, wrong_details, failed_details)
    with pq.ParquetFile(path) as parquet_file:
        for batch in parquet_file.iter_batches(batch_size=64, columns=["doc", "model_response"]):
            for row in batch.to_pylist():
                doc, response = row["doc"], row["model_response"]
                choices = doc["choices"]
                gold_indices = doc["gold_index"] if isinstance(doc["gold_index"], list) else [doc["gold_index"]]
                golds = []
                for index in gold_indices:
                    gold = choices[index]
                    golds.extend(gold if isinstance(gold, list) else [gold])
                ground_truth = str(golds[0]) if len(golds) == 1 else json.dumps(golds, ensure_ascii=False)
                model_input = response.get("input")
                if no_cot and isinstance(model_input, str):
                    model_input = doc.get("query", "")
                if no_cot and response.get("logprobs"):
                    choice_scores = response["logprobs"][: len(choices)]
                    predicted_index = max(range(len(choice_scores)), key=choice_scores.__getitem__)
                    scores, answers, finish_reasons = (
                        [float(predicted_index in gold_indices)],
                        [chr(ord("A") + predicted_index)],
                        ["stop"],
                    )
                elif response["text"]:
                    scores = doc["specific"]["rwkv_rollout_scores"]
                    answers = doc["specific"]["rwkv_rollout_extracted_answers"]
                    finish_reasons = response["finish_reasons"] or ["stop"] * len(answers)
                else:
                    choice_scores = response["logprobs"][: len(choices)]
                    predicted_index = max(range(len(choice_scores)), key=choice_scores.__getitem__)
                    scores, answers, finish_reasons = (
                        [float(predicted_index in gold_indices)],
                        [choices[predicted_index]],
                        ["stop"],
                    )
                for score, answer, finish_reason in zip(scores, answers, finish_reasons, strict=True):
                    answer = str(answer)
                    outcome = 2 if finish_reason == "length" or not answer.strip() else int(score != 1)
                    if len(buckets[outcome]) < 20:
                        messages = build_uploaded_messages(
                            model_input,
                            answer,
                            prompt_template=prompt_template,
                            generation_prompt=generation_prompt,
                            fallback_query=doc.get("query", ""),
                            rendered_prompt=rendered_prompt_from_response(response),
                        )
                        buckets[outcome].append(Detail(messages, sampling_config, answer, ground_truth, outcome == 0))
                if all(len(bucket) == 20 for bucket in buckets):
                    return buckets
    return buckets


def get_sampling_config(file_path: str | Path) -> SamplingConfig:
    """读取调度器保存的完整 SamplingConfig JSON；缺失字段或 null 均报错。"""
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
    """Register the benchmark and upload its score to the Scoreboard API.

    ``api_url`` is the API root (use ``https://eval.rwkv.rs/api`` for production).
    All sampled details must share one sampling configuration; if there are no
    details, provide ``sampling_config`` explicitly.
    """
    details = score.passed_details + score.wrong_details + score.failed_details
    if sampling_config is None:
        if not details:
            raise ValueError("sampling_config is required when the score has no details")
        sampling_config = details[0].sampling_config
    if any(detail.sampling_config != sampling_config for detail in details):
        raise ValueError("all details must use the uploaded sampling_config")

    model = asdict(score.model)
    # Results filenames are parsed in lowercase, but the API requires G1a/G1i/etc.
    model["data_version"] = model["data_version"].capitalize()
    payload = {
        "model": model,
        "cot_mode": cot_mode,
        "sampling_config": asdict(sampling_config),
        "score": score.score,
        "truncation_rate": score.truncation_rate,
        "benchmark_name": score.benchmark_name,
        "field": field,
        "num_samples": score.num_samples,
        "avg_k": score.avg_k,
    }
    for name in ("passed_details", "wrong_details", "failed_details"):
        payload[name] = [
            {
                "messages": detail.messages,
                "answer": detail.answer,
                "ground_truth": detail.ground_truth,
                "is_passed": detail.is_passed,
            }
            for detail in getattr(score, name)
        ]

    with httpx.Client(
        base_url=f"{api_url.rstrip('/')}/",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    ) as client:
        response = client.post("upload", json=payload)
        response.raise_for_status()
        return response.json()
