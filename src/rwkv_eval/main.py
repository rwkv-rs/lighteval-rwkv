# 读取 configs/benchmarks.toml 和 configs/models.toml
# 按照 benchmarks.toml 流式加载数据集到内存, 优先加载题目量少的, 优先加载无需 CoT 的.
# 每完成一个题目集加载, 按照 avg@k 构造请求队列, 按照 max_num_seqs 并发
# 高难度数据集 (如gpqa) 采用 ❀ 的 prompt template, 常规数据集使用 User Assistant 的 prompt template
# 答案提取器与判分器流式操作, 每道题拿到 is_passed 立即释放 prompts 和 completions 占用的内存
# 生成式选择题的作答 / gpqa 需要使用 albatross 提供的答案提取器, 并且学习它尽可能对模型作答进行兜底的做法.
# 中文题干的生成式选择题答案提取器的补丁在old分支.
# math500 同样参考 albatross.
# 答案提取器代码请在 src/rwkv_eval/answer_extract 完成.
# 请求采用五次重试策略, 若仍然报错立即终止
# 得到 Score 后立即调取 upload 完成上传
# 若上传失败, 重试五次, 仍然失败则存入上传失败池, 等待全部评估结束后再次上传

"""RWKV evaluation entry point.

The entry point only translates the RWKV TOML manifests into LightEval's
native model, pipeline, metrics, and tracker interfaces.  Backend scheduling,
prompt preparation, task loading, sampling counts, and scoring belong to
LightEval itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Sequence

import httpx

from src.rwkv_eval.configs import BenchmarkSpec, ModelEndpoint, RwkvModel, SamplingConfig, read_benchmarks, read_models
from src.rwkv_eval.upload import Detail, Score, build_uploaded_messages, rendered_prompt_from_response
from src.rwkv_eval.upload import upload as upload_score


LOGGER = logging.getLogger(__name__)
REQUEST_RETRIES = 5
RETRY_DELAY = 1.0
NORMAL_MAX_GENERATED_TOKENS = 4096
HIGH_DIFFICULTY_MAX_GENERATED_TOKENS = 16384
TEST_MODE_MAX_SAMPLES = 10
ENDPOINT_MAX_CONTEXT_LENGTH = 32768

BenchmarkField = Literal[
    "knowledge",
    "reasoning",
    "maths",
    "coding",
    "instruction_following",
    "agentic",
    "vision",
]
CotMode = Literal["NoCoT", "FakeCoT", "CoT"]
_NO_COT_BENCHMARKS = (
    "mmlu",
    "mmlu_pro",
    "mmlu_redux_2",
    "arc",
    "ceval",
    "ceval_zho_mcf",
    "truthfulqa",
    "openbookqa",
    "hellaswag",
    "winogrande",
    "commonsenseqa",
    "med_qa",
    "med_mcqa",
    "mathqa",
    "bigbench_hard",
    "agieval",
)
_HIGH_COT_BENCHMARKS = (
    "gpqa",
    "aime24",
    "aime25",
    "aimo_progress_prize_1",
    "olympiad_bench",
    "math",
    "lcb",
)
_FAKE_COT_BENCHMARKS = ("asdiv", "arithmetic")


class EvaluationError(RuntimeError):
    """Raised when a score cannot be published after all retries."""


def _prompt_template(selector: str, requested: str) -> str:
    if requested != "assistant":
        return requested
    name = selector.casefold()
    difficult = ("gpqa", "math", "aime", "olympiad", "code")
    return "bot" if any(keyword in name for keyword in difficult) else "assistant"


def _benchmark_matches(selector: str, names: Sequence[str]) -> bool:
    selector = selector.casefold()
    return any(selector == name or selector.startswith(f"{name}:") for name in names)


def _auto_cot_mode(selector: str) -> tuple[CotMode, int]:
    """Select the evaluation mode and budget for one benchmark.

    The request type is a better signal than the eventual output length: a
    multiple-choice task should remain NoCoT even when a model could generate
    a long explanation, while difficult generative math tasks need a larger
    budget before generation starts.
    """
    if _benchmark_matches(selector, _NO_COT_BENCHMARKS):
        return "NoCoT", NORMAL_MAX_GENERATED_TOKENS
    if _benchmark_matches(selector, _FAKE_COT_BENCHMARKS):
        return "FakeCoT", NORMAL_MAX_GENERATED_TOKENS
    if _benchmark_matches(selector, _HIGH_COT_BENCHMARKS):
        return "CoT", HIGH_DIFFICULTY_MAX_GENERATED_TOKENS
    return "CoT", NORMAL_MAX_GENERATED_TOKENS


def _sampling_config(cot_mode: CotMode, max_tokens: int, seed: int) -> SamplingConfig:
    values = {
        "NoCoT": (0.0, 0, 1.0, 0.0, 0.0, 1.0),
        "FakeCoT": (1.0, 32, 0.28, 0.0, 0.0, 1.0),
        "CoT": (0.96, 32, 0.76, 1.0, 0.1, 0.988),
    }
    try:
        temp, top_k, top_p, presence, frequency, decay = values[cot_mode]
    except KeyError as error:
        raise ValueError(f"unsupported cot_mode: {cot_mode!r}") from error
    return SamplingConfig(max_tokens, temp, top_k, top_p, presence, frequency, decay, seed)


def _litelm_model(
    endpoint: ModelEndpoint,
    replicas: Sequence[ModelEndpoint],
    cot_mode: CotMode,
    template: str | None,
    max_tokens: int,
    seed: int,
    cache_dir: str | None = None,
):
    """Build the shared LiteLLM configuration for one RWKV endpoint pool."""
    from lighteval.models.endpoints.litellm_model import LiteLLMModelConfig
    from lighteval.models.model_input import GenerationParameters

    if any(item.url != endpoint.url for item in replicas):
        raise ValueError("LiteLLM endpoint pooling requires replicas to share one base URL")
    sampling = _sampling_config(cot_mode, max_tokens, seed)
    generation_prompt = {"FakeCoT": "fake_think", "CoT": "open_think"}.get(cot_mode)
    extra_body: dict[str, Any] = {"penalty_decay": sampling.penalty_decay}
    if cot_mode != "NoCoT":
        extra_body.update(
            {
                "return_prompt_text": True,
                "chat_template_kwargs": {
                    "rwkv_prompt_template": template,
                    "rwkv_generation_prompt": generation_prompt,
                },
            }
        )
    return LiteLLMModelConfig(
        model_name=endpoint.model_name,
        provider="openai",
        base_url=f"{endpoint.url.rstrip('/')}/v1",
        api_key=endpoint.api_key,
        concurrent_requests=sum(item.max_num_seqs for item in replicas),
        max_model_length=max(endpoint.ctx_len, ENDPOINT_MAX_CONTEXT_LENGTH),
        cache_dir=cache_dir or "~/.cache/huggingface/lighteval",
        api_max_retry=5,
        extra_body=extra_body,
        generation_parameters=GenerationParameters(
            temperature=sampling.temp,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            presence_penalty=sampling.presence_penalty,
            frequency_penalty=sampling.frequency_penalty,
            max_new_tokens=sampling.max_generated_tokens,
            seed=sampling.seed,
        ),
    )


def _order_benchmarks(
    benchmarks: Sequence[BenchmarkSpec],
    max_samples: int | None,
    *,
    lightweight: bool = False,
) -> list[BenchmarkSpec]:
    """Order tasks without generation/CoT first, then sort by sample count.

    A normal run calculates the exact count after applying each task's
    formatter/filter.  Test runs only need a stable size estimate: loading all
    subtasks just to sort them can take longer than the evaluation itself (for
    example, MMLU and C-Eval expand to dozens of datasets).  In lightweight
    mode we load one representative subtask per selector and scale its count
    by the number of matched subtasks.  This preserves the intended ordering
    while keeping the test-mode startup bounded.
    """
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.registry import Registry
    from lighteval.tasks.requests import SamplingMethod

    registry = Registry(tasks=",".join(spec.selector for spec in benchmarks), load_multilingual=True)
    tasks = registry.load_tasks()
    estimates: dict[str, tuple[int, int]] = {}
    for spec in benchmarks:
        selector = spec.selector.rsplit("|", 1)[0]
        matched = [task for task in tasks.values() if task.name == selector or task.name.startswith(f"{selector}:")]
        requires_cot = any(SamplingMethod.GENERATIVE in task.sampling_methods for task in matched)
        sample_count = 0

        if lightweight and matched:
            representative = matched[0]
            started = time.perf_counter()
            try:
                LightevalTask.load_datasets({representative.full_name: representative}, 1)
                representative_count = sum(
                    len(representative.dataset[split]) for split in representative.evaluation_split
                )
                if max_samples is not None:
                    representative_count = min(representative_count, max_samples)
                sample_count = representative_count * len(matched)
                LOGGER.info(
                    "benchmark size estimate: selector=%s subtasks=%d representative_samples=%d load_seconds=%.2f",
                    spec.selector,
                    len(matched),
                    representative_count,
                    time.perf_counter() - started,
                )
            except Exception as error:
                # The actual pipeline will still report a hard dataset error if
                # this selector cannot be loaded.  Do not make ordering itself
                # fail and hide the actionable error behind a pre-scan.
                LOGGER.warning(
                    "could not estimate benchmark size for %s (%s); using subtask-count fallback",
                    spec.selector,
                    error,
                )
                sample_count = (max_samples or 1) * len(matched)
            finally:
                representative.dataset = None
                representative._docs = None
                representative._fewshot_docs = None
        else:
            for task in matched:
                started = time.perf_counter()
                try:
                    LightevalTask.load_datasets({task.full_name: task}, 1)
                    sample_count += sum(len(task.dataset[split]) for split in task.evaluation_split)
                    LOGGER.info(
                        "benchmark subtask loaded: task=%s samples=%d load_seconds=%.2f",
                        task.full_name,
                        sum(len(task.dataset[split]) for split in task.evaluation_split),
                        time.perf_counter() - started,
                    )
                finally:
                    task.dataset = None
                    task._docs = None
                    task._fewshot_docs = None
            if max_samples is not None:
                sample_count = min(sample_count, max_samples * max(len(matched), 1))

        estimates[spec.selector] = (sample_count, int(requires_cot))

    ordered = sorted(
        enumerate(benchmarks),
        key=lambda item: (
            estimates.get(item[1].selector, (10**18, 1))[1],
            estimates.get(item[1].selector, (10**18, 1))[0],
            item[0],
        ),
    )
    LOGGER.info(
        "benchmark order: %s",
        [f"{spec.selector}(samples={estimates.get(spec.selector, ('?', '?'))[0]})" for _, spec in ordered],
    )
    return [spec for _, spec in ordered]


def _metric_value(value: Any) -> float:
    if isinstance(value, dict):
        value = next((item for item in value.values() if isinstance(item, (int, float))), 0.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _ground_truth(doc: Any) -> str:
    golds = [str(gold) for gold in doc.get_golds()]
    return golds[0] if len(golds) == 1 else str(golds)


def _choice_indices(doc: Any) -> list[int]:
    value = doc.gold_index
    return [int(index) for index in value] if isinstance(value, (list, tuple)) else [int(value)]


def _display_choice_prompt(query: str, choices: list[Any]) -> str:
    """Show a readable multiple-choice prompt without changing model inputs."""
    if any(re.search(rf"(?mi)^\s*{label}\s*[.)]", query) for label in "ABCDE"[: len(choices)]):
        return query
    options = "\n".join(f"{chr(ord('A') + index)}. {choice}" for index, choice in enumerate(choices))
    return f"{query.rstrip()}\n\n{options}\n\nAnswer:"


def _choice_distribution(choices: list[Any], logprobs: list[float], selected: int) -> str:
    maximum = max(logprobs)
    weights = [math.exp(value - maximum) for value in logprobs]
    normalizer = sum(weights) or 1.0
    payload = {
        "selected": chr(ord("A") + selected),
        "choices": [
            {
                "label": chr(ord("A") + index),
                "logprob": float(logprob),
                "probability": weights[index] / normalizer,
            }
            for index, logprob in enumerate(logprobs)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _native_detail_score(native: Any) -> float | None:
    metric = getattr(native, "metric", None)
    if not isinstance(metric, dict) or not metric:
        return None
    return _metric_value(next(iter(metric.values())))


def _native_score(pipeline: Any, task_name: str, metrics: dict[str, Any]) -> float:
    del pipeline, task_name
    values = [value for name, value in metrics.items() if not name.endswith("_stderr")]
    return _metric_value(values[0]) if values else 0.0


def _detail_score(task: Any, doc: Any, response: Any, index: int) -> float:
    metric = next(
        (item for item in task.metrics if str(getattr(item.category, "value", item.category)) == "GENERATIVE"),
        None,
    ) or (task.metrics[0] if task.metrics else None)
    if metric is None:
        return 0.0
    sample = response[index] if response.text and index < len(response.text) else response
    scorer = metric.sample_level_fn
    try:
        if hasattr(scorer, "compute_score"):
            return _metric_value(scorer.compute_score(doc, sample))
        if hasattr(scorer, "compute"):
            try:
                # Keyword arguments support both the native LightEval
                # ``compute(doc, model_response)`` contract and metrics such
                # as LiveCodeBench that declare the parameters in reverse.
                return _metric_value(scorer.compute(doc=doc, model_response=sample))
            except TypeError:
                return _metric_value(scorer.compute(doc, sample))
        return _metric_value(metric.compute_sample(doc=doc, model_response=sample))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return 0.0


def _internal_task_name(pipeline: Any, public_task_name: str) -> str:
    """Translate LightEval's display name back to its internal task key.

    ``EvaluationTracker.generate_final_dict`` replaces the separator in a
    full task name (``task|fewshot``) with ``:`` for display.  Task selectors
    themselves may also contain colons, so a plain ``replace`` in the other
    direction is ambiguous; use the pipeline's authoritative task map.
    """
    task_names = pipeline.tasks_dict
    if public_task_name in task_names:
        return public_task_name
    display_to_internal = {name.replace("|", ":"): name for name in task_names}
    try:
        return display_to_internal[public_task_name]
    except KeyError as error:
        raise KeyError(
            f"LightEval returned task {public_task_name!r}, but the pipeline contains {sorted(task_names)!r}"
        ) from error


def _make_score(  # noqa: C901
    pipeline: Any,
    task_name: str,
    metrics: dict[str, Any],
    sampling: SamplingConfig,
    *,
    prompt_template: str | None = None,
    generation_prompt: str | None = None,
    cot_mode: CotMode | None = None,
) -> Score:
    task = pipeline.tasks_dict[task_name]
    native_details = pipeline.get_details().get(task_name, [])
    passed, wrong, failed = [], [], []
    truncated = total = 0
    for native in native_details:
        doc, response = native.doc, native.model_response
        ground_truth = _ground_truth(doc)
        answers = [str(answer) for answer in (response.final_text or [])]
        choices = getattr(doc, "choices", None) or []
        logprobs = getattr(response, "logprobs", []) or []
        nocot_choice_score: float | None = None
        if cot_mode == "NoCoT" and choices and len(logprobs) >= len(choices):
            # NoCoT details expose the native choice distribution.  A
            # multi-answer task such as TruthfulQA cannot be represented by a
            # single greedy letter without hiding the behavior used by MC2.
            if not (doc.specific or {}).get("_rwkv_missing_answer"):
                predicted = max(range(len(choices)), key=logprobs.__getitem__)
                gold_indices = _choice_indices(doc)
                ground_truth = ", ".join(chr(ord("A") + index) for index in gold_indices)
                if len(gold_indices) > 1:
                    answers = [_choice_distribution(choices, logprobs[: len(choices)], predicted)]
                else:
                    answers = [chr(ord("A") + predicted)]
                    nocot_choice_score = float(predicted in gold_indices)
        elif not any(answer.strip() for answer in answers) and choices and len(logprobs) >= len(choices):
            predicted = max(range(len(choices)), key=logprobs.__getitem__)
            answers = [chr(ord("A") + predicted)]
        for index, answer in enumerate(answers or [""]):
            finish_reasons = response.finish_reasons or []
            finish_reason = finish_reasons[index] if index < len(finish_reasons) else "stop"
            is_truncated = finish_reason.lower() in {"length", "max_tokens"}
            truncated += int(is_truncated)
            total += 1

            detail_score = nocot_choice_score
            if detail_score is None:
                detail_score = _native_detail_score(native)
            if detail_score is None:
                detail_score = _detail_score(task, doc, response, index)

            # NoCoT has several actual candidate requests, so do not expose
            # their JSON audit envelope as if it were one user prompt.  The
            # detail view shows the original query and choices once; scoring
            # still uses the native candidate logprobs above.
            detail_input = (
                _display_choice_prompt(doc.query, choices) if cot_mode == "NoCoT" and choices else response.input
            )
            messages = build_uploaded_messages(
                detail_input,
                str(answer),
                prompt_template=prompt_template,
                generation_prompt=generation_prompt,
                fallback_query=doc.query,
                rendered_prompt=None if cot_mode == "NoCoT" else rendered_prompt_from_response(response),
            )
            detail = Detail(
                messages=messages,
                sampling_config=sampling,
                answer=str(answer),
                ground_truth=ground_truth,
                is_passed=not is_truncated and bool(str(answer).strip()) and detail_score == 1.0,
            )
            bucket = failed if is_truncated or not str(answer).strip() else passed if detail.is_passed else wrong
            if len(bucket) < 20:
                bucket.append(detail)

    completions = pipeline.task_completion_counts.get(task_name, total)
    avg_k = 1.0 if cot_mode == "NoCoT" else pipeline.task_avg_k.get(task_name, 1)
    return Score(
        model=RwkvModel(*_model_parts(pipeline.model.config.model_name, pipeline.model.config.max_model_length)),
        benchmark_name=task_name.split("|", 1)[0],
        num_samples=pipeline.task_sample_counts.get(task_name, len(native_details)),
        avg_k=float(avg_k),
        score=_native_score(pipeline, task_name, metrics),
        truncation_rate=pipeline.task_truncation_counts.get(task_name, truncated) / completions
        if completions
        else 0.0,
        passed_details=passed,
        wrong_details=wrong,
        failed_details=failed,
    )


def _model_parts(name: str, ctx_len: int) -> tuple[str, str, str, int]:
    parts = Path(name).name.removesuffix(".pth").lower().split("-")
    if len(parts) < 3 or parts[0] not in {"rwkv7", "rwkv7a", "rwkv7b"}:
        raise ValueError(f"cannot parse RWKV model name: {name}")
    if not re.fullmatch(r"g1[a-z0-9]*", parts[1]) or parts[2] not in {"1.5b", "2.9b", "7.2b", "13.3b"}:
        raise ValueError(f"invalid RWKV model name: {name}")
    return parts[0], parts[1].capitalize(), parts[2], ctx_len


def _field_for_task(task_name: str, benchmarks: Sequence[BenchmarkSpec]) -> BenchmarkField:
    leaf = task_name.split("|", 1)[0]
    for benchmark in benchmarks:
        selector = benchmark.selector.rsplit("|", 1)[-1]
        if leaf == selector or leaf.startswith(f"{selector}:"):
            return benchmark.field
    return benchmarks[0].field


def _score_path(score: Score, output_dir: str) -> Path:
    model = score.model
    model_id = f"{model.arch_version}-{model.data_version}-{model.param_size}-ctx{model.ctx_len}"
    benchmark_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", score.benchmark_name)
    path = Path(output_dir) / "scores" / model_id / f"{benchmark_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _persist_score(
    score: Score,
    field: BenchmarkField,
    output_dir: str,
    status: str,
    *,
    prompt_template: str | None = None,
    cot_mode: CotMode | None = None,
) -> Path:
    path = _score_path(score, output_dir)
    payload = asdict(score)
    payload.update({"field": field, "upload_status": status})
    if prompt_template is not None:
        payload["prompt_template"] = prompt_template
    if cot_mode is not None:
        payload["cot_mode"] = cot_mode
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def _set_score_status(path: Path, status: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["upload_status"] = status
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


async def _upload(score: Score, field: BenchmarkField, cot_mode: CotMode, token: str, url: str) -> None:
    last_error = None
    for attempt in range(REQUEST_RETRIES):
        try:
            await asyncio.to_thread(upload_score, score, field=field, cot_mode=cot_mode, token=token, api_url=url)
            return
        except Exception as error:
            last_error = error
            if attempt + 1 < REQUEST_RETRIES:
                await asyncio.sleep(RETRY_DELAY * 2**attempt)
    raise EvaluationError(f"score upload failed after {REQUEST_RETRIES} attempts") from last_error


async def evaluate(  # noqa: C901
    models: Sequence[ModelEndpoint],
    benchmarks: Sequence[BenchmarkSpec],
    *,
    prompt_template: str = "assistant",
    max_samples: int | None = None,
    test_mode: bool = False,
    seed: int = 42,
    scoreboard_token: str | None = None,
    scoreboard_url: str = "https://eval.rwkv.rs/test/api",
    no_upload: bool = False,
    output_dir: str = "results",
) -> tuple[list[Score], list[Score]]:
    if not models or not benchmarks:
        raise ValueError("models and benchmarks must not be empty")
    if prompt_template not in {"bot", "assistant", "function_calling"}:
        raise ValueError(f"unsupported prompt_template: {prompt_template!r}")
    if not no_upload and not scoreboard_token:
        raise ValueError("scoreboard token is required unless --no-upload is used")

    effective_max_samples = max_samples
    if test_mode:
        effective_max_samples = (
            TEST_MODE_MAX_SAMPLES if max_samples is None else min(max_samples, TEST_MODE_MAX_SAMPLES)
        )
        LOGGER.info(
            "test mode enabled: evaluating at most %d questions per benchmark",
            effective_max_samples,
        )

    groups: dict[tuple[str, int], list[ModelEndpoint]] = {}
    group_model_keys: dict[tuple[str, int], tuple[str, str, str, int]] = {}
    for endpoint in models:
        group = (endpoint.model_name, endpoint.ctx_len)
        groups.setdefault(group, []).append(endpoint)
        model = _model_parts(endpoint.model_name, endpoint.ctx_len)
        group_model_keys[group] = (
            model[0].casefold(),
            model[1].casefold(),
            model[2].casefold(),
            model[3],
        )

    if no_upload:
        pending_pairs = {
            (model_key, benchmark.selector, benchmark.field)
            for model_key in group_model_keys.values()
            for benchmark in benchmarks
        }
    else:
        task_rows: list[Any] | None = None
        last_error: Exception | None = None
        for attempt in range(REQUEST_RETRIES):
            try:
                async with httpx.AsyncClient(
                    base_url=f"{scoreboard_url.rstrip('/')}/",
                    headers={"Authorization": f"Bearer {scoreboard_token}"},
                    timeout=30.0,
                ) as client:
                    response = await client.post("get_tasks", json={"frameworks": ["lighteval"]})
                    response.raise_for_status()
                    payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
                    raise ValueError("Scoreboard get_tasks response must contain a tasks list")
                task_rows = payload["tasks"]
                break
            except Exception as error:
                last_error = error
                if attempt + 1 < REQUEST_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * 2**attempt)
        if task_rows is None:
            raise RuntimeError(f"could not get pending tasks after {REQUEST_RETRIES} attempts") from last_error

        pending_pairs: set[tuple[tuple[str, str, str, int], str, str]] = set()
        for task in task_rows:
            if (
                not isinstance(task, dict)
                or not isinstance(task.get("model"), dict)
                or not isinstance(task.get("benchmark"), dict)
            ):
                raise ValueError(f"invalid task returned by Scoreboard: {task!r}")
            task_model = task["model"]
            task_benchmark = task["benchmark"]
            if task_benchmark.get("framework", "lighteval") != "lighteval":
                continue
            try:
                benchmark_name = task_benchmark["name"]
                benchmark_field = task_benchmark["field"]
                task_model_key = (
                    str(task_model["arch_version"]).casefold(),
                    str(task_model["data_version"]).casefold(),
                    str(task_model["param_size"]).casefold(),
                    int(task_model["ctx_len"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid task returned by Scoreboard: {task!r}") from error
            if (
                not isinstance(benchmark_name, str)
                or not benchmark_name
                or not isinstance(benchmark_field, str)
                or not benchmark_field
            ):
                raise ValueError(f"invalid benchmark in Scoreboard task: {task!r}")
            pending_pairs.add((task_model_key, benchmark_name, benchmark_field))

        LOGGER.info("Scoreboard returned %d pending model/benchmark pairs", len(pending_pairs))

    benchmarks = [
        benchmark
        for benchmark in benchmarks
        if any((group_model_keys[group], benchmark.selector, benchmark.field) in pending_pairs for group in groups)
    ]
    if not benchmarks:
        LOGGER.info("all configured model/benchmark pairs already have an active score")
        return [], []

    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    benchmarks = _order_benchmarks(
        benchmarks,
        effective_max_samples,
        lightweight=test_mode,
    )
    successful: list[Score] = []
    pending: list[tuple[Score, BenchmarkField, CotMode, Path]] = []
    for benchmark in benchmarks:
        benchmark_cot_mode, benchmark_max_tokens = _auto_cot_mode(benchmark.selector)
        LOGGER.info(
            "benchmark mode: selector=%s cot_mode=%s max_generated_tokens=%d",
            benchmark.selector,
            benchmark_cot_mode,
            benchmark_max_tokens,
        )
        generation_prompt = {"FakeCoT": "fake_think", "CoT": "open_think"}.get(benchmark_cot_mode)
        sampling = _sampling_config(benchmark_cot_mode, benchmark_max_tokens, seed)
        # NoCoT uses LightEval's loglikelihood/greedy-choice path.  It has no
        # RWKV generation prompt to render into the uploaded user message.
        template = None if benchmark_cot_mode == "NoCoT" else _prompt_template(benchmark.selector, prompt_template)
        model_groups = [
            (group, replicas)
            for group, replicas in groups.items()
            if (group_model_keys[group], benchmark.selector, benchmark.field) in pending_pairs
        ]
        pipelines = []
        for (model_name, ctx_len), replicas in model_groups:
            LOGGER.info(
                "starting LightEval for model=%s benchmark=%s replicas=%d concurrent_requests=%d",
                model_name,
                benchmark.selector,
                len(replicas),
                sum(item.max_num_seqs for item in replicas),
            )
            model_config = _litelm_model(
                replicas[0],
                replicas,
                benchmark_cot_mode,
                template,
                benchmark_max_tokens,
                seed,
                cache_dir=str(Path(output_dir) / ".lighteval_cache"),
            )
            tracker = EvaluationTracker(
                output_dir=output_dir,
                save_details=False,
                save_streaming_completions=True,
            )
            params = PipelineParameters(
                launcher_type=ParallelismManager.NONE,
                max_samples=effective_max_samples,
                max_total_samples=effective_max_samples if test_mode else None,
                load_tasks_multilingual=True,
                streaming_evaluation=True,
            )
            pipelines.append(
                await asyncio.to_thread(Pipeline, benchmark.selector, params, tracker, model_config=model_config)
            )

        # Evaluate one benchmark concurrently across independent model pools;
        # the next benchmark starts only after this wave has been collected.
        await asyncio.gather(*(asyncio.to_thread(pipeline.evaluate) for pipeline in pipelines))
        for pipeline in pipelines:
            pipeline.show_results()
            rows: list[Score] = []
            for public_task_name, metrics in pipeline.get_results()["results"].items():
                if public_task_name == "all" or ":_average:" in public_task_name or ":_average|" in public_task_name:
                    continue
                task_name = _internal_task_name(pipeline, public_task_name)
                rows.append(
                    _make_score(
                        pipeline,
                        task_name,
                        metrics,
                        sampling,
                        prompt_template=template,
                        generation_prompt=generation_prompt,
                        cot_mode=benchmark_cot_mode,
                    )
                )

            # A selector that expands to multiple LightEval tasks is one
            # benchmark in the external scoreboard.
            if len(rows) > 1:
                weights = [item.num_samples * item.avg_k for item in rows]
                total_samples, total_completions = sum(item.num_samples for item in rows), sum(weights) or 1
                aggregate = rows[0]
                aggregate.benchmark_name, aggregate.num_samples = benchmark.selector, total_samples
                aggregate.avg_k = total_completions / total_samples if total_samples else 0.0
                aggregate.score = sum(item.score * weight for item, weight in zip(rows, weights)) / total_completions
                aggregate.truncation_rate = (
                    sum(item.truncation_rate * weight for item, weight in zip(rows, weights)) / total_completions
                )
                details = [
                    [detail for item in rows for detail in getattr(item, name)][:20]
                    for name in ("passed_details", "wrong_details", "failed_details")
                ]
                aggregate.passed_details, aggregate.wrong_details, aggregate.failed_details = details
                rows = [aggregate]
            for score in rows:
                successful.append(score)
                score_path = _persist_score(
                    score,
                    benchmark.field,
                    output_dir,
                    "local_only" if no_upload else "pending_upload",
                    prompt_template=template,
                    cot_mode=benchmark_cot_mode,
                )
                if not no_upload:
                    try:
                        await _upload(
                            score,
                            benchmark.field,
                            benchmark_cot_mode,
                            scoreboard_token or "",
                            scoreboard_url,
                        )
                    except EvaluationError:
                        pending.append((score, benchmark.field, benchmark_cot_mode, score_path))
                    else:
                        _set_score_status(score_path, "uploaded")
                        score.passed_details.clear()
                        score.wrong_details.clear()
                        score.failed_details.clear()
            await asyncio.to_thread(pipeline.save_and_push_results)

    failures: list[Score] = []
    for score, field, score_cot_mode, score_path in pending:
        try:
            await _upload(score, field, score_cot_mode, scoreboard_token or "", scoreboard_url)
        except EvaluationError:
            _set_score_status(score_path, "upload_failed")
            failures.append(score)
        else:
            _set_score_status(score_path, "uploaded")
            score.passed_details.clear()
            score.wrong_details.clear()
            score.failed_details.clear()
    return successful, failures


def _argument_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run RWKV through the LightEval pipeline.")
    parser.add_argument("--models", type=Path, default=root / "configs/models.toml")
    parser.add_argument("--benchmarks", type=Path, default=root / "configs/benchmarks.toml")
    parser.add_argument("--prompt-template", choices=("bot", "assistant", "function_calling"), default="assistant")
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
    parser.add_argument("--scoreboard-token", default=os.environ.get("SCOREBOARD_API_TOKEN"))
    parser.add_argument(
        "--scoreboard-url", default=os.environ.get("SCOREBOARD_API_URL", "https://eval.rwkv.rs/test/api")
    )
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    scores, failures = asyncio.run(
        evaluate(
            read_models(args.models),
            read_benchmarks(args.benchmarks),
            prompt_template=args.prompt_template,
            max_samples=args.max_samples,
            test_mode=args.test_mode,
            seed=args.seed,
            scoreboard_token=args.scoreboard_token,
            scoreboard_url=args.scoreboard_url,
            no_upload=args.no_upload,
            output_dir=args.output_dir,
        )
    )
    if failures:
        LOGGER.error("%d score uploads remain pending", len(failures))
        return 1
    LOGGER.info("completed %d scores", len(scores))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
