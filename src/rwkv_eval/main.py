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
import collections
import copy
import json
import logging
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal, Sequence

import httpx

from lighteval.metrics.metrics_sample import ExactMatches, PassAtK, SamplingMetric
from lighteval.pipeline import Pipeline
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import SamplingMethod
from lighteval.tasks.tasks.ifbench.main import RWKVIFBench
from src.rwkv_eval.answer_extract.free_response import (
    answer_extractor_for_benchmark,
    extract_answer,
    extract_math_answer,
    verify_math_answer,
)
from src.rwkv_eval.configs import BenchmarkSpec, ModelEndpoint, RwkvModel, SamplingConfig, read_benchmarks, read_models
from src.rwkv_eval.upload import Score, get_detail
from src.rwkv_eval.upload import upload as upload_score


LOGGER = logging.getLogger(__name__)
REQUEST_RETRIES = 6
RETRY_DELAY = 1.0
NORMAL_MAX_GENERATED_TOKENS = 4096
HIGH_DIFFICULTY_MAX_GENERATED_TOKENS = 24576
TEST_MODE_MAX_SAMPLES = 10

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
    "math_500",
    "lcb",
)
_FAKE_COT_BENCHMARKS = ("asdiv", "arithmetic")


class EvaluationError(RuntimeError):
    """Raised when a score cannot be published after all retries."""


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


class RWKVExactMatches(ExactMatches):
    """RWKV comparison, following ExactMatches.compute_one_item."""

    def __init__(self, *, allow_text: bool = False, choice: bool = False):
        super().__init__()
        self.allow_text = allow_text
        self.choice = choice

    def compute_one_item(self, gold: str, pred: str) -> float:
        if self.choice:
            return super().compute_one_item(gold, pred) if pred else 0.25
        return int(verify_math_answer(gold, pred, allow_text=self.allow_text))


class RWKVPassAtK(PassAtK):
    """PassAtK with the user-approved RWKV missing-choice expectation."""

    def pass_at_k(self, all_scores: list[float]) -> float:
        if self.n and self.k == 1 and self.n == len(all_scores) and all_scores.count(1) == 0:
            missing = all_scores.count(0.25)
            if missing:
                return 1.0 - 0.75**missing
        return super().pass_at_k(all_scores)


def rwkv_prompt(line, task_name=None, *, prompt_function):  # noqa: C901
    """Adapt a native prompt only for the RWKV registry's copied configs."""
    doc = prompt_function(line, task_name)
    if doc is None:
        return None
    extractor = answer_extractor_for_benchmark(task_name)
    if extractor == "short_answer":
        # Apply ASDiv's native unit removal to every answer, not only the
        # first: its dataset uses semicolons for ordered multi-part targets.
        doc.choices = ["; ".join(part.split(" (")[0].strip() for part in line["answer"].split(";"))]
    if extractor in {"math", "short_answer"}:
        # Dataset golds may be worked solutions or a list of accepted answers.
        # Keep the native gold-index structure and normalize each alternative.
        choices = [doc.choices] if isinstance(doc.choices, str) else doc.choices
        doc.choices = [
            [str(gold) if extractor == "short_answer" else extract_math_answer(str(gold)) for gold in choice]
            if isinstance(choice, list)
            else str(choice)
            if extractor == "short_answer"
            else extract_math_answer(str(choice))
            for choice in choices
        ]
        if any(
            choice is None or isinstance(choice, list) and any(gold is None for gold in choice)
            for choice in doc.choices
        ):
            raise EvaluationError(f"RWKV task {task_name} has an unextractable gold answer")
        if task_name.startswith("olympiad_bench:") and line["is_multiple_answer"]:
            # OlympiadBench explicitly asks for all components, not any one
            # accepted alternative. Keep its comma-connected answer contract.
            doc.choices = [", ".join(choice) if isinstance(choice, list) else choice for choice in doc.choices]
    elif _auto_cot_mode(task_name)[0] == "NoCoT":
        if task_name == "commonsenseqa":
            doc.query = doc.query.removeprefix(doc.instruction or "")
            doc.instruction = ""
    # Prototype: arc_prompt/mmlu_prompt. All RWKV LOGPROBS candidates follow
    # a newline-delimited answer cue; retain an existing cue's whitespace.
    # Only generation delegates the answer boundary to the chat template.
    if _auto_cot_mode(task_name)[0] == "NoCoT":
        if not doc.query.rstrip().endswith("\nAnswer:"):
            doc.query = doc.query.rstrip().removesuffix("Answer:").rstrip() + "\nAnswer: "
    elif doc.query.rstrip().endswith("Answer:"):
        doc.query = doc.query.rstrip()[: -len("Answer:")].rstrip() + "\n"
    return doc


class RWKVLightevalTaskConfig(LightevalTaskConfig):
    """Prototype: LightevalTaskConfig.__str__; stable RWKV cache identity."""

    def __str__(self, lite: bool = False):
        if lite:
            config = copy.copy(self)
            # Native formatting names individual callables. Solver lists and
            # grouped corpus-function dictionaries must also be process-stable.
            config.solver = [solver.__qualname__ for solver in self.solver] if self.solver else self.solver
            return re.sub(r" at 0x[0-9a-fA-F]+(?=>)", "", LightevalTaskConfig.__str__(config, lite=True))
        return super().__str__(lite=lite)


class RWKVLightevalTask(LightevalTask):
    """Prototype: LightevalTask._get_docs_from_split; split-local cache IDs."""

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

    def load_tasks(self) -> dict[str, LightevalTask]:
        if len(self.task_to_configs) == 0:
            return {config.full_name: RWKVLightevalTask(config=config) for config in self._task_registry.values()}
        return {
            config.full_name: RWKVLightevalTask(config=config)
            for configs in self.task_to_configs.values()
            for config in configs
        }

    def _update_task_configs(self):  # noqa: C901
        configs = super()._update_task_configs()
        from inspect_ai.scorer import choice
        from inspect_ai.solver import multiple_choice

        from lighteval.metrics.metrics import Metrics

        for task_configs in configs.values():
            for index, original in enumerate(task_configs):
                config = RWKVLightevalTaskConfig(
                    **{field.name: getattr(original, field.name) for field in fields(original) if field.init}
                )
                task_configs[index] = config
                config.prompt_function = partial(rwkv_prompt, prompt_function=config.prompt_function)
                mode, _ = _auto_cot_mode(config.name)
                if mode == "NoCoT":
                    # Prototype: native task version; invalidate pre-cue caches.
                    config.version += 1
                    config.solver = [multiple_choice(cache=True)]
                    config.scorer = choice()
                    # Request routing is determined by metric category, not
                    # solver metadata (e.g. native MMLU uses exact_match).
                    if any(metric.category == SamplingMethod.GENERATIVE for metric in config.metrics):
                        config.metrics = [copy.deepcopy(Metrics.loglikelihood_acc.value)]
                elif config.name in {"ifbench_test", "ifbench_multiturn"}:
                    config.metrics[0].sample_level_fn = RWKVIFBench()
                elif answer_extractor_for_benchmark(config.name) in {"math", "short_answer", "choice"}:
                    extractor = answer_extractor_for_benchmark(config.name)
                    # Change only private generative comparators. Native
                    # LOGPROBS and math aggregation retain their own behavior.
                    for metric in config.metrics:
                        if metric.category != SamplingMethod.GENERATIVE:
                            continue
                        comparator = RWKVExactMatches(
                            allow_text=extractor == "short_answer", choice=extractor == "choice"
                        )
                        sample_fn = metric.sample_level_fn
                        if extractor == "choice" and isinstance(sample_fn, PassAtK):
                            metric.sample_level_fn = RWKVPassAtK(
                                k=sample_fn.k,
                                n=sample_fn.n,
                                normalize=sample_fn.normalize,
                                strip_strings=sample_fn.strip_strings,
                                sample_scoring_function=comparator.compute,
                            )
                        elif isinstance(sample_fn, SamplingMetric):
                            sample_fn.compute_score = comparator.compute
                        else:
                            metric.sample_level_fn = comparator
        return configs


class RWKVPipeline(Pipeline):
    """Prototype: native Pipeline; RWKV task/output policy is isolated here."""

    def _init_tasks_and_requests(self, tasks: str):
        # Follow the native initialization order without monkeypatching Registry
        # or altering TASKS_TABLE/metric singleton objects for other models.
        self.registry = RWKVRegistry(
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
                    extract_answer(text, extractor, len(doc.choices), rendered_prompt=response.rendered_prompt) or ""
                    for text in response.text
                ]
        return outputs

    def _post_process_outputs(self, sampling_method_responses):
        # Already processed against the actual batch in _run_model_for_docs;
        # raw responses and question-level aggregates stay separate.
        for method, responses in sampling_method_responses.items():
            if method != SamplingMethod.GENERATIVE:
                continue
            for response in responses:
                if response.text_post_processed is None:
                    raise EvaluationError("RWKV response is missing benchmark post-processing")

    def _compute_metrics_for_sampling(self, sampling_docs, sampling_method_responses):
        for method, responses in sampling_method_responses.items():
            if method != SamplingMethod.GENERATIVE:
                continue
            for doc, response in zip(sampling_docs[method], responses, strict=True):
                task = self.tasks_dict[doc.task_name]
                if answer_extractor_for_benchmark(task.name) not in {"math", "short_answer", "choice"}:
                    continue
                sample_fn = task.metrics[0].sample_level_fn
                comparator = sample_fn.compute_score if isinstance(sample_fn, SamplingMetric) else sample_fn.compute
                if doc.specific is None:
                    doc.specific = {}
                # Per-rollout scores must not inherit a question's maj@n or
                # pass@k aggregate.  Details and native aggregation share the
                # same comparator and processed gold/prediction values.
                doc.specific["rwkv_rollout_scores"] = [
                    float(comparator(doc=doc, model_response=response[index])) for index in range(len(response.text))
                ]
        return super()._compute_metrics_for_sampling(sampling_docs, sampling_method_responses)


def _sampling_config(cot_mode: CotMode, max_tokens: int, seed: int) -> SamplingConfig:
    values = {
        "NoCoT": (0.0, 0, 1.0, 0.0, 0.0, 1.0),
        "FakeCoT": (1.0, 32, 0.28, 0.0, 0.0, 1.0),
        "CoT": (0.96, -1, 0.90, 1.0, 0.09, 0.994),
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
    if sampling.top_k == -1:
        extra_body["top_k"] = -1
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
        max_model_length=endpoint.ctx_len,
        scoreboard_ctx_len=endpoint.ctx_len,
        cache_dir=cache_dir or "~/.cache/huggingface/lighteval",
        api_max_retry=6,
        timeout=14400.0,
        extra_body=extra_body,
        generation_only=cot_mode != "NoCoT",
        generation_parameters=GenerationParameters(
            temperature=sampling.temp,
            top_k=None if sampling.top_k == -1 else sampling.top_k,
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

    registry = RWKVRegistry(tasks=",".join(spec.selector for spec in benchmarks), load_multilingual=True)
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


def _native_score(metrics: dict[str, Any]) -> float:
    values = [value for name, value in metrics.items() if not name.endswith("_stderr")]
    return _metric_value(values[0]) if values else 0.0


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
    cot_mode: CotMode | None = None,
) -> Score:
    native_details = pipeline.get_details().get(task_name, [])
    buckets = ([], [], [])
    truncated = total = 0
    for native in native_details:
        doc, response = native.doc, native.model_response
        count = 1 if cot_mode == "NoCoT" else max(len(response.text), 1)
        for index in range(count):
            detail, bucket = get_detail(doc, response, sampling, native.metric, index, no_cot=cot_mode == "NoCoT")
            finish = response.finish_reasons[index] if index < len(response.finish_reasons) else "stop"
            truncated += int(finish.lower() in {"length", "max_tokens"})
            total += 1
            if len(buckets[bucket]) < 20:
                buckets[bucket].append(detail)
    passed, wrong, failed = buckets

    completions = pipeline.task_completion_counts.get(task_name, total)
    avg_k = 1.0 if cot_mode == "NoCoT" else pipeline.task_avg_k.get(task_name, 1)
    return Score(
        model=RwkvModel(
            *_model_parts(
                pipeline.model.config.model_name,
                pipeline.model.config.scoreboard_ctx_len or pipeline.model.config.max_model_length,
            )
        ),
        benchmark_name=task_name.split("|", 1)[0],
        num_samples=pipeline.task_sample_counts.get(task_name, len(native_details)),
        avg_k=float(avg_k),
        score=_native_score(metrics),
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


def _evaluate_pool(
    models: Sequence[ModelEndpoint], benchmarks: Sequence[BenchmarkSpec], options: dict[str, Any], log_level: int
) -> tuple[list[Score], list[Score]]:
    # Prototype: LightevalTask.load_datasets process workers. Native Pipeline
    # seeds, task formatters and graders must not share state across pools.
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
    return asyncio.run(evaluate(models, benchmarks, **options))


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

    if len(groups) > 1:
        options = {
            "prompt_template": prompt_template,
            "max_samples": max_samples,
            "test_mode": test_mode,
            "seed": seed,
            "scoreboard_token": scoreboard_token,
            "scoreboard_url": scoreboard_url,
            "no_upload": no_upload,
            "output_dir": output_dir,
        }
        loop = asyncio.get_running_loop()
        # Each process owns its entire benchmark queue, not one benchmark at
        # a time: fast pools never wait at a slow pool's benchmark barrier.
        with ProcessPoolExecutor(max_workers=len(groups), mp_context=get_context("spawn")) as executor:
            jobs = [
                loop.run_in_executor(
                    executor, _evaluate_pool, replicas, benchmarks, options, logging.getLogger().getEffectiveLevel()
                )
                for replicas in groups.values()
            ]
            results = await asyncio.gather(*jobs, return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise errors[0]
        return (
            [score for scores, _ in results for score in scores],
            [score for _, failures in results for score in failures],
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
    from lighteval.pipeline import ParallelismManager, PipelineParameters

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
        sampling = _sampling_config(benchmark_cot_mode, benchmark_max_tokens, seed)
        # NoCoT uses LightEval's loglikelihood/greedy-choice path.  It has no
        # RWKV generation prompt to render into the uploaded user message.
        if benchmark_cot_mode == "NoCoT":
            template = None
        elif prompt_template != "assistant":
            template = prompt_template
        else:
            difficult = ("gpqa", "math", "aime", "olympiad", "code")
            template = "bot" if any(keyword in benchmark.selector.casefold() for keyword in difficult) else "assistant"
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
                await asyncio.to_thread(RWKVPipeline, benchmark.selector, params, tracker, model_config=model_config)
            )

        # This worker owns one model pool; publish each completed benchmark
        # immediately, independently of the other model processes.
        for pipeline in pipelines:
            await asyncio.to_thread(pipeline.evaluate)
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
