# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import ast
import asyncio
import collections
import copy
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum, auto

import numpy as np
from tqdm import tqdm

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics import apply_metric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_loader import TransformersModel, load_model
from lighteval.models.model_output import (
    ModelResponse,
)
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import SamplingMethod
from lighteval.tasks.rwkv_prompt import (
    TaskPromptMode,
    apply_task_prompt_override,
)
from lighteval.utils.imports import is_package_available
from lighteval.utils.parallelism import test_all_gather
from lighteval.utils.utils import make_results_table, remove_reasoning_tags


if is_package_available("accelerate"):
    from accelerate import Accelerator, InitProcessGroupKwargs
else:
    from unittest.mock import Mock

    Accelerator = InitProcessGroupKwargs = Mock()

if is_package_available("nanotron"):
    from nanotron import distributed as dist
    from nanotron.parallel.context import ParallelContext

    from lighteval.models.nanotron.nanotron_model import NanotronLightevalModel


import logging


logger = logging.getLogger(__name__)


_RWKV_CHOICE_GENERATION_SIZE = 8192
_CHOICE_MARKUP = re.compile(r"\*\*|__|`+")
_CHOICE_SINGLE_LABEL = r"[A-Z](?![A-Z])"
_CHOICE_LABELS = rf"{_CHOICE_SINGLE_LABEL}(?:\s*(?:,|/|&|\+|\band\b)\s*{_CHOICE_SINGLE_LABEL})*"
_CHOICE_PATTERNS = (
    re.compile(
        rf"\\boxed\s*\{{\s*(?:\\(?:text|mathrm)\s*\{{\s*)?({_CHOICE_LABELS})\s*\}}?\s*\}}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?i:(?:final\s+answer|correct\s+answer|answer|choice|option|最终答案|正确答案|答案|选项)\s*"
        rf"(?:(?:choice|option)\s*)?(?:is\s*|would\s+be\s*|[是为]\s*|[:：=]\s*)"
        rf"(?:<letter>\s*)?(?:<\s*(?:answer|choice|b)\s*>\s*)?[\"'\[(]*\s*)"
        rf"({_CHOICE_LABELS})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?i:(?:final\s+answer|correct\s+answer|answer|choice|option|最终答案|正确答案|答案|选项)\s*"
        rf"(?:(?:choice|option)\s*)?(?:is\s*|would\s+be\s*|[是为]\s*|[:：=]\s*)<\s*)"
        rf"({_CHOICE_LABELS})\s*>",
        re.IGNORECASE,
    ),
    re.compile(
        rf"<\s*(?:answer|choice|b|final|final_answer|letter|span)(?:\s+[^>]*)?\s*>\s*"
        rf"({_CHOICE_LABELS})(?:\s*[.:]\s*[^<]*)?\s*"
        rf"</\s*(?:answer|choice|b|final|final_answer|letter|span)\s*>",
        re.IGNORECASE,
    ),
    re.compile(rf"<\s*({_CHOICE_LABELS})\s*>[^<]+</\s*[A-Z]\s*>", re.IGNORECASE),
    re.compile(rf"<\s*(?:answer|choice)\s*[:=]?\s*({_CHOICE_LABELS})\s*>", re.IGNORECASE),
    re.compile(
        rf"\\?[\"']answer\\?[\"']\s*:\s*\\?[\"']({_CHOICE_LABELS})\\?[\"']",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?i:(?:(?:choice|option)\s*)?\(?\s*)({_CHOICE_LABELS})"
        rf"(?i:\s*\)?\s+is\s+(?:the\s+)?(?:final\s+|correct\s+)?answer)",
    ),
)
_CHOICE_FALLBACK_PATTERNS = (
    re.compile(
        rf"(?i:\b(?:choose|select|pick)\s+(?:(?:choice|option|answer)\s*)?[:=]?\s*\(?\s*)"
        rf"({_CHOICE_LABELS})(?i:\s*\)?\b)",
    ),
    re.compile(
        rf"(?i:\b(?:corresponds?|maps?)\s+to\s+(?:(?:choice|option|answer)\s*)?\(?\s*)"
        rf"({_CHOICE_LABELS})(?i:\s*\)?\b)",
    ),
    re.compile(
        rf"(?i:\b(?:aligns?|matches?)\s+with\s+(?:(?:choice|option|answer)\s*)?\(?\s*)"
        rf"({_CHOICE_LABELS})(?i:\s*\)?\b)",
    ),
    re.compile(
        rf"(?i:\b(?:therefore|thus|hence|so|consequently)[,:]?\s+(?:the\s+)?"
        rf"(?:(?:correct|final)\s+)?(?:answer|choice|option)\s+(?:is|would\s+be)\s+\(?\s*)"
        rf"({_CHOICE_LABELS})(?i:\s*\)?\b)",
    ),
    re.compile(rf"(?i:\b\(?\s*)({_CHOICE_LABELS})(?i:\s*\)?\s+(?:the\s+)?correct\b)"),
)
_CHOICE_BARE = re.compile(
    rf"\s*(?:final\s+answer\s*[:=]?\s*)?[\[(<]*({_CHOICE_LABELS})[\])>]*"
    r"(?:\s*[.:：](?:\s*\S.*)?)?\s*",
    re.IGNORECASE,
)
_CHOICE_TEXT_EXPLICIT = re.compile(
    r"(?:final\s+answer|correct\s+answer|answer|最终答案|正确答案|答案)\s*"
    r"(?:is\s*|would\s+be\s*|是\s*|[:：=]\s*)(.+?)(?:[.。]\s*$|$)",
    re.IGNORECASE | re.MULTILINE,
)


def _choice_gold_indices(doc) -> tuple[int, ...] | None:
    gold = doc.gold_index
    indices = tuple(gold) if isinstance(gold, (list, tuple)) else (gold,)
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(
            not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(doc.choices)
            for index in indices
        )
    ):
        return None
    return tuple(sorted(indices))


def _is_choice(doc) -> bool:
    if not (
        isinstance(doc.query, str)
        and isinstance(doc.choices, list)
        and 2 <= len(doc.choices) <= 26
        and all(isinstance(choice, str) and choice.strip() for choice in doc.choices)
        and _choice_gold_indices(doc) is not None
    ):
        return False
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(doc.choices)]
    native_letter_choice = SamplingMethod.GENERATIVE in doc.sampling_methods and [
        choice.strip().upper() for choice in doc.choices
    ] == list(labels)
    return SamplingMethod.LOGPROBS in doc.sampling_methods or native_letter_choice


def _convert_choice(doc) -> None:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(doc.choices)]
    gold_indices = _choice_gold_indices(doc)
    assert gold_indices is not None
    answer_format = "<letter>" if len(gold_indices) == 1 else "<letters separated by commas>"
    answer_instruction = f'After reasoning, end with "Answer: {answer_format}".'
    if answer_instruction not in doc.query:
        if [choice.strip().upper() for choice in doc.choices] == list(labels):
            doc.query = f"{doc.query.rstrip()}\n\n{answer_instruction}"
        else:
            options = "\n".join(
                f"{label}. {choice.strip()}" for label, choice in zip(labels, doc.choices, strict=True)
            )
            doc.query = f"{doc.query.rstrip()}\n\n{options}\n\n{answer_instruction}"
    doc.sampling_methods = list(
        dict.fromkeys(
            SamplingMethod.GENERATIVE if method == SamplingMethod.LOGPROBS else method
            for method in doc.sampling_methods
        )
    )
    doc.generation_size = _RWKV_CHOICE_GENERATION_SIZE
    doc.stop_sequences = []
    doc.specific = dict(doc.specific or {}, rwkv_choice=True)


class _ChoiceExactMatches(SampleLevelComputation):
    def compute(self, doc, model_response, **_kwargs) -> float:
        gold_indices = _choice_gold_indices(doc)
        if gold_indices is None:
            return 0.0
        expected = _canonical_choice_answer(gold_indices, doc.choices)
        return float(any(prediction == expected for prediction in model_response.final_text))


def _choice_metrics(metric):
    names = (metric.metric_name,) if isinstance(metric.metric_name, str) else tuple(metric.metric_name)
    grouped = not isinstance(metric.metric_name, str)
    return tuple(
        SampleLevelMetric(
            metric_name=name,
            sample_level_fn=_ChoiceExactMatches(),
            category=SamplingMethod.GENERATIVE,
            corpus_level_fn=metric.corpus_level_fn[name] if grouped else metric.corpus_level_fn,
            higher_is_better=metric.higher_is_better[name] if grouped else metric.higher_is_better,
        )
        for name in names
    )


def _parse_choice_labels(value: str, labels: str) -> tuple[int, ...] | None:
    normalized = re.sub(r"\band\b", ",", value.upper())
    parts = [part.strip() for part in re.split(r"[,/&+]", normalized)]
    if not parts or any(not part for part in parts):
        return None
    if len(parts) == 1 and " " in parts[0]:
        parts = parts[0].split()
    if any(len(part) != 1 or part not in labels for part in parts) or len(set(parts)) != len(parts):
        return None
    return tuple(sorted(labels.index(part) for part in parts))


def _canonical_choice_answer(indices: tuple[int, ...], choices: list[str]) -> str:
    selected = [choices[index] for index in indices]
    if len(selected) == 1:
        return selected[0]
    return json.dumps(selected, ensure_ascii=False, separators=(",", ":"))


def _normalized_choice_text(value: str) -> str:
    normalized = " ".join(_CHOICE_MARKUP.sub("", value).replace("<", "").replace(">", "").casefold().split())
    normalized = re.sub(r"(?<=\d)\s*(?:[-–—]|\band\b)\s*(?=\d)", "-", normalized)
    normalized = re.sub(r"(?<=\d)\s+(?=[a-z])", "", normalized)
    normalized = re.sub(r"\s+([%/.,，。])", r"\1", normalized)
    return normalized.strip(" \t\r\n.。,:：;；!?！？()[]{}\"'")


def _choice_texts(query: str | None, choices: list[str], labels: str) -> list[str]:
    if [choice.strip().upper() for choice in choices] != list(labels) or not isinstance(query, str):
        return choices
    parsed = {}
    for match in re.finditer(r"(?m)^\s*([A-Z])\s*[.)。、]\s*(.+?)\s*$", query, re.IGNORECASE):
        label = match.group(1).upper()
        if label in labels:
            parsed[label] = match.group(2)
    return [parsed.get(label, choice) for label, choice in zip(labels, choices, strict=True)]


def _choice_text_answer(
    text: str,
    choices: list[str],
    labels: str,
    query: str | None,
) -> tuple[int, ...] | None:
    normalized_text = _normalized_choice_text(text)
    matched = []
    choice_texts = _choice_texts(query, choices, labels)
    for index, (label, choice) in enumerate(zip(labels, choice_texts, strict=True)):
        normalized_choice = _normalized_choice_text(choice)
        without_label = re.sub(rf"^\s*(?:\({label}\)|{label}[.)、])\s*", "", normalized_choice, flags=re.I)
        variants = {variant for variant in (normalized_choice, without_label) if variant}
        matching_variants = [variant for variant in variants if variant in normalized_text]
        if matching_variants:
            matched.append((index, max(matching_variants, key=len)))
    if not matched:
        return None
    maximal = [
        index for index, variant in matched if not any(variant != other and variant in other for _, other in matched)
    ]
    return (maximal[0],) if len(maximal) == 1 else None


def _choice_payload_answer(
    text: str,
    choices: list[str],
    labels: str,
    query: str | None,
) -> tuple[int, tuple[int, ...]] | None:
    choice_texts = _choice_texts(query, choices, labels)
    matches = []
    for match in _CHOICE_TEXT_EXPLICIT.finditer(text):
        payload = _normalized_choice_text(match.group(1))
        if not payload or payload in {"letter", "answer_letter"}:
            continue
        candidates = []
        for index, choice in enumerate(choice_texts):
            normalized_choice = _normalized_choice_text(choice)
            without_label = re.sub(
                rf"^\s*(?:\({labels[index]}\)|{labels[index]}[.)、])\s*",
                "",
                normalized_choice,
                flags=re.I,
            )
            if payload in {normalized_choice, without_label} or (
                len(payload) > 1 and (payload in without_label or without_label in payload)
            ):
                candidates.append(index)
        if len(candidates) == 1:
            matches.append((match.start(), (candidates[0],)))
    return max(matches, key=lambda item: item[0]) if matches else None


def _choice_answer(raw: str, choices: list[str], query: str | None = None) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""
    _, think_closed, suffix = raw.rpartition("</think>")
    answer_text = _CHOICE_MARKUP.sub("", suffix if think_closed else raw)
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)]
    matches = [
        (match.start(), parsed)
        for pattern in (*_CHOICE_PATTERNS, *_CHOICE_FALLBACK_PATTERNS)
        for match in pattern.finditer(answer_text)
        if (parsed := _parse_choice_labels(match.group(1), labels)) is not None
    ]
    if payload := _choice_payload_answer(answer_text, choices, labels, query):
        matches.append(payload)
    for line_match in re.finditer(r"(?m)^.*$", answer_text):
        if match := _CHOICE_BARE.fullmatch(line_match.group()):
            if parsed := _parse_choice_labels(match.group(1), labels):
                matches.append((line_match.start(), parsed))
    if matches:
        return _canonical_choice_answer(max(matches, key=lambda item: item[0])[1], choices)
    if parsed := _choice_text_answer(answer_text, choices, labels, query):
        return _canonical_choice_answer(parsed, choices)
    return ""


class ParallelismManager(Enum):
    ACCELERATE = auto()
    NANOTRON = auto()
    TGI = auto()
    OPENAI = auto()
    VLLM = auto()
    CUSTOM = auto()
    NONE = auto()
    SGLANG = auto()


@dataclass
class PipelineParameters:
    launcher_type: ParallelismManager
    # Env parameters
    job_id: int = 0
    dataset_loading_processes: int = 1
    nanotron_checkpoint_path: str | None = None  # only for nanotron models
    # Dataset
    custom_tasks_directory: str | None = None
    num_fewshot_seeds: int = 1
    max_samples: int | None = None
    cot_prompt: str | None = None
    remove_reasoning_tags: bool = True
    reasoning_tags: str | list[tuple[str, str]] = "[('<think>', '</think>')]"
    load_responses_from_details_date_id: str | None = None
    bootstrap_iters: int = 1000
    load_tasks_multilingual: bool = False
    task_prompt: str | None = None
    task_prompt_mode: str | TaskPromptMode | None = None
    convert_logprob_choices_to_generation: bool = False

    def __post_init__(self):  # noqa C901
        task_prompt_configured = self.task_prompt is not None
        task_prompt_mode_configured = self.task_prompt_mode is not None
        if task_prompt_configured != task_prompt_mode_configured:
            raise ValueError("task_prompt and task_prompt_mode must be configured together")
        if task_prompt_configured:
            if not isinstance(self.task_prompt, str):
                raise TypeError("task_prompt must be a string")
            try:
                self.task_prompt_mode = TaskPromptMode(self.task_prompt_mode)
            except (TypeError, ValueError) as error:
                choices = ", ".join(item.value for item in TaskPromptMode)
                raise ValueError(f"task_prompt mode must be one of: {choices}") from error
        if not isinstance(self.reasoning_tags, list):
            try:
                self.reasoning_tags = ast.literal_eval(self.reasoning_tags)
            except ValueError as e:
                raise ValueError(
                    "reasoning_tags must be a list of pair tuples, e.g. [('start_tag', 'end_tag'), ...]. "
                    f"Got {self.reasoning_tags} instead, which caused parsing error {e}."
                )

        # Make sure format is correct
        if not all(isinstance(tag, tuple) and len(tag) == 2 for tag in self.reasoning_tags):
            raise ValueError(
                "reasoning_tags must be a list of pair tuples, e.g. [('start_tag', 'end_tag'), ...]. "
                f"Got {self.reasoning_tags} instead."
            )


class Pipeline:
    def __init__(
        self,
        tasks: str,
        pipeline_parameters: PipelineParameters,
        evaluation_tracker: EvaluationTracker,
        model_config: ModelConfig | None = None,
        model=None,
        metric_options=None,
    ):
        if not (model or model_config):
            raise ValueError("Must provide either a model or model config when creating a pipeline.")

        self.pipeline_parameters = pipeline_parameters
        if self.pipeline_parameters.max_samples:
            logger.warning(
                "--max_samples WAS SET. THESE NUMBERS ARE ONLY PARTIAL AND SHOULD NOT BE USED FOR COMPARISON UNLESS YOU KNOW WHAT YOU ARE DOING."
            )

        self.launcher_type = self.pipeline_parameters.launcher_type
        self._metric_options = metric_options or {}
        self.evaluation_tracker = evaluation_tracker

        # We init tasks first to fail fast if one is badly defined
        self._init_random_seeds()
        self._init_tasks_and_requests(tasks=tasks)

        self.model_config = model_config
        self.accelerator, self.parallel_context = self._init_parallelism_manager()
        self.model = self._init_model(model_config, model)
        # Must occur after model and task init
        self.model._cache._init_registry(self.registry)
        # Must occur after model init
        self._init_accelerator_seeds()

        self.evaluation_tracker.general_config_logger.log_model_info(model_config=self.model.config)

        # Final results
        self.final_dict: dict | None = None

    def _init_parallelism_manager(self):
        accelerator, parallel_context = None, None
        if self.launcher_type == ParallelismManager.ACCELERATE:
            if not is_package_available("accelerate"):
                raise ValueError("You are trying to launch an accelerate model, but accelerate is not installed")
            accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
            test_all_gather(accelerator=accelerator)
        elif self.launcher_type == ParallelismManager.NANOTRON:
            if not is_package_available("nanotron"):
                raise ValueError("You are trying to launch a nanotron model, but nanotron is not installed")
            dist.initialize_torch_distributed()
            parallel_context = ParallelContext(
                tensor_parallel_size=self.model_config.lighteval_config.parallelism.tp,
                pipeline_parallel_size=self.model_config.lighteval_config.parallelism.pp,
                data_parallel_size=self.model_config.lighteval_config.parallelism.dp,
            )
            test_all_gather(parallel_context=parallel_context)

        return accelerator, parallel_context

    def _init_model(self, model_config, model):
        logger.info("--- LOADING MODEL ---")

        if model is not None and model_config is not None:
            if isinstance(model, LightevalModel):
                raise ValueError(
                    "You are trying to provide both a LightevalModel and a model config. Please provide only one of them."
                )
            return TransformersModel.from_model(
                model=model,
                config=model_config,
                accelerator=self.accelerator,
            )

        elif model is not None:
            if isinstance(model, LightevalModel):
                return model
            raise ValueError("If not providing a model_config, you need to provide a Lighteval model.")

        elif model_config is not None:
            if self.parallel_context:
                return NanotronLightevalModel(
                    checkpoint_path=os.path.dirname(self.pipeline_parameters.nanotron_checkpoint_path)
                    if self.pipeline_parameters.nanotron_checkpoint_path
                    else "",
                    nanotron_config=model_config,
                    parallel_context=self.parallel_context,
                    debug_one_layer_model=False,
                    model_class=None,
                )
            else:
                return load_model(config=model_config)

    def _init_tasks_and_requests(self, tasks: str):
        logger.info("--- LOADING TASKS ---")

        # The registry contains all the potential tasks
        self.registry = Registry(
            tasks=tasks,
            load_multilingual=self.pipeline_parameters.load_tasks_multilingual,
            custom_tasks=self.pipeline_parameters.custom_tasks_directory,
        )

        # load the tasks from the configs and their datasets
        self.tasks_dict: dict[str, LightevalTask] = self.registry.load_tasks()
        LightevalTask.load_datasets(self.tasks_dict, self.pipeline_parameters.dataset_loading_processes)
        self.documents_dict = {
            task.full_name: task.get_docs(self.pipeline_parameters.max_samples) for _, task in self.tasks_dict.items()
        }
        if self.pipeline_parameters.task_prompt is not None:
            self._apply_task_prompt_override()
        logger.info(
            "convert_logprob_choices_to_generation: %s, recommended value for RWKV models: True",
            self.pipeline_parameters.convert_logprob_choices_to_generation,
        )
        if self.pipeline_parameters.convert_logprob_choices_to_generation:
            self._prepare_choice_tasks()

        self.sampling_docs = collections.defaultdict(list)
        for _, docs in self.documents_dict.items():
            for doc in docs:
                for sampling in doc.sampling_methods:
                    self.sampling_docs[sampling].append(doc)

        # If there are metric_options defined from the yaml file,
        # review if they have to be updated.
        if self._metric_options:
            self._update_num_samples(list(self.tasks_dict.values()))

        self.evaluation_tracker.task_config_logger.log(self.tasks_dict)

    def _prepare_choice_tasks(self):
        for task in self.tasks_dict.values():
            docs = self.documents_dict[task.full_name]
            unsupported_logprob_docs = [
                doc for doc in docs if SamplingMethod.LOGPROBS in doc.sampling_methods and not _is_choice(doc)
            ]
            if unsupported_logprob_docs:
                raise ValueError(
                    f"task {task.full_name} contains {len(unsupported_logprob_docs)} "
                    "logprob documents which cannot be converted to generation"
                )

            choice_docs = [doc for doc in docs if _is_choice(doc)]
            if not choice_docs:
                continue

            for doc in choice_docs:
                _convert_choice(doc)
            task.config = copy.copy(task.config)
            task.config.original_num_docs = len(docs)
            task.config.effective_num_docs = len(docs)
            converted_metrics = []
            for metric in task.metrics:
                if metric.category == SamplingMethod.LOGPROBS:
                    converted_metrics.extend(_choice_metrics(metric))
                else:
                    converted_metrics.append(metric)
            task.metrics = tuple(converted_metrics)
            task.config.metrics = task.metrics
            task.sampling_methods = list(dict.fromkeys(metric.category for metric in task.metrics))

    def _apply_task_prompt_override(self):
        task_prompt = self.pipeline_parameters.task_prompt
        task_prompt_mode = self.pipeline_parameters.task_prompt_mode
        assert task_prompt is not None
        assert task_prompt_mode is not None
        for task in self.tasks_dict.values():
            task.config = copy.copy(task.config)
            identities = [
                apply_task_prompt_override(
                    doc,
                    task_prompt,
                    task_prompt_mode,
                )
                for doc in self.documents_dict[task.full_name]
            ]
            task.config.configured_task_prompt = task_prompt
            task.config.task_prompt_mode = TaskPromptMode(task_prompt_mode).value
            task.config.task_prompt_digests = sorted({identity.digest for identity in identities})

    def _update_num_samples(self, tasks: list[LightevalTask]):
        """Helper function to update the num_samples of a given metric via the yaml file.
        As it has to be done at the metric level, it's better to update the value per metric.
        It will add a num_samples to the already defined metrics' num_samples if defined in the yaml file.
        As later when constructing the requests the max is taken over the num_samples, this is valid.
        """
        for task in tasks:
            for metric in task.metrics:
                if metric_data := self._metric_options.get(metric.metric_name, None):
                    num_samples = metric_data.get("num_samples", None)
                    if num_samples:
                        task.num_samples = [num_samples]

    def _init_random_seeds(self):
        logger.info("--- INIT SEEDS ---")
        random.seed(1234)
        np.random.seed(1234)

    def _init_accelerator_seeds(self):
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        if self.parallel_context is not None:
            dist.barrier()

    def is_main_process(self):
        if self.accelerator:
            return self.accelerator.is_main_process
        if self.parallel_context:
            return dist.get_rank(self.parallel_context.world_pg) == 0
        return True

    def evaluate(self):
        self.evaluation_tracker.general_config_logger.log_args_info(
            num_fewshot_seeds=self.pipeline_parameters.num_fewshot_seeds,
            max_samples=self.pipeline_parameters.max_samples,
            job_id=str(self.pipeline_parameters.job_id),
        )

        if self.pipeline_parameters.load_responses_from_details_date_id:
            try:
                outputs = self._load_responses_from_details()
            except FileNotFoundError as e:
                logger.warning(
                    f"No responses found for {self.pipeline_parameters.load_responses_from_details_date_id} in details directory: {e}. Running model instead."
                )
                outputs = self._run_model()
        else:
            outputs = self._run_model()

        if self.is_main_process():
            self._post_process_outputs(outputs)
            self._compute_metrics(outputs)

            self.evaluation_tracker.general_config_logger.log_end_time()
            self.evaluation_tracker.metrics_logger.aggregate(
                task_dict=self.tasks_dict, bootstrap_iters=self.pipeline_parameters.bootstrap_iters
            )
            self.evaluation_tracker.details_logger.aggregate()
            for task_name, summary in self.evaluation_tracker.details_logger.compiled_details.items():
                self.evaluation_tracker.metrics_logger.metric_aggregated[task_name].update(
                    n_samples=summary.n_samples,
                    n_completions=summary.n_completions,
                    n_truncated=summary.n_truncated,
                    truncation_rate=summary.truncation_rate,
                )

    async def _run_model_async(self):
        outputs = {}
        for sampling_method, docs in self.sampling_docs.items():
            logger.info(f"Running {sampling_method} requests")
            match sampling_method:
                case SamplingMethod.GENERATIVE:
                    model_outputs = await self.model.greedy_until(docs)
                    outputs[sampling_method] = model_outputs
                case SamplingMethod.LOGPROBS:
                    model_outputs = await self.model.loglikelihood(docs)
                    outputs[sampling_method] = model_outputs

        return outputs

    def _run_model_sync(self):
        # Running all requests depending on the model call type (log likelihood, generative, ...)
        # to be able to batch them
        outputs = {}
        for sampling_method, docs in self.sampling_docs.items():
            logger.info(f"Running {sampling_method} requests")
            match sampling_method:
                case SamplingMethod.GENERATIVE:
                    model_outputs = self.model.greedy_until(docs)
                    outputs[sampling_method] = model_outputs
                case SamplingMethod.LOGPROBS:
                    model_outputs = self.model.loglikelihood(docs)
                    outputs[sampling_method] = model_outputs
                case SamplingMethod.PERPLEXITY:
                    model_outputs = self.model.loglikelihood_rolling(docs)
                    outputs[sampling_method] = model_outputs

        return outputs

    def _run_model(self):
        # Running all requests depending on the model call type (log likelihood, generative, ...)
        # to be able to batch them
        logger.info("--- RUNNING MODEL ---")

        if self.model.is_async:
            outputs = asyncio.run(self._run_model_async())
        else:
            outputs = self._run_model_sync()

        # Cleaning up the model before running metrics
        self.model.cleanup()

        return outputs

    def _post_process_outputs(self, sampling_method_responses: dict[str, list[ModelResponse]]):
        # Removes reasoning tags if needed
        logger.info("--- POST-PROCESSING MODEL RESPONSES ---")

        if self.pipeline_parameters.remove_reasoning_tags:
            for _, responses in sampling_method_responses.items():
                for response in responses:
                    response.text_post_processed = [
                        remove_reasoning_tags(
                            text=text,
                            tag_pairs=self.pipeline_parameters.reasoning_tags,
                        )
                        for text in response.text
                    ]

        if not self.pipeline_parameters.convert_logprob_choices_to_generation:
            return

        responses = sampling_method_responses.get(SamplingMethod.GENERATIVE, [])
        for doc, response in zip(self.sampling_docs.get(SamplingMethod.GENERATIVE, []), responses):
            if not isinstance(doc.specific, dict) or doc.specific.get("rwkv_choice") is not True:
                continue
            response.text_post_processed = [
                ""
                if index < len(response.finish_reasons) and response.finish_reasons[index] == "length"
                else _choice_answer(text, doc.choices, doc.query)
                for index, text in enumerate(response.text)
            ]

    def _compute_metrics(self, sampling_method_responses: dict[str, list[ModelResponse]]):
        # To compute the metrics we first group the samples and task and then by metrics.
        # This way we can batch the metrics computation for each task and metric category

        # This variable will hold the samples grouped by task and metric category
        # example:
        # task_metric_category_groups = {
        #     "gsm8k_1": {
        #         "GENERATIVE": [
        #             (doc1, response1), (doc2, response2), ...,
        #         }
        #         "LOGLIKELIHOOD": [
        #             (doc1, response1), (doc2, response2), ...,
        #         ]
        logger.info("--- COMPUTING METRICS ---")
        task_metric_category_groups = collections.defaultdict(lambda: collections.defaultdict(list))

        for sampling_method, model_responses in sampling_method_responses.items():
            for doc, model_reponse in zip(self.sampling_docs[sampling_method], model_responses):
                task_metric_category_groups[doc.task_name][sampling_method].append((doc, model_reponse))

        for task_name, samples_per_method in task_metric_category_groups.items():
            task: LightevalTask = self.tasks_dict[task_name]
            for sampling_method, samples in samples_per_method.items():
                metric_category_metrics = [metric for metric in task.metrics if metric.category == sampling_method]

                docs = [doc for doc, _ in samples]
                responses = [response for _, response in samples]

                outputs = apply_metric(
                    docs=docs,
                    responses=responses,
                    metrics=metric_category_metrics,
                )

                for output, doc, response in zip(outputs, docs, responses):
                    self.evaluation_tracker.metrics_logger.log(task_name, output)
                    self.evaluation_tracker.details_logger.log(task_name, doc, response, output)

    def _load_responses_from_details(self):
        logger.info("--- LOADING RESPONSES FROM DETAILS ---")
        model_responses = {}
        tasks_names = list(self.tasks_dict.keys())
        sampling_methods = list(self.sampling_docs.keys())

        if len(sampling_methods) > 1:
            raise ValueError(
                "Loading responses from details when there are multiple request types is currently not supported"
            )

        assert self.pipeline_parameters.load_responses_from_details_date_id is not None

        details_datasets = self.evaluation_tracker.load_details_datasets(
            self.pipeline_parameters.load_responses_from_details_date_id, tasks_names
        )

        for _, dataset in tqdm(details_datasets.items(), desc="Loading responses from details for tasks"):
            for sampling_method in sampling_methods:
                model_responses[sampling_method] = [
                    ModelResponse(**model_response["model_response"]) for model_response in dataset
                ]

        return model_responses

    def save_and_push_results(self):
        logger.info("--- SAVING AND PUSHING RESULTS ---")
        if self.is_main_process():
            self.evaluation_tracker.save()

    def _init_final_dict(self):
        if self.is_main_process():
            if self.final_dict is None:
                self.final_dict = self.evaluation_tracker.generate_final_dict()

    def show_results(self):
        logger.info("--- DISPLAYING RESULTS ---")
        self._init_final_dict()
        if self.is_main_process():
            print(make_results_table(self.final_dict))

    def get_results(self):
        self._init_final_dict()
        return self.final_dict

    def get_details(self):
        return self.evaluation_tracker.details_logger.details
