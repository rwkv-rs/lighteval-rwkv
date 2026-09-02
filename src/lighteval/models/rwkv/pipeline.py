# MIT License

from __future__ import annotations

import asyncio
import logging
import math
import queue
import threading
import time
from collections import defaultdict
from copy import copy
from typing import Mapping

import typer

from lighteval.metrics.metrics_sample import SampleLevelComputation, SamplingMetric
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.rwkv.http_model import MAX_NEW_TOKENS
from lighteval.pipeline import Pipeline
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.rwkv_answer_extractor import (
    convert_rwkv_choice,
    extract_rwkv_choice_answer,
    is_rwkv_choice,
    rwkv_choice_metrics,
)
from lighteval.tasks.rwkv_prompt import TaskPromptMode, apply_task_prompt_override


_MIN_COMPLETIONS = 5000
_LARGE_BENCHMARK_SIZE = 50_000
_LARGE_BENCHMARK_FRACTION = 0.2
logger = logging.getLogger(__name__)


def _open_think_answer(text: str) -> str:
    _, closed, answer = text.partition("</think>")
    return answer.replace("</think>", "") if closed else text


class RWKVAvgAtK(SampleLevelComputation):
    """Average one native task scorer over exactly k independent completions."""

    def __init__(self, k: int, metric) -> None:
        self.k = k
        self.metric = metric

    def __str__(self) -> str:
        return f"RWKVAvgAtK(k={self.k})"

    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        return sum(self.score_rollout(doc, model_response[index]) for index in range(self.k)) / self.k

    def score_rollout(self, doc: Doc, model_response) -> float:
        if model_response.finish_reasons == ["length"]:
            return 0.0
        scorer = self.metric.sample_level_fn
        if isinstance(scorer, SamplingMetric):
            return float(scorer.compute_score(doc, model_response))
        value = next(iter(self.metric.compute_sample(doc=doc, model_response=model_response).values()))
        if isinstance(value, (list, tuple)):
            return sum(float(item) for item in value) / len(value)
        return float(value)

    def extract_rollout_answer(self, doc: Doc, model_response) -> str:
        if model_response.finish_reasons == ["length"]:
            return ""
        scorer = self.metric.sample_level_fn
        if extractor := getattr(scorer, "extract_answer", None):
            return extractor(doc, model_response)
        return model_response.final_text[0] if model_response.final_text else ""


def _evaluation_plan(num_docs: int) -> tuple[int, int, str]:
    """Return evaluated documents, completions per document, and the only metric name."""
    if num_docs > _LARGE_BENCHMARK_SIZE:
        return int(num_docs * _LARGE_BENCHMARK_FRACTION), 1, "avg@0.2"
    k = 1
    while k * num_docs <= _MIN_COMPLETIONS:
        k *= 2
    return num_docs, k, f"avg@{k}"


def _configure_task_evaluation_plan(pipeline, task, docs) -> list[Doc]:
    original_num_docs = len(task.eval_docs())
    max_samples = pipeline.pipeline_parameters.max_samples
    if max_samples is None:
        effective_num_docs, k, metric_name = _evaluation_plan(original_num_docs)
    else:
        task_max_samples = getattr(pipeline, "_task_max_samples", {}).get(task.full_name, max_samples)
        effective_num_docs, k, metric_name = min(original_num_docs, task_max_samples), 1, "avg@1"
    docs = docs[:effective_num_docs]
    source_metric = task.metrics[0]
    source_name = source_metric.metric_name[0] if isinstance(source_metric.metric_name, (list, tuple)) else None
    corpus_level_fn = source_metric.corpus_level_fn[source_name] if source_name else source_metric.corpus_level_fn
    higher_is_better = source_metric.higher_is_better[source_name] if source_name else source_metric.higher_is_better
    task.metrics = (
        SampleLevelMetric(
            metric_name=metric_name,
            sample_level_fn=RWKVAvgAtK(k, source_metric),
            category=SamplingMethod.GENERATIVE,
            corpus_level_fn=corpus_level_fn,
            higher_is_better=higher_is_better,
        ),
    )
    task.num_samples = [1, k]
    for doc in docs:
        doc.num_samples = k
    task.config = copy(task.config)
    task.config.metrics = task.metrics
    task.config.original_num_docs = original_num_docs
    task.config.effective_num_docs = len(docs)
    task.sampling_methods = [SamplingMethod.GENERATIVE]
    return docs


class RWKVPipeline(Pipeline):
    """Run each native LightEval task as soon as its dataset is ready."""

    _DATASET_LOADERS = 8

    def __init__(self, *args, selector_tasks: Mapping[str, tuple[str, ...]], task_max_samples=None, **kwargs) -> None:
        self._configured_selector_tasks = selector_tasks
        self._configured_task_max_samples = task_max_samples
        super().__init__(*args, **kwargs)

    def _init_tasks_and_requests(self, tasks: str) -> None:
        logger.info("--- LOADING TASKS ---")
        self.registry = Registry(
            tasks=tasks,
            load_multilingual=self.pipeline_parameters.load_tasks_multilingual,
            custom_tasks=self.pipeline_parameters.custom_tasks_directory,
        )
        self.tasks_dict = self.registry.load_tasks()
        full_names = {task_name.rsplit("|", 1)[0]: task_name for task_name in self.tasks_dict}
        self._selector_tasks = {
            selector: tuple(full_names[leaf] for leaf in leaves if leaf in full_names)
            for selector, leaves in self._configured_selector_tasks.items()
        }
        self._task_selectors = {
            task_name: selector for selector, task_names in self._selector_tasks.items() for task_name in task_names
        }
        if self._configured_task_max_samples is None:
            self._task_max_samples = {}
            self._task_names = tuple(self.tasks_dict)
        else:
            self._task_max_samples = {
                full_names[leaf]: budget
                for leaf, budget in self._configured_task_max_samples.items()
                if budget and leaf in full_names
            }
            self.tasks_dict = {
                task_name: task for task_name, task in self.tasks_dict.items() if task_name in self._task_max_samples
            }
            self._task_names = tuple(self.tasks_dict)
            self._selector_tasks = {
                selector: tuple(task_name for task_name in task_names if task_name in self._task_max_samples)
                for selector, task_names in self._selector_tasks.items()
            }
            self._task_selectors = {
                task_name: selector
                for selector, task_names in self._selector_tasks.items()
                for task_name in task_names
            }
        self.documents_dict = {}
        self.sampling_docs = defaultdict(list)
        self.task_callback = None
        self._datasets_loaded = 0
        self._all_datasets_ready_at = None
        if self._metric_options:
            self._update_num_samples(list(self.tasks_dict.values()))

    def _prepare_task_documents(self, task):
        max_samples = self._task_max_samples.get(task.full_name, self.pipeline_parameters.max_samples)
        docs = task.get_docs(max_samples)
        if self.pipeline_parameters.task_prompt is not None:
            self._apply_task_prompt_to_docs(task, docs)
        self._prepare_truthfulqa_mc1(task, docs)
        self._prepare_choice_task(task, docs)
        if self.model.config.cot_mode == "open_think":
            self._prepare_open_think_task(task, docs)
        return docs

    @staticmethod
    def _prepare_truthfulqa_mc1(task, docs) -> None:
        if task.full_name.rsplit("|", 1)[0] != "truthfulqa:mc":
            return
        for doc in docs:
            len_mc1 = doc.specific["len_mc1"]
            gold_indices = doc.gold_index if isinstance(doc.gold_index, list) else [doc.gold_index]
            doc.choices = doc.choices[:len_mc1]
            doc.gold_index = next(index for index in gold_indices if index < len_mc1)
            doc.specific = dict(doc.specific, rwkv_truthfulqa_metric="mc1")

    @staticmethod
    def _prepare_open_think_task(task, docs) -> None:
        for doc in docs:
            doc.generation_size = MAX_NEW_TOKENS
            doc.stop_sequences = []
        task.config = copy(task.config)
        task.config.generation_size = MAX_NEW_TOKENS
        task.config.stop_sequence = []

    def _apply_task_prompt_to_docs(self, task, docs) -> None:
        task_prompt = self.pipeline_parameters.task_prompt
        task_prompt_mode = self.pipeline_parameters.task_prompt_mode
        task.config = copy(task.config)
        identities = [apply_task_prompt_override(doc, task_prompt, task_prompt_mode) for doc in docs]
        task.config.configured_task_prompt = task_prompt
        task.config.task_prompt_mode = TaskPromptMode(task_prompt_mode).value
        task.config.task_prompt_digests = sorted({identity.digest for identity in identities})

    @staticmethod
    def _prepare_choice_task(task, docs) -> None:
        unsupported = [
            doc for doc in docs if SamplingMethod.LOGPROBS in doc.sampling_methods and not is_rwkv_choice(doc)
        ]
        if unsupported:
            raise ValueError(
                f"task {task.full_name} contains {len(unsupported)} "
                "logprob documents which cannot be converted to generation"
            )
        choice_docs = [doc for doc in docs if is_rwkv_choice(doc)]
        if not choice_docs:
            return
        for doc in choice_docs:
            convert_rwkv_choice(doc)
        task.config = copy(task.config)
        task.config.original_num_docs = len(docs)
        task.config.effective_num_docs = len(docs)
        converted_metrics = tuple(
            converted
            for metric in task.metrics
            for converted in (rwkv_choice_metrics(metric) if metric.category == SamplingMethod.LOGPROBS else (metric,))
        )
        task.metrics = (
            converted_metrics[:1] if task.full_name.rsplit("|", 1)[0] == "truthfulqa:mc" else converted_metrics
        )
        task.config.metrics = task.metrics
        task.sampling_methods = list(dict.fromkeys(metric.category for metric in task.metrics))

    def _post_process_outputs(self, sampling_method_responses) -> None:
        if self.model.config.cot_mode != "open_think":
            super()._post_process_outputs(sampling_method_responses)
        else:
            logger.info("--- POST-PROCESSING MODEL RESPONSES ---")
            for responses in sampling_method_responses.values():
                for response in responses:
                    response.text_post_processed = [_open_think_answer(text) for text in response.text]

        responses = sampling_method_responses.get(SamplingMethod.GENERATIVE, [])
        for doc, response in zip(self.sampling_docs.get(SamplingMethod.GENERATIVE, []), responses):
            if not isinstance(doc.specific, dict) or doc.specific.get("rwkv_choice") is not True:
                continue
            response.text_post_processed = [
                ""
                if index < len(response.finish_reasons) and response.finish_reasons[index] == "length"
                else extract_rwkv_choice_answer(text, doc.choices, doc.query)
                for index, text in enumerate(response.text)
            ]

    def _index_task_documents(self, task, docs) -> None:
        self.documents_dict[task.full_name] = docs

    def evaluate(self) -> None:
        self.evaluation_tracker.general_config_logger.log_args_info(
            num_fewshot_seeds=self.pipeline_parameters.num_fewshot_seeds,
            max_samples=self.pipeline_parameters.max_samples,
            job_id=str(self.pipeline_parameters.job_id),
        )
        scoring_queue = queue.Queue()
        evaluation_errors = []

        # Native math scorers use process-main-thread signals. Keep scoring here
        # while one background event loop continues dataset and HTTP work.
        def run_evaluation() -> None:
            try:
                asyncio.run(
                    self._evaluate_tasks(
                        lambda task_name, sampling_docs, outputs: self._submit_score(
                            scoring_queue,
                            task_name,
                            sampling_docs,
                            outputs,
                        )
                    )
                )
            except BaseException as error:
                evaluation_errors.append(error)
            finally:
                scoring_queue.put(None)

        evaluation_thread = threading.Thread(target=run_evaluation, name="rwkv-http-event-loop")
        evaluation_thread.start()
        scoring_error = None
        while (scoring := scoring_queue.get()) is not None:
            task_name, sampling_docs, outputs, future = scoring
            if scoring_error is None:
                try:
                    self._score_task(task_name, sampling_docs, outputs)
                except BaseException as error:
                    scoring_error = error
            future.get_loop().call_soon_threadsafe(self._resolve_score, future, scoring_error)
        evaluation_thread.join()
        typer.echo(
            "RWKV pool peak in-flight: "
            f"observed={self.model.pool.peak_inflight} "
            f"capacity={tuple(replica.max_concurrency for replica in self.model.pool.manifest.replicas)}"
        )
        typer.echo(
            "RWKV task-ready evidence: "
            "first_http_before_all_datasets="
            f"{self.model.pool.first_request_at is not None and (self._all_datasets_ready_at is None or self.model.pool.first_request_at < self._all_datasets_ready_at)} "
            f"datasets={self._datasets_loaded}/{len(self._task_names)}"
        )
        if scoring_error is not None:
            raise scoring_error
        if evaluation_errors:
            raise evaluation_errors[0]
        if self.is_main_process():
            self.evaluation_tracker.general_config_logger.log_end_time()
            self.evaluation_tracker.metrics_logger.aggregate(
                task_dict=self.tasks_dict,
                bootstrap_iters=self.pipeline_parameters.bootstrap_iters,
            )
            self.evaluation_tracker.details_logger.aggregate()
            for task_name, summary in self.evaluation_tracker.details_logger.compiled_details.items():
                self.evaluation_tracker.metrics_logger.metric_aggregated[task_name].update(
                    n_samples=summary.n_samples,
                    n_completions=summary.n_completions,
                    n_truncated=summary.n_truncated,
                    truncation_rate=summary.truncation_rate,
                )

    async def _evaluate_tasks(self, score_task) -> None:  # noqa: C901
        task_concurrency = self._task_concurrency()
        task_queue = asyncio.Queue()
        prepared_queue = asyncio.Queue(maxsize=min(self._DATASET_LOADERS, task_concurrency))
        for task_name in self._task_names:
            task_queue.put_nowait(task_name)

        async def load_datasets() -> None:
            while not task_queue.empty():
                try:
                    task_name = task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                task = self.tasks_dict[task_name]
                dataset = await asyncio.to_thread(task.download_dataset_worker, task)
                self._datasets_loaded += 1
                logger.info("RWKV dataset ready: task=%s", task_name)
                if self._datasets_loaded == len(self._task_names):
                    self._all_datasets_ready_at = time.monotonic()
                    typer.echo(f"RWKV datasets ready: {self._datasets_loaded}/{len(self._task_names)}")
                await prepared_queue.put((task_name, dataset))

        async def evaluate_tasks() -> None:
            while True:
                prepared = await prepared_queue.get()
                if prepared is None:
                    return
                task_name, dataset = prepared
                task = self.tasks_dict[task_name]
                task.dataset = dataset
                docs = self._prepare_task_documents(task)
                docs = _configure_task_evaluation_plan(self, task, docs)
                self._index_task_documents(task, docs)
                sampling_docs = defaultdict(list)
                for doc in docs:
                    for sampling_method in doc.sampling_methods:
                        sampling_docs[sampling_method].append(doc)
                rollouts = sum(doc.num_samples for doc in docs)
                logger.info(
                    "RWKV task model call started: task=%s documents=%d rollouts=%d", task_name, len(docs), rollouts
                )
                outputs = {SamplingMethod.GENERATIVE: await self.model.greedy_until(docs)}
                await score_task(task_name, sampling_docs, outputs)

        loaders = [
            asyncio.create_task(load_datasets()) for _ in range(min(self._DATASET_LOADERS, len(self._task_names)))
        ]
        consumers = [asyncio.create_task(evaluate_tasks()) for _ in range(task_concurrency)]

        async def finish_loading() -> None:
            await asyncio.gather(*loaders)
            for _ in consumers:
                await prepared_queue.put(None)

        coordinator = asyncio.create_task(finish_loading())
        running = [coordinator, *loaders, *consumers]
        try:
            await asyncio.gather(coordinator, *consumers)
            self.evaluation_tracker.task_config_logger.log(self.tasks_dict)
        except BaseException:
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            raise
        finally:
            await self.model.acleanup()

    def _task_concurrency(self) -> int:
        if self.pipeline_parameters.max_samples is None:
            return 1
        return min(
            len(self._task_names),
            math.ceil(self.model.pool.http_worker_limit / self.pipeline_parameters.max_samples),
        )

    @staticmethod
    async def _submit_score(scoring_queue, task_name, sampling_docs, outputs) -> None:
        future = asyncio.get_running_loop().create_future()
        scoring_queue.put((task_name, sampling_docs, outputs, future))
        await future

    @staticmethod
    def _resolve_score(future, error) -> None:
        if future.done():
            return
        if error is None:
            future.set_result(None)
        else:
            future.set_exception(error)

    def _score_task(self, task_name, sampling_docs, outputs) -> None:
        self.sampling_docs = sampling_docs
        self._post_process_outputs(outputs)
        self._compute_metrics(outputs)
        if self.task_callback is not None:
            self.task_callback(task_name, self.evaluation_tracker.details_logger.details[task_name])
