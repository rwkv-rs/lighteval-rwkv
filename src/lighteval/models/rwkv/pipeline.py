# MIT License

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import inspect
import json
import logging
import queue
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from typing import Mapping

import typer
from datasets import config as datasets_config

from lighteval.metrics import apply_metric
from lighteval.metrics.metrics_sample import SampleLevelComputation, SamplingMetric
from lighteval.models.rwkv.http_model import STREAMING_FLUSH_BATCH_SIZE
from lighteval.pipeline import Pipeline
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc, SamplingMethod


_TARGET_COMPLETIONS = 4096
_MIN_COMPLETIONS = 3000
_MAX_COMPLETIONS = 6000
logger = logging.getLogger(__name__)


def _dataset_cache_key(task) -> str:
    identity = json.dumps(
        {
            "data_files": task.data_files,
            "dataset_config_name": task.dataset_config_name,
            "dataset_path": task.dataset_path,
            "dataset_revision": task.dataset_revision,
        },
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode()).hexdigest()


@contextmanager
def _dataset_cache_lock(task):
    key = _dataset_cache_key(task)
    lock_dir = Path(datasets_config.HF_DATASETS_CACHE) / ".rwkv-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    with (lock_dir / key).open("a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        logger.info(
            "RWKV dataset cache lock acquired: task=%s key=%s wait_seconds=%.3f",
            task.full_name,
            key[:12],
            time.monotonic() - started_at,
        )
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _download_dataset(task):
    with _dataset_cache_lock(task):
        return task.download_dataset_worker(task)


class RWKVAvgAtK(SampleLevelComputation):
    """Average one native task scorer over exactly k independent completions."""

    def __init__(self, k: int, metric) -> None:
        self.k = k
        self.metric = metric

    def __str__(self) -> str:
        return f"RWKVAvgAtK(k={self.k})"

    def compute(self, doc: Doc, model_response, **kwargs):
        grouped_values = None
        if isinstance(self.metric.metric_name, (list, tuple)):
            names = tuple(self.metric.metric_name)
            grouped_values = {name: [] for name in names}
            scores = []
            extracted_answers = []
            for index in range(self.k):
                response = model_response[index]
                if response.finish_reasons == ["length"]:
                    native = dict.fromkeys(names, 0.0)
                else:
                    native = self.metric.compute_sample(doc=doc, model_response=response)
                for name in names:
                    grouped_values[name].append(float(native[name]))
                scores.append(float(native[names[0]]))
                extracted_answers.append(self.extract_rollout_answer(doc, response))
        else:
            scores = []
            extracted_answers = []
            for index in range(self.k):
                response = model_response[index]
                scores.append(self.score_rollout(doc, response))
                extracted_answers.append(self.extract_rollout_answer(doc, response))

        # DetailsLogger serializes Doc.specific with the native details artifact.
        # Keep the producer's per-rollout facts next to the document so downstream
        # publishers can report them without invoking a benchmark scorer again.
        specific = dict(doc.specific or {})
        # Keep the historical primary-metric fields while retaining all native
        # generative metrics when a task registers more than one scorer.
        specific.setdefault("rwkv_rollout_scores", scores)
        specific["rwkv_rollout_extracted_answers"] = extracted_answers
        specific["rwkv_model_answers"] = extracted_answers
        metric_names = (
            tuple(self.metric.metric_name)
            if isinstance(self.metric.metric_name, (list, tuple))
            else (self.metric.metric_name,)
        )
        if len(metric_names) > 1:
            score_by_metric = dict(specific.get("rwkv_rollout_scores_by_metric", {}))
            for metric_name in metric_names:
                score_by_metric[metric_name] = (
                    [grouped_values[metric_name][index] for index in range(self.k)]
                    if grouped_values is not None
                    else scores
                )
            specific["rwkv_rollout_scores_by_metric"] = score_by_metric
        if self.k == 1:
            specific["rwkv_model_answer"] = extracted_answers[0] if extracted_answers else ""
        doc.specific = specific

        if grouped_values is not None:
            return {name: sum(items) / self.k for name, items in grouped_values.items()}
        return sum(scores) / self.k

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
        if scorer_owner := getattr(getattr(scorer, "compute_score", None), "__self__", None):
            if extractor := getattr(scorer_owner, "extract_answer", None):
                return extractor(doc, model_response)
        # Native extractive scorers record their canonical prediction in the
        # document during scoring (for example, MultilingualExtractiveMatchMetric).
        # Reuse that producer-owned value instead of leaking the full reasoning
        # response into the details artifact.
        if extracted := (doc.specific or {}).get("extracted_predictions"):
            return str(extracted[0])
        return model_response.final_text[0] if model_response.final_text else ""


def _evaluation_plan(num_docs: int) -> tuple[int, int, str]:
    """Return evaluated documents, completions per document, and the only metric name."""
    if num_docs == 0:
        return 0, 1, "avg@1"
    # Pick the power-of-two rollout count whose total is in the agreed
    # 3000--6000 window and closest to the 4096 target.  This deliberately
    # allows the lower side of 4096 (e.g. AIME: 30*128=3840) so that we do
    # not jump to an 8192-sized workload.
    candidates = []
    k = 1
    while k * num_docs <= _MAX_COMPLETIONS:
        if k * num_docs >= _MIN_COMPLETIONS:
            candidates.append(k)
        k *= 2
    if candidates:
        k = min(candidates, key=lambda candidate: (abs(candidate * num_docs - _TARGET_COMPLETIONS), candidate))
    return num_docs, k, f"avg@{k}"


def _selector_priority(
    selector_rollouts: Mapping[str, int], configured_order: tuple[str, ...] = ()
) -> tuple[str, ...]:
    """Order benchmarks by their remaining rollout count."""
    order = {selector: index for index, selector in enumerate(configured_order)}
    return tuple(sorted(selector_rollouts, key=lambda selector: (selector_rollouts[selector], order.get(selector, 0))))


def _make_document_ids_unique(docs: list[Doc]) -> None:
    """Disambiguate repeated source IDs for the cache while retaining provenance."""
    occurrences = defaultdict(int)
    for doc in docs:
        source_id = str(doc.id)
        occurrence = occurrences[source_id]
        occurrences[source_id] += 1
        if occurrence:
            doc.specific = {**(doc.specific or {}), "rwkv_source_document_id": source_id}
            doc.id = f"{source_id}#{occurrence}"


def _configure_task_evaluation_plan(pipeline, task, docs) -> list[Doc]:
    """Apply RWKV rollouts without changing the task's native request types."""
    original_num_docs = len(task.eval_docs())
    max_samples = pipeline.pipeline_parameters.max_samples
    has_generative = any(metric.category == SamplingMethod.GENERATIVE for metric in task.metrics)
    if has_generative:
        if max_samples is None:
            effective_num_docs, k, _ = _evaluation_plan(original_num_docs)
        else:
            task_max_samples = getattr(pipeline, "_task_max_samples", {}).get(task.full_name, max_samples)
            effective_num_docs, k = min(original_num_docs, task_max_samples), 1
    else:
        effective_num_docs, k = (
            min(original_num_docs, max_samples) if max_samples is not None else original_num_docs,
            1,
        )

    docs = docs[:effective_num_docs]
    _make_document_ids_unique(docs)
    wrapped_metrics = []
    for metric in task.metrics:
        if metric.category != SamplingMethod.GENERATIVE:
            wrapped_metrics.append(metric)
            continue
        wrapped = copy(metric)
        wrapped.sample_level_fn = RWKVAvgAtK(k, metric)
        wrapped_metrics.append(wrapped)
    task.metrics = tuple(wrapped_metrics)
    task.num_samples = list(dict.fromkeys([*getattr(task, "num_samples", [1]), *([k] if has_generative else [])]))
    for doc in docs:
        # LOGPROBS/PERPLEXITY requests are single requests.  Only documents
        # participating in a generative metric receive independent rollouts.
        doc.num_samples = k if SamplingMethod.GENERATIVE in doc.sampling_methods else 1
    task.config = copy(task.config)
    task.config.metrics = task.metrics
    task.config.original_num_docs = original_num_docs
    task.config.effective_num_docs = len(docs)
    task.sampling_methods = list(dict.fromkeys(metric.category for metric in task.metrics))
    return docs


class RWKVPipeline(Pipeline):
    """Run native LightEval selectors in shortest-remaining-rollout order."""

    _DATASET_LOADERS = 8

    def __init__(self, *args, selector_tasks: Mapping[str, tuple[str, ...]], task_max_samples=None, **kwargs) -> None:
        self._configured_selector_tasks = selector_tasks
        self._configured_task_max_samples = task_max_samples
        self._skip_selectors: frozenset[str] = frozenset()
        super().__init__(*args, **kwargs)

    def set_skip_selectors(self, selectors) -> None:
        """Exclude selectors already finalized by an external result publisher."""
        self._skip_selectors = frozenset(selectors)

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
        # Streaming scoring path: counts docs written into DetailsLogger's pending batch per task,
        # since each doc is scored as soon as its k rollouts arrive rather than once per task.
        self._pending_flush_counts: dict[str, int] = defaultdict(int)
        if self._metric_options:
            self._update_num_samples(list(self.tasks_dict.values()))

    def _prepare_task_documents(self, task):
        # Task formatters and request categories belong to LightEval.  In
        # particular, choices must remain available to LOGPROBS requests.
        max_samples = self._task_max_samples.get(task.full_name, self.pipeline_parameters.max_samples)
        return task.get_docs(max_samples)

    def _post_process_outputs(self, sampling_method_responses) -> None:
        # Use LightEval's normal response processing.  RWKV-specific answer
        # extraction is performed only by the native generative scorer wrapped
        # by RWKVAvgAtK; log-probability responses never pass through it.
        if SamplingMethod.GENERATIVE in sampling_method_responses:
            super()._post_process_outputs(
                {SamplingMethod.GENERATIVE: sampling_method_responses[SamplingMethod.GENERATIVE]}
            )

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
                asyncio.run(self._evaluate_tasks(scoring_queue))
            except BaseException as error:
                evaluation_errors.append(error)
            finally:
                scoring_queue.put(None)

        evaluation_thread = threading.Thread(target=run_evaluation, name="rwkv-http-event-loop")
        evaluation_thread.start()
        scoring_error = None
        while (scoring := scoring_queue.get()) is not None:
            kind, task_name, doc, response, future = scoring
            if scoring_error is None:
                try:
                    if kind == "doc":
                        self._score_doc(task_name, doc, response)
                    else:
                        self._finalize_task(task_name)
                except BaseException as error:
                    scoring_error = error
            self._resolve_score_threadsafe(future, scoring_error)
        evaluation_thread.join()
        typer.echo(
            "RWKV pool peak in-flight: "
            f"observed={self.model.pool.peak_inflight} "
            f"capacity={tuple(replica.max_concurrency for replica in self.model.pool.manifest.replicas)}"
        )
        if scoring_error is not None:
            raise scoring_error
        if evaluation_errors:
            raise evaluation_errors[0]
        if self.is_main_process():
            self._finalize_metrics()

    async def _evaluate_tasks(self, scoring_queue) -> None:  # noqa: C901
        selector_tasks = {
            selector: task_names
            for selector, task_names in self._selector_tasks.items()
            if selector not in getattr(self, "_skip_selectors", frozenset())
        }
        task_names = tuple(task_name for names in selector_tasks.values() for task_name in names)
        skip_selectors = getattr(self, "_skip_selectors", frozenset())
        if skip_selectors:
            logger.info("RWKV selectors skipped before dataset preparation: %s", ", ".join(sorted(skip_selectors)))
        if not task_names:
            self.evaluation_tracker.task_config_logger.log(self.tasks_dict)
            await self.model.acleanup()
            return

        load_semaphore = asyncio.Semaphore(min(self._DATASET_LOADERS, len(task_names)))
        scoring_tasks: set[asyncio.Task] = set()
        scoring_failures: list[BaseException] = []

        async def prepare_task(task_name):
            async with load_semaphore:
                task = self.tasks_dict[task_name]
                dataset = await asyncio.to_thread(_download_dataset, task)
                self._datasets_loaded += 1
                logger.info("RWKV dataset ready: task=%s", task_name)
                if self._datasets_loaded == len(task_names):
                    typer.echo(f"RWKV datasets ready: {self._datasets_loaded}/{len(task_names)}")
            task.dataset = dataset
            docs = self._prepare_task_documents(task)
            docs = _configure_task_evaluation_plan(self, task, docs)
            self._index_task_documents(task, docs)
            return task_name, docs, self.model.pending_rollouts(docs)

        async def evaluate_task(task_name, docs, pending_rollouts) -> None:  # noqa: C901
            logger.info(
                "RWKV task model call started: task=%s documents=%d pending_rollouts=%d",
                task_name,
                len(docs),
                pending_rollouts,
            )

            async def on_document_ready(doc, response, sampling_method) -> None:
                # ModelResponse is deliberately tagged outside the serialized
                # response schema.  This keeps the queue compatible with the
                # native details format while allowing mixed tasks to share a
                # document and be scored by their actual request category.
                response._rwkv_sampling_method = sampling_method
                await self._submit_doc(scoring_queue, task_name, doc, response)

            async def run_model_call(sampling_method, method_name, method_docs) -> None:
                if not method_docs:
                    return

                async def ready(doc, response):
                    await on_document_ready(doc, response, sampling_method)

                method = getattr(self.model, method_name)
                if inspect.iscoroutinefunction(method):
                    await method(method_docs, on_document_ready=ready)
                else:
                    # Native synchronous adapters are allowed in the same
                    # pipeline; run their blocking call off the HTTP loop.
                    responses = await asyncio.to_thread(method, method_docs)
                    for doc, response in zip(method_docs, responses, strict=True):
                        await ready(doc, response)

            async def run_task() -> None:
                for sampling_method, method_name in (
                    (SamplingMethod.GENERATIVE, "greedy_until"),
                    (SamplingMethod.LOGPROBS, "loglikelihood"),
                    (SamplingMethod.PERPLEXITY, "loglikelihood_rolling"),
                ):
                    method_docs = [doc for doc in docs if sampling_method in doc.sampling_methods]
                    await run_model_call(sampling_method, method_name, method_docs)
                await self._submit_task_done(scoring_queue, task_name)

            scoring_task = asyncio.create_task(run_task())
            scoring_tasks.add(scoring_task)
            running.append(scoring_task)

            def on_scoring_done(completed: asyncio.Task) -> None:
                scoring_tasks.discard(completed)
                if completed.cancelled():
                    return
                error = completed.exception()
                if error is None:
                    return
                scoring_failures.append(error)
                for running_task in running:
                    if running_task is not completed and not running_task.done():
                        running_task.cancel()

            scoring_task.add_done_callback(on_scoring_done)
            # Keep the selector active until its generation and scoring task finishes.
            await scoring_task

        selector_order = sorted(selector_tasks, key=lambda selector: len(selector_tasks[selector]))
        preparation_tasks = {}
        running = []
        next_selector_index = 0

        def enqueue_next_selector() -> None:
            nonlocal next_selector_index
            if next_selector_index >= len(selector_order):
                return
            selector = selector_order[next_selector_index]
            next_selector_index += 1
            for task_name in selector_tasks[selector]:
                task = asyncio.create_task(prepare_task(task_name))
                preparation_tasks[task] = selector
                running.append(task)

        for _ in range(min(self._DATASET_LOADERS, len(selector_order))):
            enqueue_next_selector()
        try:
            prepared_by_selector = defaultdict(list)
            selector_rollouts = {}
            ready_selectors = {}
            active_selectors: dict[asyncio.Task, str] = {}
            evaluation_started = False

            async def evaluate_selector(selector, queued: asyncio.Event) -> None:
                task_calls = []
                for task_name, docs, pending_rollouts in sorted(
                    prepared_by_selector[selector], key=lambda item: item[2]
                ):
                    task_call = asyncio.create_task(evaluate_task(task_name, docs, pending_rollouts))
                    task_calls.append(task_call)
                    running.append(task_call)
                    await asyncio.sleep(0)
                queued.set()
                await asyncio.gather(*task_calls)

            async def admit_selector(selector) -> None:
                nonlocal evaluation_started
                if not evaluation_started:
                    typer.echo(f"RWKV evaluation started: selector={selector}")
                    evaluation_started = True
                logger.info(
                    "RWKV selector admitted: selector=%s pending_rollouts=%d",
                    selector,
                    selector_rollouts[selector],
                )
                queued = asyncio.Event()
                selector_task = asyncio.create_task(evaluate_selector(selector, queued))
                active_selectors[selector_task] = selector
                running.append(selector_task)
                await queued.wait()

            while preparation_tasks or ready_selectors or active_selectors:
                while ready_selectors:
                    selector = _selector_priority(
                        {value: selector_rollouts[value] for value in ready_selectors}, tuple(selector_order)
                    )[0]
                    positive_active = sum(
                        selector_rollouts[active_selector] > 0 for active_selector in active_selectors.values()
                    )
                    if selector_rollouts[selector] > 0 and positive_active >= 1:
                        break
                    del ready_selectors[selector]
                    await admit_selector(selector)

                done, _ = await asyncio.wait(
                    (*preparation_tasks, *active_selectors),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                preparation_order = {
                    task: selector_order.index(selector) for task, selector in preparation_tasks.items()
                }
                for task in sorted(done, key=lambda completed: preparation_order.get(completed, len(selector_order))):
                    if task in active_selectors:
                        del active_selectors[task]
                        await task
                        continue
                    selector = preparation_tasks.pop(task)
                    prepared_by_selector[selector].append(await task)
                    if len(prepared_by_selector[selector]) == len(selector_tasks[selector]):
                        selector_rollouts[selector] = sum(item[2] for item in prepared_by_selector[selector])
                        ready_selectors[selector] = None
                        enqueue_next_selector()
                        logger.info(
                            "RWKV selector ready: selector=%s pending_rollouts=%d",
                            selector,
                            selector_rollouts[selector],
                        )
            if scoring_tasks:
                await asyncio.gather(*tuple(scoring_tasks))
            if scoring_failures:
                raise scoring_failures[0]
            self.evaluation_tracker.task_config_logger.log(self.tasks_dict)
        except BaseException:
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            if scoring_failures:
                raise scoring_failures[0]
            raise
        finally:
            self.evaluation_tracker.abort_task_writers()
            await self.model.acleanup()

    @staticmethod
    async def _submit_doc(scoring_queue, task_name, doc, response) -> None:
        future = asyncio.get_running_loop().create_future()
        scoring_queue.put(("doc", task_name, doc, response, future))
        await future

    @staticmethod
    async def _submit_task_done(scoring_queue, task_name) -> None:
        future = asyncio.get_running_loop().create_future()
        scoring_queue.put(("task_done", task_name, None, None, future))
        await future

    @staticmethod
    def _resolve_score(future, error) -> None:
        if future.done():
            return
        if error is None:
            future.set_result(None)
        else:
            future.set_exception(error)

    @classmethod
    def _resolve_score_threadsafe(cls, future, error) -> None:
        loop = future.get_loop()
        if not loop.is_closed():
            loop.call_soon_threadsafe(cls._resolve_score, future, error)

    def _score_doc(self, task_name: str, doc: Doc, response) -> None:
        sampling_method = getattr(response, "_rwkv_sampling_method", None)
        if sampling_method is None:
            categories = [
                method
                for method in (SamplingMethod.GENERATIVE, SamplingMethod.LOGPROBS, SamplingMethod.PERPLEXITY)
                if method in doc.sampling_methods
            ]
            if len(categories) != 1:
                raise ValueError(f"cannot infer request category for mixed task document {doc.id}")
            sampling_method = categories[0]
        if sampling_method != SamplingMethod.GENERATIVE:
            response.text_post_processed = None
        self.sampling_docs = {sampling_method: [doc]}
        if sampling_method == SamplingMethod.GENERATIVE:
            self._post_process_outputs({sampling_method: [response]})
        task = self.tasks_dict[task_name]
        metric_category_metrics = [metric for metric in task.metrics if metric.category == sampling_method]
        if not metric_category_metrics:
            raise ValueError(f"task {task_name} has no metrics for {sampling_method.name}")
        outputs = apply_metric(docs=[doc], responses=[response], metrics=metric_category_metrics)
        output = outputs[0]
        self.evaluation_tracker.metrics_logger.log(task_name, output)
        self.evaluation_tracker.details_logger.log_streaming(task_name, doc, response, output)
        self._pending_flush_counts[task_name] += 1
        if self._pending_flush_counts[task_name] >= STREAMING_FLUSH_BATCH_SIZE:
            self.evaluation_tracker.flush_task_docs(task_name)
            self._pending_flush_counts[task_name] = 0

    def _finalize_task(self, task_name: str) -> None:
        self.evaluation_tracker.flush_task_docs(task_name)
        self._pending_flush_counts.pop(task_name, None)
        self.evaluation_tracker.close_task_writer(task_name)
        self.evaluation_tracker.details_logger.finalize_task(task_name)
        if self.task_callback is not None:
            self.task_callback(task_name)
