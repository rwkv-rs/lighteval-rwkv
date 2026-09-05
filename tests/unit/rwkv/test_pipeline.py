import asyncio
import multiprocessing
import threading
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import lighteval.main_rwkv as main_rwkv
import lighteval.models.rwkv.pipeline as rwkv_pipeline
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import AvgAtN, ExactMatches, MajAtN, MathVerifyMatch
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks.ifbench.instructions import EmojiSentenceChecker, NGramOverlapChecker


def _streaming_pipeline(task_names, model, download, *, max_samples):
    if not hasattr(model, "pending_rollouts"):
        model.pending_rollouts = lambda docs: sum(doc.num_samples for doc in docs)
    pipeline = rwkv_pipeline.RWKVPipeline.__new__(rwkv_pipeline.RWKVPipeline)
    pipeline._task_names = tuple(task_names)
    pipeline._selector_tasks = {task_name: (task_name,) for task_name in task_names}
    pipeline._task_selectors = {task_name: task_name for task_name in task_names}
    pipeline.tasks_dict = {
        task_name: SimpleNamespace(
            full_name=task_name,
            dataset_path="dataset",
            dataset_config_name=None,
            dataset_revision=None,
            data_files={"validation": task_name},
            download_dataset_worker=lambda task: download(task.full_name),
        )
        for task_name in task_names
    }
    pipeline.documents_dict = {}
    pipeline.sampling_docs = defaultdict(list)
    pipeline._datasets_loaded = 0
    pipeline.pipeline_parameters = SimpleNamespace(max_samples=max_samples)
    pipeline.model = model
    pipeline.evaluation_tracker = SimpleNamespace(
        task_config_logger=SimpleNamespace(log=lambda _tasks: None),
    )
    pipeline._prepare_task_documents = lambda task: [
        SimpleNamespace(
            task_name=task.full_name,
            num_samples=1,
            sampling_methods=[SamplingMethod.GENERATIVE],
        )
    ]
    return pipeline


def _hold_dataset_cache_lock(cache_dir, data_file, acquired, release, fail=False):
    rwkv_pipeline.datasets_config.HF_DATASETS_CACHE = Path(cache_dir)
    task = SimpleNamespace(
        full_name=data_file,
        dataset_path="dataset",
        dataset_config_name=None,
        dataset_revision=None,
        data_files={"validation": data_file},
    )
    try:
        with rwkv_pipeline._dataset_cache_lock(task):
            acquired.set()
            if fail:
                raise RuntimeError("dataset load failed")
            release.wait(5)
    except RuntimeError:
        pass


@pytest.mark.parametrize(("second_data_file", "blocked"), [("same", True), ("other", False)])
def test_dataset_cache_lock_is_keyed_across_processes(tmp_path, second_data_file, blocked):
    context = multiprocessing.get_context("fork")
    first_acquired = context.Event()
    second_acquired = context.Event()
    release = context.Event()
    first = context.Process(target=_hold_dataset_cache_lock, args=(tmp_path, "same", first_acquired, release))
    second = context.Process(
        target=_hold_dataset_cache_lock,
        args=(tmp_path, second_data_file, second_acquired, release),
    )
    first.start()
    assert first_acquired.wait(2)
    second.start()

    assert second_acquired.wait(0.2) is not blocked
    release.set()
    assert second_acquired.wait(2)
    first.join(2)
    second.join(2)
    assert (first.exitcode, second.exitcode) == (0, 0)


def test_dataset_cache_lock_is_released_after_failure(tmp_path):
    context = multiprocessing.get_context("fork")
    failed_acquired = context.Event()
    acquired = context.Event()
    release = context.Event()
    failed = context.Process(
        target=_hold_dataset_cache_lock,
        args=(tmp_path, "same", failed_acquired, release, True),
    )
    retry = context.Process(target=_hold_dataset_cache_lock, args=(tmp_path, "same", acquired, release))

    failed.start()
    assert failed_acquired.wait(2)
    failed.join(2)
    assert failed.exitcode == 0
    retry.start()
    assert acquired.wait(2)
    release.set()
    retry.join(2)
    assert retry.exitcode == 0


async def _evaluate_streaming_pipeline(pipeline):
    async def score(task_name, sampling_docs, outputs):
        pipeline._score_task(task_name, sampling_docs, outputs)

    await pipeline._evaluate_tasks(score)


def test_rwkv_pipeline_starts_ready_selector_before_all_datasets_finish(monkeypatch):
    slow_release = threading.Event()
    slow_finished = threading.Event()
    calls = []

    def download(task_name):
        if task_name == "slow|0":
            slow_release.wait(2)
            slow_finished.set()
        return task_name

    class Model:
        pool = SimpleNamespace(http_worker_limit=20)

        def pending_rollouts(self, docs):
            return sum(doc.num_samples for doc in docs)

        async def greedy_until(self, docs):
            calls.append(docs[0].task_name)
            return []

        async def acleanup(self):
            pass

    pipeline = _streaming_pipeline(("fast|0", "slow|0"), Model(), download, max_samples=10)
    pipeline._selector_tasks = {"small": ("slow|0",), "large": ("fast|0",)}
    pipeline._task_selectors = {"slow|0": "small", "fast|0": "large"}
    pipeline._prepare_task_documents = lambda task: [
        SimpleNamespace(
            task_name=task.full_name,
            num_samples=1 if task.full_name == "slow|0" else 30,
            sampling_methods=[SamplingMethod.GENERATIVE],
        )
    ]
    pipeline._score_task = lambda *_args: None
    monkeypatch.setattr(rwkv_pipeline, "_configure_task_evaluation_plan", lambda _pipeline, _task, docs: docs)

    async def run():
        evaluation = asyncio.create_task(_evaluate_streaming_pipeline(pipeline))
        try:
            for _ in range(20):
                if calls:
                    break
                await asyncio.sleep(0.01)
            assert calls == ["fast|0"]
            assert not slow_finished.is_set()
        finally:
            slow_release.set()
        await evaluation
        assert calls == ["fast|0", "slow|0"]

    asyncio.run(run())


def test_cached_selector_does_not_consume_a_rollout_slot(monkeypatch):
    release = asyncio.Event()
    calls = []

    class Model:
        pool = SimpleNamespace(http_worker_limit=20)

        def pending_rollouts(self, docs):
            return {"cached|0": 0, "small|0": 10, "spare|0": 20}[docs[0].task_name]

        async def greedy_until(self, docs):
            calls.append(docs[0].task_name)
            await release.wait()
            return []

        async def acleanup(self):
            pass

    pipeline = _streaming_pipeline(
        ("cached|0", "small|0", "spare|0"),
        Model(),
        lambda task_name: task_name,
        max_samples=10,
    )
    pipeline._score_task = lambda *_args: None
    monkeypatch.setattr(rwkv_pipeline, "_configure_task_evaluation_plan", lambda _pipeline, _task, docs: docs)

    async def run():
        evaluation = asyncio.create_task(_evaluate_streaming_pipeline(pipeline))
        for _ in range(20):
            if len(calls) == 3:
                break
            await asyncio.sleep(0.01)
        assert calls == ["cached|0", "small|0", "spare|0"]
        release.set()
        await evaluation

    asyncio.run(run())


def test_cached_selector_bypasses_full_rollout_slots(monkeypatch):
    cached_dataset_ready = threading.Event()
    release = asyncio.Event()
    rollout_slots_full = asyncio.Event()
    cached_started = asyncio.Event()
    calls = []

    def download(task_name):
        if task_name == "cached|0":
            cached_dataset_ready.wait(2)
        return task_name

    class Model:
        pool = SimpleNamespace(http_worker_limit=20)

        def pending_rollouts(self, docs):
            return 0 if docs[0].task_name == "cached|0" else 10

        async def greedy_until(self, docs):
            calls.append(docs[0].task_name)
            if len(calls) == 2:
                rollout_slots_full.set()
            if docs[0].task_name == "cached|0":
                cached_started.set()
            await release.wait()
            return []

        async def acleanup(self):
            pass

    pipeline = _streaming_pipeline(("cached|0", "first|0", "second|0"), Model(), download, max_samples=10)
    pipeline._score_task = lambda *_args: None
    monkeypatch.setattr(rwkv_pipeline, "_configure_task_evaluation_plan", lambda _pipeline, _task, docs: docs)

    async def run():
        evaluation = asyncio.create_task(_evaluate_streaming_pipeline(pipeline))
        await asyncio.wait_for(rollout_slots_full.wait(), 1)
        assert calls == ["first|0", "second|0"]
        cached_dataset_ready.set()
        await asyncio.wait_for(cached_started.wait(), 1)
        assert calls == ["first|0", "second|0", "cached|0"]
        release.set()
        await evaluation

    asyncio.run(run())


def test_selector_priority_uses_shortest_remaining_benchmark_first():
    assert rwkv_pipeline._selector_priority({"large": 9, "small": 5, "spare": 7}) == (
        "small",
        "spare",
        "large",
    )


def test_ifbench_checkers_treat_empty_responses_as_failed():
    overlap = NGramOverlapChecker("test")
    overlap.build_description(reference_text="reference text", percentage=50)
    assert overlap.check_following("") is False

    emoji = EmojiSentenceChecker("test")
    assert emoji.check_following("!!!") is False


def test_duplicate_source_document_ids_are_disambiguated_for_cache():
    docs = [
        Doc(query="first", choices=["answer"], gold_index=0, id="939"),
        Doc(query="second", choices=["answer"], gold_index=0, id="939", specific={"split": "test"}),
    ]

    rwkv_pipeline._make_document_ids_unique(docs)

    assert [doc.id for doc in docs] == ["939", "939#1"]
    assert docs[0].specific is None
    assert docs[1].specific == {"split": "test", "rwkv_source_document_id": "939"}


def test_rwkv_pipeline_runs_scorer_on_process_main_thread():
    pipeline = rwkv_pipeline.RWKVPipeline.__new__(rwkv_pipeline.RWKVPipeline)
    pipeline.pipeline_parameters = SimpleNamespace(num_fewshot_seeds=1, max_samples=10, job_id=0)
    pipeline.model = SimpleNamespace(
        pool=SimpleNamespace(
            peak_inflight=(2,),
            first_request_at=1.0,
            manifest=SimpleNamespace(replicas=(SimpleNamespace(max_concurrency=4),)),
        )
    )
    pipeline._datasets_loaded = 1
    pipeline._task_names = ("task|0",)
    pipeline.evaluation_tracker = SimpleNamespace(
        general_config_logger=SimpleNamespace(log_args_info=lambda **_kwargs: None),
    )
    pipeline.is_main_process = lambda: False
    score_threads = []

    async def evaluate_tasks(score):
        await score("task|0", {}, {})

    pipeline._evaluate_tasks = evaluate_tasks
    pipeline._score_task = lambda *_args: score_threads.append(threading.current_thread())

    pipeline.evaluate()

    assert score_threads == [threading.main_thread()]


def test_rwkv_pipeline_scores_only_after_every_rollout_finishes(monkeypatch):
    pending_rollout = asyncio.Event()
    first_rollouts_done = asyncio.Event()
    scored = []

    class Model:
        pool = SimpleNamespace(http_worker_limit=4)

        async def greedy_until(self, _docs):
            first_rollouts_done.set()
            await pending_rollout.wait()
            return []

        async def acleanup(self):
            pass

    pipeline = _streaming_pipeline(("task|0",), Model(), lambda task_name: task_name, max_samples=None)
    pipeline._prepare_task_documents = lambda task: [
        SimpleNamespace(
            task_name=task.full_name,
            num_samples=3,
            sampling_methods=[SamplingMethod.GENERATIVE],
        )
    ]
    pipeline._score_task = lambda *_args: scored.append("task|0")
    monkeypatch.setattr(rwkv_pipeline, "_configure_task_evaluation_plan", lambda _pipeline, _task, docs: docs)

    async def run():
        evaluation = asyncio.create_task(_evaluate_streaming_pipeline(pipeline))
        await first_rollouts_done.wait()
        assert scored == []
        pending_rollout.set()
        await evaluation

    asyncio.run(run())


def test_rwkv_pipeline_scoring_failure_cancels_other_tasks(monkeypatch):
    second_started = asyncio.Event()
    cancelled = []

    class Model:
        pool = SimpleNamespace(http_worker_limit=2)

        async def greedy_until(self, docs):
            if docs[0].task_name == "failed|0":
                await second_started.wait()
                return []
            second_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.append(docs[0].task_name)
                raise

        async def acleanup(self):
            cancelled.append("closed")

    pipeline = _streaming_pipeline(
        ("failed|0", "pending|0"),
        Model(),
        lambda task_name: task_name,
        max_samples=1,
    )
    pipeline._score_task = lambda task_name, *_args: (_ for _ in ()).throw(ValueError(task_name))
    monkeypatch.setattr(rwkv_pipeline, "_configure_task_evaluation_plan", lambda _pipeline, _task, docs: docs)

    with pytest.raises(ValueError, match=r"failed\|0"):
        asyncio.run(_evaluate_streaming_pipeline(pipeline))

    assert cancelled == ["pending|0", "closed"]


@pytest.mark.parametrize("generation_size", [1, 5, 256, 1280, 2048])
def test_open_think_uses_full_generation_contract(generation_size):
    doc = Doc(
        query="question",
        choices=["answer"],
        gold_index=0,
        generation_size=generation_size,
        stop_sequences=["\n"],
    )
    task = SimpleNamespace(
        config=SimpleNamespace(generation_size=generation_size, stop_sequence=["\n"]),
    )

    rwkv_pipeline.RWKVPipeline._prepare_open_think_task(task, [doc])

    assert doc.generation_size == 8192
    assert doc.stop_sequences == []
    assert task.config.generation_size == 8192
    assert task.config.stop_sequence == []


def test_truthfulqa_conversion_keeps_only_mc1():
    doc = Doc(
        query="question",
        choices=["true", "false", "true", "also true", "false", "also false"],
        gold_index=[0, 2, 3],
        specific={"len_mc1": 2},
        sampling_methods=[SamplingMethod.LOGPROBS],
    )
    metric = SimpleNamespace(
        metric_name=["truthfulqa_mc1", "truthfulqa_mc2"],
        category=SamplingMethod.LOGPROBS,
        corpus_level_fn={"truthfulqa_mc1": sum, "truthfulqa_mc2": sum},
        higher_is_better={"truthfulqa_mc1": True, "truthfulqa_mc2": True},
    )
    task = SimpleNamespace(
        full_name="truthfulqa:mc|0",
        metrics=(metric,),
        config=SimpleNamespace(metrics=(metric,), original_num_docs=-1, effective_num_docs=-1),
    )

    rwkv_pipeline.RWKVPipeline._prepare_truthfulqa_mc1(task, [doc])
    rwkv_pipeline.RWKVPipeline._prepare_choice_task(task, [doc])

    assert doc.choices == ["true", "false"]
    assert doc.gold_index == 0
    assert doc.specific["rwkv_truthfulqa_metric"] == "mc1"
    assert doc.specific["rwkv_choice"] is True
    assert [converted.metric_name for converted in task.metrics] == ["truthfulqa_mc1"]


def test_open_think_postprocessing_keeps_only_final_answer():
    pipeline = rwkv_pipeline.RWKVPipeline.__new__(rwkv_pipeline.RWKVPipeline)
    pipeline.model = SimpleNamespace(config=SimpleNamespace(cot_mode="open_think"))
    pipeline.sampling_docs = defaultdict(list)
    response = ModelResponse(text=[">reasoning</think>final", "answer without tags"])

    pipeline._post_process_outputs({SamplingMethod.GENERATIVE: [response]})

    assert response.final_text == ["final", "answer without tags"]


def test_open_think_postprocessing_ignores_duplicate_closing_tag():
    pipeline = rwkv_pipeline.RWKVPipeline.__new__(rwkv_pipeline.RWKVPipeline)
    pipeline.model = SimpleNamespace(config=SimpleNamespace(cot_mode="open_think"))
    pipeline.sampling_docs = defaultdict(list)
    response = ModelResponse(text=[">reasoning</think>final</think>"])

    pipeline._post_process_outputs({SamplingMethod.GENERATIVE: [response]})

    assert response.final_text == ["final"]


def test_rwkv_pipeline_always_converts_choices_after_task_prompt_override():
    doc = Doc(
        query="Upstream instruction: Question?",
        instruction="Upstream instruction: ",
        choices=["one", "two"],
        gold_index=1,
        sampling_methods=[SamplingMethod.LOGPROBS],
    )
    metric = Metrics.loglikelihood_acc.value
    task = SimpleNamespace(
        full_name="fixture|0",
        get_docs=lambda _max_samples: [doc],
        metrics=(metric,),
        config=SimpleNamespace(metrics=(metric,)),
    )
    pipeline = rwkv_pipeline.RWKVPipeline.__new__(rwkv_pipeline.RWKVPipeline)
    pipeline._task_max_samples = {}
    pipeline.pipeline_parameters = SimpleNamespace(
        max_samples=None,
        task_prompt="Evaluation: ",
        task_prompt_mode="replace",
    )
    pipeline.model = SimpleNamespace(config=SimpleNamespace(cot_mode="fake_think"))

    assert pipeline._prepare_task_documents(task) == [doc]
    assert doc.instruction == "Evaluation: "
    assert doc.query.startswith("Question?")
    assert "A. one" in doc.query
    assert doc.sampling_methods == [SamplingMethod.GENERATIVE]
    assert doc.specific["rwkv_choice"] is True


def test_fake_think_postprocessing_extracts_converted_choice_answer():
    doc = Doc(
        query="Question?\nA. one\nB. two",
        choices=["one", "two"],
        gold_index=1,
        sampling_methods=[SamplingMethod.GENERATIVE],
        specific={"rwkv_choice": True},
    )
    pipeline = rwkv_pipeline.RWKVPipeline.__new__(rwkv_pipeline.RWKVPipeline)
    pipeline.model = SimpleNamespace(config=SimpleNamespace(cot_mode="fake_think"))
    pipeline.pipeline_parameters = SimpleNamespace(
        remove_reasoning_tags=True,
        reasoning_tags=[("<think>", "</think>")],
    )
    pipeline.sampling_docs = {SamplingMethod.GENERATIVE: [doc]}
    response = ModelResponse(text=["<think>x</think>Answer: B"], finish_reasons=["stop"])

    pipeline._post_process_outputs({SamplingMethod.GENERATIVE: [response]})

    assert response.final_text == ["two"]


def test_rwkv_pipeline_discards_truncated_choice_answer():
    doc = Doc(
        query="Question?\nA. one\nB. two",
        choices=["one", "two"],
        gold_index=1,
        sampling_methods=[SamplingMethod.GENERATIVE],
        specific={"rwkv_choice": True},
    )
    pipeline = rwkv_pipeline.RWKVPipeline.__new__(rwkv_pipeline.RWKVPipeline)
    pipeline.model = SimpleNamespace(config=SimpleNamespace(cot_mode="open_think"))
    pipeline.sampling_docs = {SamplingMethod.GENERATIVE: [doc]}
    response = ModelResponse(text=["<think>Answer: B"], finish_reasons=["length"])

    pipeline._post_process_outputs({SamplingMethod.GENERATIVE: [response]})

    assert response.final_text == [""]


@pytest.mark.parametrize(
    ("num_docs", "effective_docs", "k", "metric_name"),
    [
        (30, 30, 256, "avg@256"),
        (500, 500, 16, "avg@16"),
        (1251, 1251, 4, "avg@4"),
        (5000, 5000, 2, "avg@2"),
        (5001, 5001, 1, "avg@1"),
        (50_001, 10_000, 1, "avg@0.2"),
    ],
)
def test_evaluation_plan_uses_power_of_two_k_or_twenty_percent(num_docs, effective_docs, k, metric_name):
    assert rwkv_pipeline._evaluation_plan(num_docs) == (effective_docs, k, metric_name)
    if num_docs <= 50_000:
        assert k & (k - 1) == 0
        assert k * num_docs > 5000
        assert k == 1 or (k // 2) * num_docs <= 5000


def test_partial_budget_is_ten_per_selector_not_ten_per_leaf():
    leaves = tuple(f"mmlu:subject_{index}" for index in range(57))
    resolved = main_rwkv.ResolvedBenchmarks(
        selector_count=2,
        leaf_tasks=(*leaves, "gsm8k"),
        selector_tasks=(("mmlu", leaves), ("gsm8k", ("gsm8k",))),
    )

    selector_tasks, budgets = main_rwkv._selector_sample_budgets(resolved, 10)

    assert selector_tasks == {"mmlu": leaves, "gsm8k": ("gsm8k",)}
    assert budgets is not None
    assert sum(budgets[leaf] for leaf in leaves if leaf in budgets) == 10
    assert [index for index, leaf in enumerate(leaves) if leaf in budgets] == [0, 5, 11, 17, 22, 28, 34, 39, 45, 51]
    assert budgets["gsm8k"] == 10


def test_partial_pipeline_drops_leaf_tasks_without_selector_budget(monkeypatch):
    tasks = {"mmlu:a|0": object(), "mmlu:b|0": object(), "gsm8k|0": object()}

    class Registry:
        def __init__(self, **_kwargs):
            pass

        def load_tasks(self):
            return dict(tasks)

    monkeypatch.setattr(rwkv_pipeline, "Registry", Registry)
    pipeline = rwkv_pipeline.RWKVPipeline.__new__(rwkv_pipeline.RWKVPipeline)
    pipeline._configured_selector_tasks = {"mmlu": ("mmlu:a", "mmlu:b"), "gsm8k": ("gsm8k",)}
    pipeline._configured_task_max_samples = {"mmlu:b": 1, "gsm8k": 10}
    pipeline.pipeline_parameters = SimpleNamespace(
        load_tasks_multilingual=True,
        custom_tasks_directory=None,
    )
    pipeline._metric_options = None

    pipeline._init_tasks_and_requests("mmlu,gsm8k")

    assert tuple(pipeline.tasks_dict) == ("mmlu:b|0", "gsm8k|0")
    assert pipeline._task_names == ("mmlu:b|0", "gsm8k|0")
    assert pipeline._selector_tasks == {"mmlu": ("mmlu:b|0",), "gsm8k": ("gsm8k|0",)}
    assert pipeline._task_max_samples == {"mmlu:b|0": 1, "gsm8k|0": 10}


def test_rwkv_avg_at_k_averages_the_native_task_scorer():
    metric = SampleLevelMetric(
        metric_name="avg@n:n=64",
        sample_level_fn=AvgAtN(n=64, sample_scoring_function=ExactMatches(strip_strings=True)),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=lambda values: sum(values) / len(values),
        higher_is_better=True,
    )
    scorer = rwkv_pipeline.RWKVAvgAtK(4, metric)
    doc = Doc(query="question", choices=["one", "two"], gold_index=0)
    response = ModelResponse(text=["one", "two", "one", "two"])

    assert scorer.compute(doc, response) == 0.5
    assert str(scorer) == "RWKVAvgAtK(k=4)"


def test_rwkv_avg_at_k_delegates_answer_extraction_to_sampling_scorer():
    metric = SampleLevelMetric(
        metric_name="maj@n",
        sample_level_fn=MajAtN(n=1, sample_scoring_function=MathVerifyMatch()),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=lambda values: sum(values) / len(values),
        higher_is_better=True,
    )
    scorer = rwkv_pipeline.RWKVAvgAtK(1, metric)
    doc = Doc(query="question", choices=["-371"], gold_index=0)
    response = ModelResponse(text=["230 − 601 = −371\nFinal answer: -371"], finish_reasons=["stop"])

    assert scorer.score_rollout(doc, response) == 1.0
    assert scorer.extract_rollout_answer(doc, response) == "-371"


def test_rwkv_avg_at_k_scores_truncated_rollout_as_zero():
    metric = SampleLevelMetric(
        metric_name="accuracy",
        sample_level_fn=ExactMatches(strip_strings=True),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=lambda values: sum(values) / len(values),
        higher_is_better=True,
    )
    scorer = rwkv_pipeline.RWKVAvgAtK(1, metric)
    doc = Doc(query="question", choices=["answer"], gold_index=0)
    response = ModelResponse(text=["answer"], finish_reasons=["length"])

    assert scorer.compute(doc, response) == 0.0


def test_rwkv_pipeline_exposes_only_avg_at_k_and_updates_document_counts():
    doc = Doc(
        query="question",
        choices=["one", "two"],
        gold_index=0,
        sampling_methods=[SamplingMethod.GENERATIVE],
    )
    metric = SampleLevelMetric(
        metric_name="accuracy",
        sample_level_fn=ExactMatches(strip_strings=True),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=lambda values: sum(values) / len(values),
        higher_is_better=True,
    )
    task = SimpleNamespace(
        full_name="task|0",
        metrics=(metric,),
        eval_docs=lambda: [object()] * 500,
        config=SimpleNamespace(metrics=(metric,), original_num_docs=-1, effective_num_docs=-1),
    )
    pipeline = SimpleNamespace(
        tasks_dict={task.full_name: task},
        documents_dict={task.full_name: [doc]},
        pipeline_parameters=SimpleNamespace(max_samples=None),
        evaluation_tracker=SimpleNamespace(task_config_logger=SimpleNamespace(log=lambda _tasks: None)),
    )

    pipeline.documents_dict[task.full_name] = rwkv_pipeline._configure_task_evaluation_plan(
        pipeline,
        task,
        pipeline.documents_dict[task.full_name],
    )

    assert [metric.metric_name for metric in task.metrics] == ["avg@16"]
    assert doc.num_samples == 16
    assert task.num_samples == [1, 16]
    assert task.config.original_num_docs == 500
    assert task.config.effective_num_docs == 1


def test_rwkv_partial_run_uses_avg_at_one():
    docs = [
        Doc(
            query=f"question {index}",
            choices=["answer"],
            gold_index=0,
            sampling_methods=[SamplingMethod.GENERATIVE],
        )
        for index in range(10)
    ]
    metric = SampleLevelMetric(
        metric_name="accuracy",
        sample_level_fn=ExactMatches(strip_strings=True),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=lambda values: sum(values) / len(values),
        higher_is_better=True,
    )
    task = SimpleNamespace(
        full_name="task|0",
        metrics=(metric,),
        eval_docs=lambda: [object()] * 30,
        config=SimpleNamespace(metrics=(metric,), original_num_docs=-1, effective_num_docs=-1),
    )
    pipeline = SimpleNamespace(
        tasks_dict={task.full_name: task},
        documents_dict={task.full_name: docs},
        pipeline_parameters=SimpleNamespace(max_samples=10),
        evaluation_tracker=SimpleNamespace(task_config_logger=SimpleNamespace(log=lambda _tasks: None)),
    )

    pipeline.documents_dict[task.full_name] = rwkv_pipeline._configure_task_evaluation_plan(
        pipeline,
        task,
        pipeline.documents_dict[task.full_name],
    )

    assert [configured.metric_name for configured in task.metrics] == ["avg@1"]
    assert len(pipeline.documents_dict[task.full_name]) == 10
    assert all(doc.num_samples == 1 for doc in docs)
    assert task.num_samples == [1, 1]
    assert task.config.original_num_docs == 30
    assert task.config.effective_num_docs == 10


def test_lcb_outer_workers_use_spawn_context(monkeypatch):
    from lighteval.tasks.tasks.lcb import codegen_metrics

    contexts = []

    class Future:
        @staticmethod
        def result():
            return [True]

    class Executor:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == 1
            contexts.append(mp_context.get_start_method())

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def submit(_function, _argument):
            return Future()

    monkeypatch.setattr(codegen_metrics, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(codegen_metrics, "as_completed", iter)

    results = codegen_metrics.evaluate_generations([{}], [["code"]], num_process_evaluate=1)

    assert contexts == ["spawn"]
    assert results == {0: [True]}
