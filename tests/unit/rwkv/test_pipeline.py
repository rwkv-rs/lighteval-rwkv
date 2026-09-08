import asyncio
import threading
from collections import defaultdict
from types import SimpleNamespace

import pytest

import lighteval.main_rwkv as main_rwkv
import lighteval.models.rwkv.pipeline as rwkv_pipeline
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import AvgAtN, ExactMatches, MajAtN, SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.rwkv_free_response import RWKVFreeResponseMatch
from lighteval.tasks.tasks.ifbench.instructions import (
    EmojiSentenceChecker,
    NGramOverlapChecker,
    ParagraphLastFirstWordMatchChecker,
)


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
        assert calls[0] == "cached|0"
        assert set(calls) == {"cached|0", "small|0", "spare|0"}
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


def test_pending_scoring_releases_generation_slots_without_finishing_evaluation(monkeypatch):
    calls = []

    class Model:
        async def greedy_until(self, docs):
            calls.append(docs[0].task_name)
            return []

        async def acleanup(self):
            pass

    pipeline = _streaming_pipeline(("first|0", "second|0", "third|0"), Model(), lambda name: name, max_samples=10)
    monkeypatch.setattr(rwkv_pipeline, "_configure_task_evaluation_plan", lambda _pipeline, _task, docs: docs)

    async def run():
        release = asyncio.Event()
        all_generated = asyncio.Event()

        async def score(*_args):
            if len(calls) == 3:
                all_generated.set()
            await release.wait()

        evaluation = asyncio.create_task(pipeline._evaluate_tasks(score))
        try:
            await asyncio.wait_for(all_generated.wait(), 2)
            assert set(calls) == {"first|0", "second|0", "third|0"}
            assert not evaluation.done()
        finally:
            release.set()
            await evaluation

    asyncio.run(run())


def test_ifbench_checkers_treat_empty_responses_as_failed():
    overlap = NGramOverlapChecker("test")
    overlap.build_description(reference_text="reference text", percentage=50)
    assert overlap.check_following("") is False

    emoji = EmojiSentenceChecker("test")
    assert emoji.check_following("!!!") is False

    paragraph = ParagraphLastFirstWordMatchChecker("test")
    paragraph.build_description()
    assert paragraph.check_following("!!!") is False
    assert paragraph.check_following("word other word\n---") is False
    assert paragraph.check_following("word other word") is True


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


def test_rwkv_pipeline_always_converts_choices():
    doc = Doc(
        query="Question?",
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
    pipeline.pipeline_parameters = SimpleNamespace(max_samples=None)
    pipeline.model = SimpleNamespace(config=SimpleNamespace(cot_mode="fake_think"))

    assert pipeline._prepare_task_documents(task) == [doc]
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
        (30, 30, 128, "avg@128"),
        (0, 0, 1, "avg@1"),
        (500, 500, 8, "avg@8"),
        (1251, 1251, 4, "avg@4"),
        (4096, 4096, 1, "avg@1"),
        (5000, 5000, 1, "avg@1"),
        (5001, 5001, 1, "avg@1"),
        (50_001, 50_001, 1, "avg@1"),
    ],
)
def test_evaluation_plan_uses_bounded_power_of_two_completion_budget(num_docs, effective_docs, k, metric_name):
    assert rwkv_pipeline._evaluation_plan(num_docs) == (effective_docs, k, metric_name)
    if 0 < num_docs <= 50_000:
        assert k & (k - 1) == 0
        if num_docs * k >= 3000:
            assert 3000 <= k * num_docs <= 6000
            assert abs(k * num_docs - 4096) <= abs((k // 2) * num_docs - 4096) if k > 1 else True


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
        sample_level_fn=MajAtN(n=1, sample_scoring_function=RWKVFreeResponseMatch()),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=lambda values: sum(values) / len(values),
        higher_is_better=True,
    )
    scorer = rwkv_pipeline.RWKVAvgAtK(1, metric)
    doc = Doc(query="question", choices=["-371"], gold_index=0)
    response = ModelResponse(text=["230 − 601 = −371\nFinal answer: -371"], finish_reasons=["stop"])

    assert scorer.score_rollout(doc, response) == 1.0
    assert scorer.extract_rollout_answer(doc, response) == "-371"


def test_rwkv_math_rollouts_keep_the_prediction_used_for_each_score():
    metric = SampleLevelMetric(
        metric_name="accuracy",
        sample_level_fn=RWKVFreeResponseMatch(),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=lambda values: sum(values) / len(values),
        higher_is_better=True,
    )
    scorer = rwkv_pipeline.RWKVAvgAtK(2, metric)
    doc = Doc(query="question", choices=["19"], gold_index=0)
    response = ModelResponse(
        text=["x = 20\n\nThe value is 19", "The final answer is 20"],
        finish_reasons=["stop", "stop"],
    )

    assert scorer.compute(doc, response) == 0.5
    assert doc.specific["rwkv_rollout_scores"] == [1.0, 0.0]
    assert doc.specific["rwkv_rollout_extracted_answers"] == ["19", "20"]


def test_rwkv_avg_at_k_reuses_native_extractive_prediction():
    class NativeExtractor(SampleLevelComputation):
        def compute(self, doc, model_response=None, **_kwargs):
            doc.specific = {"extracted_predictions": ["19"]}
            return 1.0

    metric = SampleLevelMetric(
        metric_name="accuracy",
        sample_level_fn=NativeExtractor(),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=lambda values: sum(values) / len(values),
        higher_is_better=True,
    )
    scorer = rwkv_pipeline.RWKVAvgAtK(1, metric)
    doc = Doc(query="question", choices=["19"], gold_index=0)
    response = ModelResponse(text=["long reasoning... final answer: 19"], finish_reasons=["stop"])

    assert scorer.compute(doc, response) == 1.0
    assert doc.specific["rwkv_rollout_extracted_answers"] == ["19"]
    assert doc.specific["rwkv_model_answer"] == "19"


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


def test_rwkv_avg_at_k_records_empty_answer_without_index_error():
    class EmptyExtractor(SampleLevelComputation):
        def compute(self, doc, model_response=None, **_kwargs):
            return 0.0

        def extract_answer(self, doc, model_response):
            return ""

    metric = SampleLevelMetric(
        metric_name="accuracy",
        sample_level_fn=EmptyExtractor(),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=lambda values: sum(values) / len(values),
        higher_is_better=True,
    )
    scorer = rwkv_pipeline.RWKVAvgAtK(1, metric)
    doc = Doc(query="question", choices=["answer"], gold_index=0)
    response = ModelResponse(text=["wrong"], finish_reasons=["stop"])

    assert scorer.compute(doc, response) == 0.0
    assert doc.specific["rwkv_model_answer"] == ""


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

    assert [metric.metric_name for metric in task.metrics] == ["avg@8"]
    assert doc.num_samples == 8
    assert task.num_samples == [1, 8]
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


def test_lcb_outer_workers_reuse_spawn_context(monkeypatch):
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

        @staticmethod
        def submit(_function, _argument):
            return Future()

    monkeypatch.setattr(codegen_metrics, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(codegen_metrics, "as_completed", iter)

    codegen_metrics._evaluation_executor.cache_clear()
    try:
        results = codegen_metrics.evaluate_generations([{}], [["code"]], num_process_evaluate=1)
        repeated = codegen_metrics.evaluate_generations([{}], [["code"]], num_process_evaluate=1)
    finally:
        codegen_metrics._evaluation_executor.cache_clear()

    assert contexts == ["spawn"]
    assert results == repeated == {0: [True]}


def test_lcb_worker_pool_failure_is_not_scored_as_a_wrong_answer(monkeypatch):
    from concurrent.futures.process import BrokenProcessPool

    from lighteval.tasks.tasks.lcb import codegen_metrics

    class Future:
        @staticmethod
        def result():
            raise BrokenProcessPool("worker exited")

    class Executor:
        @staticmethod
        def shutdown(*_args, **_kwargs):
            pass

        @staticmethod
        def submit(_function, _argument):
            return Future()

    monkeypatch.setattr(codegen_metrics, "ProcessPoolExecutor", lambda **_kwargs: Executor())
    monkeypatch.setattr(codegen_metrics, "as_completed", iter)

    sample = {"input_output": '{"inputs": ["1", "2"]}'}
    codegen_metrics._evaluation_executor.cache_clear()
    try:
        with pytest.raises(BrokenProcessPool, match="worker exited"):
            codegen_metrics.evaluate_generations(
                samples_list=[sample], generations_list=[["code", "code2"]], num_process_evaluate=1
            )
    finally:
        codegen_metrics._evaluation_executor.cache_clear()


def test_rwkv_avg_at_k_persists_producer_rollout_facts():
    native_metric = SampleLevelMetric(
        metric_name="accuracy",
        sample_level_fn=ExactMatches(strip_strings=True),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=lambda values: sum(values) / len(values),
        higher_is_better=True,
    )
    doc = Doc(query="question", choices=["one"], gold_index=0, specific={"source": "native"})
    response = ModelResponse(text=["one", "wrong"], finish_reasons=["stop", "stop"])

    assert rwkv_pipeline.RWKVAvgAtK(2, native_metric).compute(doc, response) == 0.5
    assert doc.specific == {
        "source": "native",
        "rwkv_rollout_scores": [1.0, 0.0],
        "rwkv_rollout_extracted_answers": ["one", "wrong"],
        "rwkv_model_answers": ["one", "wrong"],
    }
