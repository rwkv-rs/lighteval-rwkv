import asyncio
import gzip
import json
import signal
import threading
from collections import Counter
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

import lighteval.logging.scoreboard as scoreboard_module
import lighteval.main_rwkv as main_rwkv
import temp.main_rwkv as launcher
from lighteval.logging.info_loggers import DetailsLogger, MetricsLogger
from lighteval.logging.scoreboard import ScoreboardCallback
from lighteval.metrics.metrics_sample import AvgAtN, ExactMatches
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


def _app():
    app = typer.Typer()
    app.command()(main_rwkv.rwkv)
    return app


def _config(tmp_path):
    return main_rwkv.RWKVEvaluationConfig(
        pool_manifest=tmp_path / "pool.json",
        output_dir=tmp_path / "results",
        prompt_template="bot",
        cot_mode="open_think",
        benchmarks=("gsm8k", "ifeval"),
    )


def _manifest():
    return SimpleNamespace(
        model_name="RWKV7-g1h-7.2B-20260710-ctx10240",
        served_model_name="served",
        model_revision="weight-sha",
        wkv_mode="fp32io16",
        vllm_version="0.11.0",
        aggregate_capacity=5,
        fingerprint="f" * 64,
    )


def _streaming_pipeline(task_names, model, download, *, max_samples):
    pipeline = main_rwkv.RWKVPipeline.__new__(main_rwkv.RWKVPipeline)
    pipeline._task_names = tuple(task_names)
    pipeline.tasks_dict = {
        task_name: SimpleNamespace(
            full_name=task_name,
            download_dataset_worker=lambda task: download(task.full_name),
        )
        for task_name in task_names
    }
    pipeline.documents_dict = {}
    pipeline.sampling_docs = main_rwkv.defaultdict(list)
    pipeline._datasets_loaded = 0
    pipeline._all_datasets_ready_at = None
    pipeline.pipeline_parameters = SimpleNamespace(max_samples=max_samples)
    pipeline.model = model
    pipeline.evaluation_tracker = SimpleNamespace(
        task_config_logger=SimpleNamespace(log=lambda _tasks: None),
    )
    pipeline._prepare_task_documents = lambda task: [
        SimpleNamespace(
            task_name=task.full_name,
            num_samples=1,
            sampling_methods=[main_rwkv.SamplingMethod.GENERATIVE],
        )
    ]
    return pipeline


async def _evaluate_streaming_pipeline(pipeline):
    async def score(task_name, sampling_docs, outputs):
        pipeline._score_task(task_name, sampling_docs, outputs)

    await pipeline._evaluate_tasks(score)


def test_dry_run_preflights_without_creating_results_or_model(tmp_path, monkeypatch):
    config = _config(tmp_path)
    pool = SimpleNamespace(close=lambda: setattr(pool, "closed", True), closed=False)
    resolved = main_rwkv.ResolvedBenchmarks(
        selector_count=2,
        leaf_tasks=("gsm8k", "ifeval"),
        selector_tasks=(("gsm8k", ("gsm8k",)), ("ifeval", ("ifeval",))),
    )
    monkeypatch.setattr(main_rwkv.RWKVEvaluationConfig, "read", lambda _path: config)
    monkeypatch.setattr(main_rwkv, "_preflight", lambda _config: (_manifest(), pool, resolved))
    monkeypatch.setattr(
        main_rwkv,
        "RWKVHttpModel",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("model must not be created")),
    )

    result = CliRunner().invoke(_app(), ["--config", str(tmp_path / "eval.toml"), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "selectors: 2" in result.output
    assert "leaf tasks: 2" in result.output
    assert "run mode: full" in result.output
    assert pool.closed is True
    assert not config.output_dir.exists()


def test_cli_reports_all_resolution_errors_with_nonzero_exit(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(main_rwkv.RWKVEvaluationConfig, "read", lambda _path: config)
    monkeypatch.setattr(
        main_rwkv,
        "_preflight",
        lambda _config: (_ for _ in ()).throw(
            main_rwkv.ConfigError("benchmark selectors are not registered: missing_a, missing_b")
        ),
    )

    result = CliRunner().invoke(_app(), ["--config", str(tmp_path / "eval.toml"), "--dry-run"])

    assert result.exit_code == 2
    assert "missing_a, missing_b" in result.output


def test_partial_run_uses_native_pipeline_and_standard_saving(tmp_path, monkeypatch):
    config = _config(tmp_path)
    pool = SimpleNamespace(close=lambda: None)
    resolved = main_rwkv.ResolvedBenchmarks(
        selector_count=2,
        leaf_tasks=("gsm8k", "ifeval"),
        selector_tasks=(("gsm8k", ("gsm8k",)), ("ifeval", ("ifeval",))),
    )
    calls = []
    monkeypatch.delenv("SCOREBOARD_API_BASE_URL", raising=False)
    monkeypatch.delenv("SCOREBOARD_PUBLICATION_TOKEN", raising=False)

    class Model:
        def __init__(self, **kwargs):
            calls.append(("model", kwargs))

        def cleanup(self):
            calls.append(("cleanup",))

    class Tracker:
        def __init__(self, **kwargs):
            calls.append(("tracker", kwargs))

    class Pipeline:
        def __init__(self, **kwargs):
            calls.append(("pipeline", kwargs))

        def evaluate(self):
            calls.append(("evaluate",))

        def show_results(self):
            calls.append(("show",))

        def save_and_push_results(self):
            calls.append(("save",))

    monkeypatch.setattr(main_rwkv.RWKVEvaluationConfig, "read", lambda _path: config)
    monkeypatch.setattr(main_rwkv, "_preflight", lambda _config: (_manifest(), pool, resolved))
    monkeypatch.setattr(main_rwkv, "RWKVHttpModel", Model)
    monkeypatch.setattr("lighteval.logging.evaluation_tracker.EvaluationTracker", Tracker)
    monkeypatch.setattr(main_rwkv, "RWKVPipeline", Pipeline)

    result = CliRunner().invoke(
        _app(),
        ["--config", str(tmp_path / "eval.toml"), "--max-samples", "3"],
    )

    assert result.exit_code == 0, result.output
    assert "run mode: partial (max_samples=3)" in result.output
    model_call = next(call for call in calls if call[0] == "model")
    assert model_call[1]["max_samples"] == 3
    pipeline_call = next(call for call in calls if call[0] == "pipeline")
    assert pipeline_call[1]["tasks"] == "gsm8k,ifeval"
    assert pipeline_call[1]["pipeline_parameters"].max_samples == 3
    assert pipeline_call[1]["pipeline_parameters"].load_tasks_multilingual is True
    assert pipeline_call[1]["pipeline_parameters"].convert_logprob_choices_to_generation is True
    assert pipeline_call[1]["selector_tasks"] == {"gsm8k": ("gsm8k",), "ifeval": ("ifeval",)}
    assert pipeline_call[1]["task_max_samples"] == {"gsm8k": 3, "ifeval": 3}
    assert ("evaluate",) in calls
    assert ("show",) in calls
    assert ("save",) in calls
    assert calls[-1] == ("cleanup",)


def test_four_model_launcher_forwards_signals_to_every_process(tmp_path, monkeypatch):
    handlers = {}
    processes = []

    class Process:
        def __init__(self, command, *, cwd, env):
            self.command = command
            self.cwd = cwd
            self.env = env
            self.done = False
            self.signals = []
            processes.append(self)

        def poll(self):
            return 0 if self.done else None

        def send_signal(self, signum):
            self.signals.append(signum)

        def wait(self):
            if self is processes[0]:
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            self.done = True
            return 0

    monkeypatch.setattr(launcher, "_validate", lambda _evaluations: None)
    monkeypatch.setattr(launcher.signal, "signal", lambda signum, handler: handlers.setdefault(signum, handler))
    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    manifests = [tmp_path / f"{size}.json" for size in ("1.5b", "2.9b", "7.2b", "13.3b")]

    result = launcher.main(
        [
            "--manifest-1.5b",
            str(manifests[0]),
            "--manifest-2.9b",
            str(manifests[1]),
            "--manifest-7.2b",
            str(manifests[2]),
            "--manifest-13.3b",
            str(manifests[3]),
        ]
    )

    assert result == 128 + signal.SIGTERM
    assert len(processes) == 4
    assert all(process.command[-2:] == ["--max-samples", "10"] for process in processes)
    assert [process.env["RWKV_EVAL_POOL_MANIFEST"] for process in processes] == [
        str(manifest.resolve()) for manifest in manifests
    ]
    assert all(process.signals == [signal.SIGTERM] for process in processes)


def test_rwkv_pipeline_starts_fast_task_before_slow_dataset_finishes(monkeypatch):
    slow_release = threading.Event()
    slow_finished = threading.Event()
    model_called = threading.Event()
    calls = []

    def download(task_name):
        if task_name == "slow|0":
            slow_release.wait(2)
            slow_finished.set()
        return task_name

    class Model:
        pool = SimpleNamespace(http_worker_limit=20)

        async def greedy_until(self, docs):
            calls.append(docs[0].task_name)
            model_called.set()
            return []

        async def acleanup(self):
            pass

    pipeline = _streaming_pipeline(("fast|0", "slow|0"), Model(), download, max_samples=10)
    pipeline._score_task = lambda *_args: None
    monkeypatch.setattr(main_rwkv, "_configure_task_evaluation_plan", lambda _pipeline, _task, docs: docs)

    async def run():
        evaluation = asyncio.create_task(_evaluate_streaming_pipeline(pipeline))
        try:
            assert await asyncio.to_thread(model_called.wait, 1)
            assert calls[0] == "fast|0"
            assert not slow_finished.is_set()
        finally:
            slow_release.set()
        await evaluation

    asyncio.run(run())


def test_rwkv_pipeline_task_concurrency_matches_run_mode():
    model = SimpleNamespace(pool=SimpleNamespace(http_worker_limit=25))
    pipeline = _streaming_pipeline(("a|0", "b|0", "c|0"), model, lambda task_name: task_name, max_samples=None)

    assert pipeline._task_concurrency() == 1
    pipeline.pipeline_parameters.max_samples = 10
    assert pipeline._task_concurrency() == 3


def test_rwkv_pipeline_runs_scorer_on_process_main_thread():
    pipeline = main_rwkv.RWKVPipeline.__new__(main_rwkv.RWKVPipeline)
    pipeline.pipeline_parameters = SimpleNamespace(num_fewshot_seeds=1, max_samples=10, job_id=0)
    pipeline.model = SimpleNamespace(
        pool=SimpleNamespace(
            peak_inflight=(2,),
            first_request_at=1.0,
            manifest=SimpleNamespace(replicas=(SimpleNamespace(max_concurrency=4),)),
        )
    )
    pipeline._all_datasets_ready_at = 2.0
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
            sampling_methods=[main_rwkv.SamplingMethod.GENERATIVE],
        )
    ]
    pipeline._score_task = lambda *_args: scored.append("task|0")
    monkeypatch.setattr(main_rwkv, "_configure_task_evaluation_plan", lambda _pipeline, _task, docs: docs)

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
    monkeypatch.setattr(main_rwkv, "_configure_task_evaluation_plan", lambda _pipeline, _task, docs: docs)

    with pytest.raises(ValueError, match=r"failed\|0"):
        asyncio.run(_evaluate_streaming_pipeline(pipeline))

    assert cancelled == ["pending|0", "closed"]


def test_scoreboard_callback_selects_twenty_samples_per_outcome():
    details = []
    for outcome in ("correct", "incorrect", "unanswered"):
        for index in range(25):
            text = {"correct": "answer", "incorrect": "wrong", "unanswered": ""}[outcome]
            response = ModelResponse(
                input=f"prompt-{outcome}-{index}",
                text=[text],
                output_tokens=[[1]],
                finish_reasons=["length" if outcome == "unanswered" else "stop"],
            )
            details.append(
                DetailsLogger.Detail(
                    doc=Doc(query="question", choices=["answer"], gold_index=0, id=str(index)),
                    model_response=response,
                    metric={"avg@1": float(outcome == "correct")},
                )
            )

    native_metric = main_rwkv.SampleLevelMetric(
        metric_name="accuracy",
        sample_level_fn=ExactMatches(strip_strings=True),
        category=main_rwkv.SamplingMethod.GENERATIVE,
        corpus_level_fn=lambda values: sum(values) / len(values),
        higher_is_better=True,
    )
    task = SimpleNamespace(
        full_name="task|0",
        metrics=(SimpleNamespace(sample_level_fn=main_rwkv.RWKVAvgAtK(1, native_metric)),),
    )
    selected, totals = ScoreboardCallback._select_samples(ScoreboardCallback._rollouts(task, details))

    assert totals == {"correct": 25, "incorrect": 25, "unanswered": 25}
    assert Counter(rollout.outcome for rollout in selected) == {
        "correct": 20,
        "incorrect": 20,
        "unanswered": 20,
    }


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_scoreboard_discards_truncated_answer(score):
    response = SimpleNamespace(final_text=["B"], finish_reasons=["length"])

    assert ScoreboardCallback._outcome(response, score) == "unanswered"


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
    assert main_rwkv._evaluation_plan(num_docs) == (effective_docs, k, metric_name)
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

    monkeypatch.setattr(main_rwkv, "Registry", Registry)
    pipeline = main_rwkv.RWKVPipeline.__new__(main_rwkv.RWKVPipeline)
    pipeline._configured_selector_tasks = {"mmlu": ("mmlu:a", "mmlu:b"), "gsm8k": ("gsm8k",)}
    pipeline._configured_task_max_samples = {"mmlu:b": 1, "gsm8k": 10}
    pipeline.pipeline_parameters = SimpleNamespace(
        load_tasks_multilingual=True,
        custom_tasks_directory=None,
        convert_logprob_choices_to_generation=True,
    )
    pipeline._metric_options = None

    pipeline._init_tasks_and_requests("mmlu,gsm8k")

    assert tuple(pipeline.tasks_dict) == ("mmlu:b|0", "gsm8k|0")
    assert pipeline._task_names == ("mmlu:b|0", "gsm8k|0")
    assert pipeline._selector_tasks == {"mmlu": ("mmlu:b|0",), "gsm8k": ("gsm8k|0",)}
    assert pipeline._task_max_samples == {"mmlu:b|0": 1, "gsm8k|0": 10}


def test_rwkv_avg_at_k_averages_the_native_task_scorer():
    metric = main_rwkv.SampleLevelMetric(
        metric_name="avg@n:n=64",
        sample_level_fn=AvgAtN(n=64, sample_scoring_function=ExactMatches(strip_strings=True)),
        category=main_rwkv.SamplingMethod.GENERATIVE,
        corpus_level_fn=lambda values: sum(values) / len(values),
        higher_is_better=True,
    )
    scorer = main_rwkv.RWKVAvgAtK(4, metric)
    doc = Doc(query="question", choices=["one", "two"], gold_index=0)
    response = ModelResponse(text=["one", "two", "one", "two"])

    assert scorer.compute(doc, response) == 0.5
    assert str(scorer) == "RWKVAvgAtK(k=4)"


def test_rwkv_pipeline_exposes_only_avg_at_k_and_updates_document_counts():
    doc = Doc(
        query="question",
        choices=["one", "two"],
        gold_index=0,
        sampling_methods=[main_rwkv.SamplingMethod.GENERATIVE],
    )
    metric = main_rwkv.SampleLevelMetric(
        metric_name="accuracy",
        sample_level_fn=ExactMatches(strip_strings=True),
        category=main_rwkv.SamplingMethod.GENERATIVE,
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

    pipeline.documents_dict[task.full_name] = main_rwkv._configure_task_evaluation_plan(
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
            sampling_methods=[main_rwkv.SamplingMethod.GENERATIVE],
        )
        for index in range(10)
    ]
    metric = main_rwkv.SampleLevelMetric(
        metric_name="accuracy",
        sample_level_fn=ExactMatches(strip_strings=True),
        category=main_rwkv.SamplingMethod.GENERATIVE,
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

    pipeline.documents_dict[task.full_name] = main_rwkv._configure_task_evaluation_plan(
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


def test_scoreboard_splits_rollouts_and_scores_each_one():
    detail = DetailsLogger.Detail(
        doc=Doc(query="question", choices=["one"], gold_index=0),
        model_response=ModelResponse(
            text=["one", ""],
            text_post_processed=["one", ""],
            output_tokens=[[1], [2, 3]],
            finish_reasons=["stop", "length"],
        ),
        metric={"avg@2": 0.5},
    )
    native_metric = main_rwkv.SampleLevelMetric(
        metric_name="accuracy",
        sample_level_fn=ExactMatches(strip_strings=True),
        category=main_rwkv.SamplingMethod.GENERATIVE,
        corpus_level_fn=lambda values: sum(values) / len(values),
        higher_is_better=True,
    )
    task = SimpleNamespace(
        full_name="task|0",
        metrics=(SimpleNamespace(sample_level_fn=main_rwkv.RWKVAvgAtK(2, native_metric)),),
    )

    rollouts = ScoreboardCallback._rollouts(task, [detail])
    samples = [ScoreboardCallback._sample(index, rollout, "avg@2") for index, rollout in enumerate(rollouts)]

    assert [(rollout.repeat_id, rollout.score, rollout.outcome) for rollout in rollouts] == [
        (0, 1.0, "correct"),
        (1, 0.0, "unanswered"),
    ]
    assert [sample["answer"]["raw_completion"] for sample in samples] == ["one", ""]
    assert [sample["answer"]["extracted_answer"] for sample in samples] == ["one", ""]
    assert [sample["answer"]["repeat_id"] for sample in samples] == [0, 1]
    assert samples[0]["metrics"] == {"scoreboard_outcome": "correct", "avg@2": 1.0}


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


def test_scoreboard_waits_for_all_internal_leaves_and_publishes_only_selector():
    callback = ScoreboardCallback.__new__(ScoreboardCallback)
    callback._pipeline = SimpleNamespace(
        _task_selectors={"mmlu:a|0": "mmlu", "mmlu:b|0": "mmlu"},
        _selector_tasks={"mmlu": ("mmlu:a|0", "mmlu:b|0")},
    )
    callback._selector_details = {}
    publications = []
    callback._publish_selector = lambda selector, tasks, details: publications.append((selector, tasks, details))

    callback("mmlu:a|0", ["a"])
    assert publications == []

    callback("mmlu:b|0", ["b"])

    assert len(publications) == 1
    assert publications[0][0] == "mmlu"
    assert publications[0][1] == ("mmlu:a|0", "mmlu:b|0")
    assert publications[0][2] == {"mmlu:a|0": ["a"], "mmlu:b|0": ["b"]}
    assert callback._selector_details == {}


def test_scoreboard_aggregates_lighteval_metadata_for_selector():
    callback = ScoreboardCallback.__new__(ScoreboardCallback)
    callback._model = SimpleNamespace(
        config=SimpleNamespace(
            model_name="RWKV7-g1h-7.2B-20260710-ctx10240", model_revision="a" * 64, wkv_mode="fp32io16"
        )
    )
    callback._task_metadata_by_name = {
        "mmlu:a": {"languages": ["english"], "tags": ["knowledge", "multiple-choice"]},
        "mmlu:b": {"languages": ["english", "chinese"], "tags": ["math", "multiple-choice"]},
    }
    tasks = [
        SimpleNamespace(
            config=SimpleNamespace(
                name=name,
                version=0,
                hf_repo="lighteval/mmlu",
                hf_subset=name,
                evaluation_splits=("test",),
            )
        )
        for name in ("mmlu:a", "mmlu:b")
    ]

    metadata = callback._task_metadata("mmlu", tasks)

    assert metadata["benchmark"] == "mmlu"
    assert metadata["task_name"] == "mmlu"
    assert metadata["languages"] == ["chinese", "english"]
    assert metadata["tags"] == ["knowledge", "math", "multiple-choice"]


def test_scoreboard_maps_lighteval_tags_to_fields():
    assert ScoreboardCallback._categories(
        [
            "general-knowledge",
            "biology",
            "arithmetic",
            "execution",
            "biomedical",
            "common-sense",
            "multi-turn",
            "translation",
            "truthfulness",
            "multimodal",
            "multiple-choice",
        ]
    ) == [
        {"id": "knowledge", "label": "世界知识"},
        {"id": "science", "label": "科学"},
        {"id": "math", "label": "数学"},
        {"id": "code", "label": "代码"},
        {"id": "medical", "label": "医疗"},
        {"id": "reasoning", "label": "推理"},
        {"id": "instruction", "label": "指令遵循"},
        {"id": "language", "label": "语言"},
        {"id": "safety", "label": "安全与价值观"},
        {"id": "multimodal", "label": "多模态"},
    ]
    assert ScoreboardCallback._categories(["multiple-choice", "qa"]) == [{"id": "other", "label": "其他"}]


def test_scoreboard_selector_keeps_unconfigured_generation_size():
    callback = ScoreboardCallback.__new__(ScoreboardCallback)
    configs = [
        SimpleNamespace(
            num_fewshots=0,
            generation_size=None,
            stop_sequence=(),
            original_num_docs=100,
            effective_num_docs=5,
        )
        for _ in range(2)
    ]

    task_config = callback._task_config([SimpleNamespace(config=config) for config in configs], "avg@1")

    assert task_config["generation_size"] is None
    assert task_config["original_num_docs"] == 200
    assert task_config["effective_num_docs"] == 10


def test_scoreboard_publication_keeps_only_display_fields(tmp_path, monkeypatch):
    campaign_id = "12345678-1234-5678-1234-567812345678"
    requests = []

    class Response:
        status_code = 200
        text = ""

        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def request(method, url, *, content, headers, timeout):
        requests.append(SimpleNamespace(method=method, url=url, data=content, headers=headers))
        assert timeout == 60
        if method == "GET":
            return Response({"status": "ready"})
        if url.endswith("/api/v1/evaluation-campaigns"):
            return Response({"campaign_id": campaign_id})
        return Response({"action": "created"})

    monkeypatch.setattr(scoreboard_module.httpx, "request", request)
    config_path = tmp_path / "eval.toml"
    config_path.write_text("schema_version = 1\n", encoding="utf-8")
    task_config = SimpleNamespace(
        name="gsm8k",
        version=0,
        hf_repo="openai/gsm8k",
        hf_subset="main",
        evaluation_splits=("test",),
        num_fewshots=0,
        generation_size=8192,
        stop_sequence=(),
        original_num_docs=1,
        effective_num_docs=1,
    )
    task = SimpleNamespace(
        config=task_config,
        full_name="gsm8k|0",
        num_samples=[1],
        metrics=(
            SimpleNamespace(
                metric_name="avg@1",
                sample_level_fn=main_rwkv.RWKVAvgAtK(
                    1,
                    main_rwkv.SampleLevelMetric(
                        metric_name="accuracy",
                        sample_level_fn=ExactMatches(strip_strings=True),
                        category=main_rwkv.SamplingMethod.GENERATIVE,
                        corpus_level_fn=lambda values: sum(values) / len(values),
                        higher_is_better=True,
                    ),
                ),
            ),
        ),
        aggregation=lambda: {"avg@1": lambda values: sum(values) / len(values)},
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_name="RWKV7-g1h-7.2B-20260710-ctx10240",
            model_revision="a" * 64,
            wkv_mode="fp32io16",
            max_samples=None,
            served_model_name="rwkv",
            vllm_version="0.11.0",
            pool_fingerprint="b" * 64,
            max_model_length=10240,
            prompt_template="bot",
            cot_mode="open_think",
        ),
        _generation_parameters={"temperature": 0.96, "top_p": 0.76, "top_k": 32},
        _completion_limit=lambda _doc: 8192,
        _stop_sequences=lambda _doc: ["✿"],
    )
    pipeline = SimpleNamespace(
        tasks_dict={"gsm8k|0": task},
        documents_dict={"gsm8k|0": [SimpleNamespace(num_samples=1)]},
        _task_selectors={"gsm8k|0": "gsm8k"},
        _selector_tasks={"gsm8k": ("gsm8k|0",)},
        registry=SimpleNamespace(
            get_tasks_dump=lambda: [
                {
                    "docstring": {"languages": ["english"], "tags": ["math", "reasoning"]},
                    "tasks": [{"name": "gsm8k"}],
                }
            ]
        ),
    )
    tracker = SimpleNamespace(
        metrics_logger=MetricsLogger(),
        general_config_logger=SimpleNamespace(lighteval_sha="c" * 40),
    )
    callback = ScoreboardCallback(
        base_url="https://scoreboard.example/test",
        token="secret",
        config_path=config_path,
        pipeline=pipeline,
        tracker=tracker,
        model=model,
    )
    detail = DetailsLogger.Detail(
        doc=Doc(query="What is 1 + 1?", choices=["2"], gold_index=0, id="7"),
        model_response=ModelResponse(
            input="User: What is 1 + 1?",
            input_tokens=[1, 2, 3],
            text=["2"],
            output_tokens=[[4]],
        ),
        metric={"avg@1": 1.0},
    )
    tracker.metrics_logger.log("gsm8k|0", detail.metric)

    callback("gsm8k|0", [detail])

    publication_request = next(request for request in requests if request.method == "PUT")
    publication = json.loads(gzip.decompress(publication_request.data))
    sample = publication["samples"][0]
    assert publication["sampling_config"]["temperature"] == 0.96
    assert publication["sampling_config"]["num_samples"] == 1
    assert publication["task_config"]["k_metrics"] == "avg@1"
    assert publication["task"]["benchmark"] == "gsm8k"
    assert publication["task"]["task_name"] == "gsm8k"
    assert publication["task"]["languages"] == ["english"]
    assert publication["task"]["tags"] == ["math", "reasoning"]
    assert publication["comparison"]["benchmark"]["categories"] == [
        {"id": "math", "label": "数学"},
        {"id": "reasoning", "label": "推理"},
    ]
    assert publication["comparison"]["coordinates"][0]["comparison"]["id"] == "precision"
    assert publication["comparison"]["coordinates"][0]["arm"] == "b"
    assert publication["comparison"]["samples"] == 1
    assert sample["document_index"] == 0
    assert sample["metrics"]["scoreboard_outcome"] == "correct"
    assert sample["model_response"]["text"] == ["2"]
    assert sample["answer"]["outcome"] == "correct"
    assert sample["answer"]["problem_id"] == "gsm8k|0:7"
    assert sample["answer"]["ground_truth"] == "2"
    assert sample["answer"]["assembled_prompt"] == "User: What is 1 + 1?"
    assert sample["answer"]["raw_completion"] == "2"
    assert "input_tokens" not in sample["model_response"]
    assert "output_tokens" not in sample["model_response"]
