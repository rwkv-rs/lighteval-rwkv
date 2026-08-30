import gzip
import json
from collections import Counter
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

import lighteval.logging.scoreboard as scoreboard_module
import lighteval.main_rwkv as main_rwkv
from lighteval.logging.info_loggers import DetailsLogger, MetricsLogger
from lighteval.logging.scoreboard import ScoreboardCallback, TaskCallbackDetailsLogger
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


def test_dry_run_preflights_without_creating_results_or_model(tmp_path, monkeypatch):
    config = _config(tmp_path)
    pool = SimpleNamespace(close=lambda: setattr(pool, "closed", True), closed=False)
    resolved = main_rwkv.ResolvedBenchmarks(selector_count=2, leaf_tasks=("gsm8k", "ifeval"))
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
    resolved = main_rwkv.ResolvedBenchmarks(selector_count=2, leaf_tasks=("gsm8k", "ifeval"))
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
    monkeypatch.setattr(main_rwkv, "_configure_evaluation_plan", lambda _pipeline: calls.append(("plan",)))
    monkeypatch.setattr("lighteval.logging.evaluation_tracker.EvaluationTracker", Tracker)
    monkeypatch.setattr("lighteval.pipeline.Pipeline", Pipeline)

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
    assert ("evaluate",) in calls
    assert ("show",) in calls
    assert ("save",) in calls
    assert calls[-1] == ("cleanup",)


def test_scoreboard_callback_selects_twenty_samples_per_outcome():
    details = []
    for outcome in ("correct", "incorrect", "unanswered"):
        for index in range(25):
            response = ModelResponse(
                input=f"prompt-{outcome}-{index}",
                text=["answer"],
                output_tokens=[[1]],
                truncated_tokens_count=int(outcome == "unanswered"),
            )
            details.append(
                DetailsLogger.Detail(
                    doc=Doc(query="question", choices=["answer"], gold_index=0, id=str(index)),
                    model_response=response,
                    metric={"accuracy": float(outcome == "correct")},
                )
            )

    selected, totals = ScoreboardCallback._select_samples(details, "accuracy")

    assert totals == {"correct": 25, "incorrect": 25, "unanswered": 25}
    assert Counter(outcome for _, outcome in selected) == {
        "correct": 20,
        "incorrect": 20,
        "unanswered": 20,
    }


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
def test_evaluation_plan_uses_power_of_two_k_or_twenty_percent(
    num_docs, effective_docs, k, metric_name
):
    assert main_rwkv._evaluation_plan(num_docs) == (effective_docs, k, metric_name)
    if num_docs <= 50_000:
        assert k & (k - 1) == 0
        assert k * num_docs > 5000
        assert k == 1 or (k // 2) * num_docs <= 5000


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
        evaluation_tracker=SimpleNamespace(task_config_logger=SimpleNamespace(log=lambda _tasks: None)),
    )

    main_rwkv._configure_evaluation_plan(pipeline)

    assert [metric.metric_name for metric in task.metrics] == ["avg@16"]
    assert doc.num_samples == 16
    assert task.num_samples == [1, 16]
    assert task.config.original_num_docs == 500
    assert task.config.effective_num_docs == 1


def test_scoreboard_partial_truncation_is_not_an_unanswered_task_sample():
    detail = DetailsLogger.Detail(
        doc=Doc(query="question", choices=["one"], gold_index=0),
        model_response=ModelResponse(text=["answer", ""], truncated_tokens_count=1),
        metric={"avg@2": 0.5},
    )

    assert ScoreboardCallback._outcome(detail, "avg@2") == "incorrect"


def test_scoreboard_callback_runs_when_each_task_stores_its_last_detail():
    calls = []
    logger = TaskCallbackDetailsLogger(
        {"task-a": 2, "task-b": 1},
        lambda task_name, details: calls.append((task_name, len(details))),
    )
    doc = Doc(query="question", choices=["answer"], gold_index=0)
    response = ModelResponse(input="prompt", text=["answer"], output_tokens=[[1]])

    logger.log("task-a", doc, response, {"accuracy": 1.0})
    assert calls == []
    logger.log("task-b", doc, response, {"accuracy": 1.0})
    logger.log("task-a", doc, response, {"accuracy": 0.0})

    assert calls == [("task-b", 1), ("task-a", 2)]


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
    )
    pipeline = SimpleNamespace(tasks_dict={"gsm8k|0": task}, documents_dict={"gsm8k|0": [object()]})
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

    publication = json.loads(gzip.decompress(requests[-1].data))
    sample = publication["samples"][0]
    assert publication["sampling_config"]["temperature"] == 0.96
    assert publication["sampling_config"]["num_samples"] == 1
    assert publication["task_config"]["k_metrics"] == "avg@1"
    assert publication["comparison"]["coordinates"][0]["comparison"]["id"] == "precision"
    assert publication["comparison"]["coordinates"][0]["arm"] == "b"
    assert publication["comparison"]["samples"] == 1
    assert sample["document_index"] == 7
    assert sample["metrics"]["scoreboard_outcome"] == "correct"
    assert sample["model_response"]["text"] == ["2"]
    assert sample["answer"]["outcome"] == "correct"
    assert sample["answer"]["ground_truth"] == "2"
    assert sample["answer"]["assembled_prompt"] == "User: What is 1 + 1?"
    assert sample["answer"]["raw_completion"] == "2"
    assert "input_tokens" not in sample["model_response"]
    assert "output_tokens" not in sample["model_response"]
