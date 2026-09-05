import gzip
import json
from collections import Counter
from types import SimpleNamespace

import pytest

import lighteval.logging.scoreboard as scoreboard_module
from lighteval.logging.info_loggers import DetailsLogger, MetricsLogger
from lighteval.logging.scoreboard import ScoreboardCallback, _sha256
from lighteval.metrics.metrics_sample import ExactMatches
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.models.rwkv.pipeline import RWKVAvgAtK
from lighteval.tasks.requests import Doc, SamplingMethod


def test_scoreboard_environment_suffix_selects_test_credentials(monkeypatch):
    captured = {}
    monkeypatch.setenv("SCOREBOARD_API_BASE_URL", "https://scoreboard.example")
    monkeypatch.setenv("SCOREBOARD_PUBLICATION_TOKEN", "production-secret")
    monkeypatch.setenv("SCOREBOARD_API_BASE_URL_TEST", "https://scoreboard.example/test")
    monkeypatch.setenv("SCOREBOARD_PUBLICATION_TOKEN_TEST", "test-secret")
    monkeypatch.setattr(
        ScoreboardCallback,
        "__init__",
        lambda _self, **kwargs: captured.update(kwargs),
    )

    ScoreboardCallback.from_environment(variable_suffix="_TEST", run_mode="test", marker="value")

    assert captured == {
        "base_url": "https://scoreboard.example/test",
        "token": "test-secret",
        "run_mode": "test",
        "rerun_reason": None,
        "marker": "value",
    }


def test_missing_test_credentials_do_not_fall_back_to_production(monkeypatch):
    monkeypatch.setenv("SCOREBOARD_API_BASE_URL", "https://scoreboard.example")
    monkeypatch.setenv("SCOREBOARD_PUBLICATION_TOKEN", "production-secret")
    monkeypatch.delenv("SCOREBOARD_API_BASE_URL_TEST", raising=False)
    monkeypatch.delenv("SCOREBOARD_PUBLICATION_TOKEN_TEST", raising=False)

    assert ScoreboardCallback.from_environment(variable_suffix="_TEST") is None


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

    native_metric = SampleLevelMetric(
        metric_name="accuracy",
        sample_level_fn=ExactMatches(strip_strings=True),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=lambda values: sum(values) / len(values),
        higher_is_better=True,
    )
    task = SimpleNamespace(
        full_name="task|0",
        metrics=(SimpleNamespace(sample_level_fn=RWKVAvgAtK(1, native_metric)),),
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

    assert ScoreboardCallback._outcome(response, score, "") == "unanswered"


def test_lcb_scoreboard_answer_is_extracted_code():
    from lighteval.tasks.tasks.lcb.main import CodegenMetric

    native_metric = SampleLevelMetric(
        metric_name="codegen",
        sample_level_fn=CodegenMetric(),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=sum,
        higher_is_better=True,
    )
    scorer = RWKVAvgAtK(1, native_metric)
    response = ModelResponse(
        text=["reasoning```python\nwrong()\n```final```python\nprint(42)\n```"],
        text_post_processed=["```python\nprint(42)\n```"],
        finish_reasons=["stop"],
    )

    assert scorer.extract_rollout_answer(Doc(query="q", choices=[""], gold_index=0), response) == "print(42)"
    assert ScoreboardCallback._outcome(response, 0.0, "") == "unanswered"


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
    native_metric = SampleLevelMetric(
        metric_name="accuracy",
        sample_level_fn=ExactMatches(strip_strings=True),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=lambda values: sum(values) / len(values),
        higher_is_better=True,
    )
    task = SimpleNamespace(
        full_name="task|0",
        metrics=(SimpleNamespace(sample_level_fn=RWKVAvgAtK(2, native_metric)),),
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


def test_scoreboard_defers_publication_error_without_retaining_details():
    callback = ScoreboardCallback.__new__(ScoreboardCallback)
    callback._pipeline = SimpleNamespace(
        _task_selectors={"task|0": "task"},
        _selector_tasks={"task": ("task|0",)},
    )
    callback._selector_details = {}
    callback.publication_errors = []
    callback._publish_selector = lambda *_args: (_ for _ in ()).throw(ValueError("payload too large"))

    callback("task|0", ["detail"])

    assert callback.publication_errors == [("task", "payload too large")]
    assert callback._selector_details == {}


def test_scoreboard_finalizes_campaign_without_pair_lookup():
    callback = ScoreboardCallback.__new__(ScoreboardCallback)
    callback._run_mode = "full"
    requests = []

    def request(method, path, *args, **kwargs):
        requests.append((method, path, args, kwargs))

    callback._request = request

    callback._finalize_campaign("campaign")

    assert [(method, path) for method, path, *_ in requests] == [
        ("POST", "/api/v1/evaluation-campaigns/campaign/finalize")
    ]


def test_scoreboard_aggregates_lighteval_metadata_for_selector():
    callback = ScoreboardCallback.__new__(ScoreboardCallback)
    callback._run_mode = "test"
    callback._model = SimpleNamespace(
        config=SimpleNamespace(
            model_name="RWKV7-g1h-7.2B-20260710-ctx10240", model_revision="a" * 64, wkv_mode="fp32io16"
        )
    )
    callback._task_metadata_by_name = {
        "mmlu:a": {"languages": ["english"], "tags": ["knowledge", "multiple-choice", "field:knowledge"]},
        "mmlu:b": {
            "languages": ["english", "chinese"],
            "tags": ["math", "multiple-choice", "field:knowledge"],
        },
    }
    callback._task_registry_by_name = {
        name: {"module": "lighteval.tasks.tasks.mmlu", "docstring": metadata}
        for name, metadata in callback._task_metadata_by_name.items()
    }
    callback._field_by_selector = {"mmlu": "knowledge"}
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
    assert metadata["field"] == "knowledge"
    assert metadata["languages"] == ["chinese", "english"]
    assert metadata["tags"] == ["knowledge", "math", "multiple-choice"]


@pytest.mark.parametrize(
    ("marker", "field"),
    [
        ("field:math", "math"),
        ("field:robotics", "robotics"),
        ("field:instruction_following", "instruction_following"),
        (f"field:a{'0_-' * 21}", f"a{'0_-' * 21}"),
    ],
)
def test_scoreboard_extracts_valid_open_task_field(marker, field):
    assert ScoreboardCallback._extract_task_field("task", ["upstream-tag", marker]) == field


@pytest.mark.parametrize("tags", [[], ["math"], ["field:math", "field:knowledge"]])
def test_scoreboard_rejects_missing_or_multiple_task_fields(tags):
    with pytest.raises(ValueError, match="exactly one field:<id> marker"):
        ScoreboardCallback._extract_task_field("task", tags)


@pytest.mark.parametrize(
    "marker",
    ["field:", "field:Math", "field:math/code", "field:math.code", "field:math:algebra", f"field:a{'0' * 64}"],
)
def test_scoreboard_rejects_invalid_task_field(marker):
    with pytest.raises(ValueError, match="invalid field marker"):
        ScoreboardCallback._extract_task_field("task", [marker])


def test_scoreboard_rejects_inconsistent_selector_fields():
    callback = ScoreboardCallback.__new__(ScoreboardCallback)
    callback._task_metadata_by_name = {
        "suite:a": {"tags": ["field:knowledge"]},
        "suite:b": {"tags": ["field:math"]},
    }
    callback._pipeline = SimpleNamespace(
        tasks_dict={
            "suite:a|0": SimpleNamespace(config=SimpleNamespace(name="suite:a")),
            "suite:b|0": SimpleNamespace(config=SimpleNamespace(name="suite:b")),
        },
        _selector_tasks={"suite": ("suite:a|0", "suite:b|0")},
    )

    with pytest.raises(ValueError, match="selector suite has inconsistent field markers"):
        callback._resolve_task_fields()


def test_scoreboard_rejects_task_field_before_network_preflight(tmp_path, monkeypatch):
    config_path = tmp_path / "eval.toml"
    config_path.write_text("schema_version = 1\n", encoding="utf-8")
    task = SimpleNamespace(config=SimpleNamespace(name="task"))
    pipeline = SimpleNamespace(
        tasks_dict={"task|0": task},
        _selector_tasks={"task": ("task|0",)},
        registry=SimpleNamespace(
            get_tasks_dump=lambda: [
                {
                    "module": "lighteval.tasks.tasks.task",
                    "docstring": {"tags": ["knowledge"]},
                    "tasks": [{"name": "task"}],
                }
            ]
        ),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_name="RWKV7-g1j-1.5B-20260831-ctx16384",
            model_revision="a" * 64,
            max_samples=None,
        )
    )
    requests = []
    monkeypatch.setattr(ScoreboardCallback, "_request", lambda *args, **kwargs: requests.append((args, kwargs)))

    with pytest.raises(ValueError, match="exactly one field:<id> marker"):
        ScoreboardCallback(
            base_url="https://scoreboard.example/test",
            token="secret",
            config_path=config_path,
            pipeline=pipeline,
            tracker=SimpleNamespace(),
            model=model,
            run_mode="test",
        )

    assert requests == []


def test_scoreboard_field_changes_all_canonical_hashes():
    callback = ScoreboardCallback.__new__(ScoreboardCallback)
    callback._run_mode = "test"
    callback._config_digest = "a" * 64
    callback._rerun_reason = None
    task = {
        "identity": "task",
        "weight_sha256": "b" * 64,
        "weight_display_name": "model",
        "wkv_mode": "fp32io16",
        "benchmark": "benchmark",
        "task_name": "benchmark",
        "field": "knowledge",
        "task_version": "0",
        "dataset": "dataset",
        "subset": "subset",
        "evaluation_splits": ["test"],
        "languages": ["english"],
        "tags": ["multiple-choice"],
    }
    other = {**task, "field": "math"}

    campaign = callback._campaign(task)
    other_campaign = callback._campaign(other)

    assert campaign["registry_sha256"] != other_campaign["registry_sha256"]
    assert campaign["run_key"] != other_campaign["run_key"]
    assert _sha256({"task": task}) != _sha256({"task": other})


def test_full_scoreboard_uses_scoreboard_v1_campaign_contract():
    callback = ScoreboardCallback.__new__(ScoreboardCallback)
    callback._run_mode = "full"
    callback._config_digest = "a" * 64
    callback._rerun_reason = None
    task = {
        "identity": f"{'b' * 64}:fp32io16:winogrande",
        "weight_sha256": "b" * 64,
        "weight_display_name": "RWKV7-g1i-1.5B-20260805-ctx16384",
        "wkv_mode": "fp32io16",
        "benchmark": "winogrande",
        "field": "reasoning",
        "task_name": "winogrande",
        "task_version": "0",
        "dataset": "allenai/winogrande",
        "subset": "winogrande_xl",
        "evaluation_splits": ["validation"],
        "languages": ["english"],
        "tags": ["commonsense", "reasoning"],
    }

    campaign = callback._campaign(task)

    assert set(campaign) == {
        "schema_version",
        "run_key",
        "source",
        "config_sha256",
        "registry_sha256",
        "contract_sha256",
        "configured_benchmarks",
        "resolved_benchmarks",
        "skipped_benchmarks",
        "expected_tasks",
        "rerun_reason",
    }
    assert campaign["schema_version"] == "scoreboard-v1"
    assert campaign["configured_benchmarks"] == ["winogrande"]
    assert campaign["expected_tasks"] == [task]


@pytest.mark.parametrize("model_name", ["RWKV7", "RWKV7-g1j", "RWKV7-1.5B"])
def test_scoreboard_rejects_model_name_without_generation_or_parameter_size(model_name):
    with pytest.raises(ValueError, match="model_name to contain generation and parameter size"):
        ScoreboardCallback._validate_model_name(model_name)


def test_scoreboard_accepts_exact_model_name():
    ScoreboardCallback._validate_model_name("RWKV7-g1j-1.5B-20260831-ctx16384")


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


def test_scoreboard_publication_keeps_only_evaluation_facts(tmp_path, monkeypatch):
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
            return Response({"status": "ready", "schema_version": "scoreboard-v1"})
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
                sample_level_fn=RWKVAvgAtK(
                    1,
                    SampleLevelMetric(
                        metric_name="accuracy",
                        sample_level_fn=ExactMatches(strip_strings=True),
                        category=SamplingMethod.GENERATIVE,
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
                    "module": "lighteval.tasks.tasks.gsm8k",
                    "docstring": {"languages": ["english"], "tags": ["math", "reasoning", "field:math"]},
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
        run_mode="test",
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
    assert set(publication) == {
        "schema_version",
        "campaign_id",
        "task",
        "result_files",
        "task_config",
        "environment",
        "sampling_config",
        "primary_metric",
        "metrics",
        "diagnostics",
        "samples",
    }
    assert "comparison" not in publication
    assert publication["sampling_config"]["temperature"] == 0.96
    assert publication["sampling_config"]["num_samples"] == 1
    assert publication["sampling_config"]["chat_template_kwargs"] == {
        "rwkv_prompt_template": "bot",
        "rwkv_generation_prompt": "open_think",
    }
    assert publication["task_config"]["k_metrics"] == "avg@1"
    assert publication["task"]["benchmark"] == "gsm8k"
    assert publication["task"]["task_name"] == "gsm8k"
    assert publication["task"]["field"] == "math"
    assert publication["task"]["weight_display_name"] == "RWKV7-g1h-7.2B-20260710-ctx10240"
    assert publication["task"]["weight_sha256"] == "a" * 64
    assert publication["task"]["wkv_mode"] == "fp32io16"
    assert publication["task"]["languages"] == ["english"]
    assert publication["task"]["tags"] == ["math", "reasoning"]
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
