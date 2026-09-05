from types import SimpleNamespace

import typer
from typer.testing import CliRunner

import lighteval.main_rwkv as main_rwkv


def _app():
    app = typer.Typer()
    app.command()(main_rwkv.rwkv)
    return app


def _config(tmp_path, *, run_mode="full", max_samples=None):
    return main_rwkv.RWKVEvaluationConfig(
        run_mode=run_mode,
        max_samples=max_samples,
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


def test_test_run_uses_configured_limit_and_standard_saving(tmp_path, monkeypatch):
    config = _config(tmp_path, run_mode="test", max_samples=10)
    pool = SimpleNamespace(close=lambda: None)
    resolved = main_rwkv.ResolvedBenchmarks(
        selector_count=2,
        leaf_tasks=("gsm8k", "ifeval"),
        selector_tasks=(("gsm8k", ("gsm8k",)), ("ifeval", ("ifeval",))),
    )
    calls = []
    monkeypatch.delenv("SCOREBOARD_API_BASE_URL_TEST", raising=False)
    monkeypatch.delenv("SCOREBOARD_PUBLICATION_TOKEN_TEST", raising=False)

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

    result = CliRunner().invoke(_app(), ["--config", str(tmp_path / "eval.toml")])

    assert result.exit_code == 0, result.output
    assert "run mode: test (max_samples=10)" in result.output
    model_call = next(call for call in calls if call[0] == "model")
    assert model_call[1]["max_samples"] == 10
    pipeline_call = next(call for call in calls if call[0] == "pipeline")
    assert pipeline_call[1]["tasks"] == "gsm8k,ifeval"
    assert pipeline_call[1]["pipeline_parameters"].max_samples == 10
    assert pipeline_call[1]["pipeline_parameters"].load_tasks_multilingual is True
    assert not hasattr(pipeline_call[1]["pipeline_parameters"], "convert_logprob_choices_to_generation")
    assert pipeline_call[1]["selector_tasks"] == {"gsm8k": ("gsm8k",), "ifeval": ("ifeval",)}
    assert pipeline_call[1]["task_max_samples"] == {"gsm8k": 10, "ifeval": 10}
    assert ("evaluate",) in calls
    assert ("show",) in calls
    assert ("save",) in calls
    assert calls[-1] == ("cleanup",)
