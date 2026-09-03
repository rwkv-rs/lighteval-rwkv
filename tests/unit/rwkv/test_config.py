from pathlib import Path
from types import SimpleNamespace

import pytest

from lighteval.main_rwkv import ConfigError, RWKVEvaluationConfig, resolve_benchmarks
from lighteval.tasks.requests import SamplingMethod


DEFAULT_BENCHMARKS = {
    "mmlu",
    "mmlu_pro",
    "mmlu_redux_2",
    "gpqa:diamond",
    "gpqa:main",
    "arc:challenge",
    "arc:easy",
    "hellaswag",
    "bigbench_hard",
    "agieval",
    "truthfulqa:mc",
    "winogrande",
    "openbookqa",
    "commonsenseqa",
    "ceval_zho_mcf",
    "med_qa",
    "med_mcqa",
    "gsm8k",
    "gsm_plus",
    "asdiv",
    "mathqa",
    "arithmetic",
    "math_500",
    "math",
    "aime24",
    "aime25",
    "aimo_progress_prize_1",
    "olympiad_bench",
    "lcb:codegeneration",
    "ifeval",
    "ifbench_test",
    "ifbench_multiturn",
}
EXCLUDED_BENCHMARKS = {
    "mmlu_sr_question_answer",
    "kmmlu",
    "minerva_math",
    "svamp",
    "beyond_aime",
    "brumo25",
    "hmmt_feb_2025",
    "math_odyssey",
    "comp_math_24_25",
    "gaokao_2023_english",
    "answer_judge",
    "simpleqa_verified",
    "humaneval",
    "humaneval_cn",
    "humaneval_fix",
    "humaneval_plus",
    "mbpp",
    "mbpp_plus",
}


def test_default_config_contains_only_the_32_native_selectors(tmp_path):
    manifest = tmp_path / "pool.json"
    manifest.write_text("{}", encoding="utf-8")

    config = RWKVEvaluationConfig.read(
        Path("configs/eval/lighteval.toml"),
        env={"RWKV_EVAL_POOL_MANIFEST": str(manifest)},
    )

    assert len(config.benchmarks) == 32
    assert set(config.benchmarks) == DEFAULT_BENCHMARKS
    assert set(config.benchmarks).isdisjoint(EXCLUDED_BENCHMARKS)
    resolved = resolve_benchmarks(config.benchmarks)
    assert resolved.selector_count == 32
    assert len(resolved.leaf_tasks) == 247


def test_config_rejects_unknown_fields_and_duplicate_selectors(tmp_path):
    manifest = tmp_path / "pool.json"
    manifest.write_text("{}", encoding="utf-8")
    config = tmp_path / "eval.toml"
    config.write_text(
        f"""
schema_version = 1
pool_manifest = "{manifest}"
output_dir = "results"
prompt_template = "bot"
cot_mode = "open_think"
benchmarks = ["gsm8k", "gsm8k"]
publish = true
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unknown RWKV eval config fields: publish"):
        RWKVEvaluationConfig.read(config)

    config.write_text(config.read_text(encoding="utf-8").replace("publish = true\n", ""), encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate benchmark selectors: gsm8k"):
        RWKVEvaluationConfig.read(config)


def test_config_requires_referenced_manifest_environment(tmp_path):
    config = tmp_path / "eval.toml"
    config.write_text(
        """
schema_version = 1
pool_manifest = "${RWKV_EVAL_POOL_MANIFEST}"
output_dir = "results"
prompt_template = "bot"
cot_mode = "open_think"
benchmarks = ["gsm8k"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="RWKV_EVAL_POOL_MANIFEST"):
        RWKVEvaluationConfig.read(config, env={})


def test_resolver_reports_all_missing_selectors(monkeypatch):
    metric = SimpleNamespace(category=SamplingMethod.GENERATIVE)

    class Registry:
        def __init__(self, **_kwargs):
            self._task_registry = {
                "present": SimpleNamespace(metrics=(metric,)),
            }

        def _expand_task_definition(self, selector):
            return [selector]

    monkeypatch.setattr("lighteval.tasks.registry.Registry", Registry)

    with pytest.raises(ConfigError, match="missing_a, missing_b.*another framework"):
        resolve_benchmarks(("present", "missing_a", "missing_b"))


def test_resolver_expands_supersets_and_rejects_perplexity(monkeypatch):
    generative = SimpleNamespace(category=SamplingMethod.GENERATIVE)
    perplexity = SimpleNamespace(category=SamplingMethod.PERPLEXITY)

    class Registry:
        def __init__(self, **_kwargs):
            self._task_registry = {
                "suite:one": SimpleNamespace(metrics=(generative,)),
                "suite:two": SimpleNamespace(metrics=(generative,)),
                "ppl": SimpleNamespace(metrics=(perplexity,)),
            }

        def _expand_task_definition(self, selector):
            return ["suite:one", "suite:two"] if selector == "suite" else [selector]

    monkeypatch.setattr("lighteval.tasks.registry.Registry", Registry)

    resolved = resolve_benchmarks(("suite",))
    assert resolved.selector_count == 1
    assert resolved.leaf_tasks == ("suite:one", "suite:two")

    with pytest.raises(ConfigError, match="incompatible.*ppl"):
        resolve_benchmarks(("ppl",))
