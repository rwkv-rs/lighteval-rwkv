from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.requests import Doc, SamplingMethod


def _run_pipeline(monkeypatch, docs, pipeline_parameters, metrics=()):
    original_config = SimpleNamespace(metrics=metrics)
    task = SimpleNamespace(
        full_name="fixture|0",
        config=original_config,
        metrics=metrics,
        sampling_methods=list(dict.fromkeys(metric.category for metric in metrics)),
        get_docs=lambda _max_samples: docs,
    )

    class FakeRegistry:
        def __init__(self, **_kwargs):
            pass

        def load_tasks(self):
            return {"fixture|0": task}

    monkeypatch.setattr("lighteval.pipeline.Registry", FakeRegistry)
    monkeypatch.setattr(
        "lighteval.pipeline.LightevalTask.load_datasets",
        lambda *_args, **_kwargs: None,
    )
    task_logger = Mock()
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.pipeline_parameters = pipeline_parameters
    pipeline._metric_options = {}
    pipeline.evaluation_tracker = SimpleNamespace(task_config_logger=task_logger)

    pipeline._init_tasks_and_requests("fixture|0")

    return pipeline, task, original_config, task_logger


def _doc(instruction="Upstream instruction: "):
    return Doc(
        query=f"{instruction}Question?",
        choices=["Answer"],
        gold_index=0,
        instruction=instruction,
        sampling_methods=[SamplingMethod.GENERATIVE],
    )


def test_pipeline_leaves_upstream_documents_and_config_untouched_by_default(monkeypatch):
    doc = _doc()
    parameters = PipelineParameters(launcher_type=ParallelismManager.NONE)

    pipeline, task, original_config, task_logger = _run_pipeline(monkeypatch, [doc], parameters)

    assert doc.query == "Upstream instruction: Question?"
    assert doc.instruction == "Upstream instruction: "
    assert doc.specific is None
    assert task.config is original_config
    assert pipeline.sampling_docs[SamplingMethod.GENERATIVE] == [doc]
    task_logger.log.assert_called_once_with(pipeline.tasks_dict)


@pytest.mark.parametrize(
    ("mode", "task_prompt", "expected_instruction"),
    [
        ("replace", "", None),
        ("prepend", "Evaluation: ", "Evaluation: Upstream instruction: "),
        ("append", " Evaluation", "Upstream instruction:  Evaluation"),
        ("inherit", "ignored", "Upstream instruction: "),
    ],
)
def test_pipeline_applies_explicit_task_prompt_modes(
    monkeypatch,
    mode,
    task_prompt,
    expected_instruction,
):
    doc = _doc()
    parameters = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        task_prompt=task_prompt,
        task_prompt_mode=mode,
    )

    pipeline, task, original_config, _ = _run_pipeline(monkeypatch, [doc], parameters)

    identity = doc.specific["rwkv_task_prompt"]
    assert doc.query == "Question?"
    assert doc.instruction == expected_instruction
    assert task.config is not original_config
    assert task.config.configured_task_prompt == task_prompt
    assert task.config.task_prompt_mode == mode
    assert task.config.task_prompt_digests == [identity["digest"]]
    assert not hasattr(task.config, "task_prompt_identities")
    assert identity["mode"] == mode
    assert pipeline.sampling_docs[SamplingMethod.GENERATIVE] == [doc]


def test_pipeline_task_prompt_digests_are_unique_and_sorted(monkeypatch):
    docs = [_doc("Instruction B: "), _doc("Instruction A: "), _doc("Instruction B: ")]
    parameters = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        task_prompt="Evaluation: ",
        task_prompt_mode="prepend",
    )

    _, task, _, _ = _run_pipeline(monkeypatch, docs, parameters)

    document_digests = {doc.specific["rwkv_task_prompt"]["digest"] for doc in docs}
    assert task.config.task_prompt_digests == sorted(document_digests)
    assert len(task.config.task_prompt_digests) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task_prompt": "Evaluation: "},
        {"task_prompt_mode": "replace"},
    ],
)
def test_pipeline_requires_task_prompt_and_mode_together(kwargs):
    with pytest.raises(ValueError, match="must be configured together"):
        PipelineParameters(launcher_type=ParallelismManager.NONE, **kwargs)


def test_pipeline_rejects_invalid_explicit_task_prompt_configuration():
    with pytest.raises(TypeError, match="task_prompt must be a string"):
        PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            task_prompt=1,
            task_prompt_mode="replace",
        )
    with pytest.raises(ValueError, match="task_prompt mode must be one of"):
        PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            task_prompt="Evaluation: ",
            task_prompt_mode="invalid",
        )
