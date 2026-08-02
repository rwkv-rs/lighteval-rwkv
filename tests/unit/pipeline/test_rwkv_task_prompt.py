from types import SimpleNamespace
from unittest.mock import Mock

from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.requests import Doc, SamplingMethod


def test_pipeline_applies_empty_task_prompt_and_persists_provenance(monkeypatch):
    doc = Doc(
        query="Upstream instruction: Question?",
        choices=["Answer"],
        gold_index=0,
        instruction="Upstream instruction: ",
        sampling_methods=[SamplingMethod.GENERATIVE],
    )
    task = SimpleNamespace(
        full_name="fixture|0",
        config=SimpleNamespace(
            configured_task_prompt=None,
            task_prompt_mode=None,
            task_prompt_digests=[],
            experimental_identity=False,
        ),
        get_docs=lambda _max_samples: [doc],
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
    pipeline.pipeline_parameters = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=None,
    )
    pipeline._metric_options = {}
    pipeline.evaluation_tracker = SimpleNamespace(task_config_logger=task_logger)

    pipeline._init_tasks_and_requests("fixture|0")

    assert doc.query == "Question?"
    assert doc.instruction is None
    assert doc.specific["rwkv_task_prompt"] == {
        "original_instruction": "Upstream instruction: ",
        "effective_instruction": "",
        "mode": "replace",
        "digest": task.config.task_prompt_digests[0],
        "experimental": True,
    }
    assert task.config.configured_task_prompt == ""
    assert task.config.task_prompt_mode == "replace"
    assert task.config.experimental_identity is True
    assert pipeline.sampling_docs[SamplingMethod.GENERATIVE] == [doc]
    task_logger.log.assert_called_once_with(pipeline.tasks_dict)


def test_pipeline_preserves_upstream_prompt_only_with_explicit_inherit(monkeypatch):
    doc = Doc(
        query="Upstream instruction: Question?",
        choices=["Answer"],
        gold_index=0,
        instruction="Upstream instruction: ",
        sampling_methods=[SamplingMethod.GENERATIVE],
    )
    task = SimpleNamespace(
        full_name="fixture|0",
        config=SimpleNamespace(),
        get_docs=lambda _max_samples: [doc],
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
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.pipeline_parameters = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        task_prompt_mode="inherit",
    )
    pipeline._metric_options = {}
    pipeline.evaluation_tracker = SimpleNamespace(task_config_logger=SimpleNamespace(log=lambda _tasks: None))

    pipeline._init_tasks_and_requests("fixture|0")

    assert doc.query == "Question?"
    assert doc.instruction == "Upstream instruction: "
    assert doc.specific["rwkv_task_prompt"] == {
        "original_instruction": "Upstream instruction: ",
        "effective_instruction": "Upstream instruction: ",
        "mode": "inherit",
        "digest": task.config.task_prompt_digests[0],
        "experimental": False,
    }
    assert task.config.task_prompt_mode == "inherit"
    assert task.config.experimental_identity is False
