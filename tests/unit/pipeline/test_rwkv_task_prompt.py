import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lighteval.metrics.metrics import Metrics
from lighteval.models.model_output import ModelResponse
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters, _choice_answer
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks.med import med_qa, med_qa_prompt
from lighteval.tasks.tasks.mmlu_pro import mmlu_pro_prompt_function
from lighteval.tasks.tasks.olympiade_bench import olympiad_bench_prompt


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


def _choice_doc(gold_index=1):
    return Doc(
        query="Question?",
        choices=["one", "two", "three"],
        gold_index=gold_index,
        sampling_methods=[SamplingMethod.LOGPROBS],
        generation_size=1,
        stop_sequences=["\n"],
    )


def test_pipeline_converts_single_and_multiselect_logprob_choices(monkeypatch):
    single_choice = _choice_doc()
    multiselect = _choice_doc(gold_index=[0, 2])
    parameters = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        convert_logprob_choices_to_generation=True,
    )
    metric = Metrics.loglikelihood_acc.value

    pipeline, task, original_config, _ = _run_pipeline(
        monkeypatch,
        [single_choice, multiselect],
        parameters,
        metrics=(metric,),
    )

    assert pipeline.documents_dict[task.full_name] == [single_choice, multiselect]
    assert pipeline.sampling_docs[SamplingMethod.GENERATIVE] == [single_choice, multiselect]
    assert single_choice.sampling_methods == [SamplingMethod.GENERATIVE]
    assert multiselect.sampling_methods == [SamplingMethod.GENERATIVE]
    assert "A. one" in single_choice.query
    assert "Answer: <letter>" in single_choice.query
    assert "Answer: <letters separated by commas>" in multiselect.query
    assert single_choice.generation_size == 8192
    assert single_choice.stop_sequences == []
    assert single_choice.specific["rwkv_choice"] is True
    assert task.config is not original_config
    assert task.config.original_num_docs == 2
    assert task.config.effective_num_docs == 2
    assert len(task.metrics) == 1
    assert task.metrics[0].metric_name == metric.metric_name
    assert task.metrics[0].category == SamplingMethod.GENERATIVE

    pipeline.model = SimpleNamespace(config=SimpleNamespace(generation_parameters=SimpleNamespace(max_new_tokens=3)))
    response = ModelResponse(
        text=["<think>work</think>Answer: B"],
        output_tokens=[[1, 2]],
        finish_reasons=["stop"],
    )
    multiselect_response = ModelResponse(
        text=["<think>work</think>Answer: C, A"],
        output_tokens=[[3, 4]],
        finish_reasons=["stop"],
    )
    pipeline._post_process_outputs({SamplingMethod.GENERATIVE: [response, multiselect_response]})
    assert response.text_post_processed == ["two"]
    assert multiselect_response.text_post_processed == ['["one","three"]']
    assert task.metrics[0].compute_sample(doc=single_choice, model_response=response) == {"acc": 1}
    assert task.metrics[0].compute_sample(doc=multiselect, model_response=multiselect_response) == {"acc": 1}


def test_pipeline_does_not_extract_truncated_choice_answer(monkeypatch):
    doc = _choice_doc()
    parameters = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        convert_logprob_choices_to_generation=True,
    )
    pipeline, task, _, _ = _run_pipeline(
        monkeypatch,
        [doc],
        parameters,
        metrics=(Metrics.loglikelihood_acc.value,),
    )
    response = ModelResponse(
        text=["<think>The answer may be B, but reasoning continues forever"],
        output_tokens=[[1, 2, 3]],
        finish_reasons=["length"],
    )

    pipeline._post_process_outputs({SamplingMethod.GENERATIVE: [response]})

    assert response.text_post_processed == [""]
    assert task.metrics[0].compute_sample(doc=doc, model_response=response) == {"acc": 0}


@pytest.mark.parametrize("choices", [[" A", " B", " C", " D"], ["A", "B", "C", "D", "E"]])
def test_pipeline_converts_native_generative_letter_choices(monkeypatch, choices):
    doc = Doc(
        query="Question?\nA. one\nB. two\nAnswer:",
        choices=choices,
        gold_index=1,
        sampling_methods=[SamplingMethod.GENERATIVE],
        generation_size=5,
        stop_sequences=["\n"],
    )
    parameters = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        convert_logprob_choices_to_generation=True,
    )
    metric = Metrics.exact_match.value

    pipeline, task, original_config, _ = _run_pipeline(monkeypatch, [doc], parameters, metrics=(metric,))

    assert pipeline.sampling_docs[SamplingMethod.GENERATIVE] == [doc]
    assert doc.query.count('After reasoning, end with "Answer: <letter>".') == 1
    assert doc.generation_size == 8192
    assert doc.stop_sequences == []
    assert doc.specific["rwkv_choice"] is True
    assert task.config is not original_config
    assert task.metrics == (metric,)


def test_pipeline_does_not_convert_native_free_form_generation(monkeypatch):
    doc = Doc(
        query="Give two acceptable summaries.",
        choices=["first summary", "second summary"],
        gold_index=0,
        sampling_methods=[SamplingMethod.GENERATIVE],
        generation_size=32,
        stop_sequences=["\n"],
    )
    parameters = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        convert_logprob_choices_to_generation=True,
    )
    metric = Metrics.exact_match.value

    pipeline, task, original_config, _ = _run_pipeline(monkeypatch, [doc], parameters, metrics=(metric,))

    assert pipeline.sampling_docs[SamplingMethod.GENERATIVE] == [doc]
    assert doc.query == "Give two acceptable summaries."
    assert doc.generation_size == 32
    assert doc.stop_sequences == ["\n"]
    assert doc.specific is None
    assert task.config is original_config


def test_med_qa_uses_official_parquet_schema_and_does_not_repeat_letter_options(monkeypatch):
    doc = med_qa_prompt(
        {
            "question": "Question?",
            "options": [
                {"key": "A", "value": "one"},
                {"key": "B", "value": "two"},
                {"key": "C", "value": "three"},
                {"key": "D", "value": "four"},
                {"key": "E", "value": "five"},
            ],
            "answer_idx": "E",
        },
        "med_qa|0",
    )
    parameters = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        convert_logprob_choices_to_generation=True,
    )

    _run_pipeline(monkeypatch, [doc], parameters, metrics=(Metrics.loglikelihood_acc.value,))

    assert med_qa.hf_subset == "default"
    assert med_qa.hf_revision == "e04abdc0672c54547fa1dbe36cfefc000e4f2657"
    assert set(med_qa.hf_data_files) == {"train", "validation", "test"}
    assert doc.query.count("A. one") == 1
    assert "Give a letter answer among A, B, C, D or E." in doc.query


def test_mmlu_pro_exposes_exact_native_letter_choices_for_rwkv_conversion():
    doc = mmlu_pro_prompt_function(
        {
            "question": "Question?",
            "options": [f"option {letter}" for letter in "ABCDEFGHIJ"],
            "answer_index": 9,
        },
        "mmlu_pro|0",
    )

    assert doc.choices == list("ABCDEFGHIJ")
    assert "where LETTER is one of ABCDEFGHIJ" in doc.query
    assert "J: option J" in doc.query


def test_olympiad_bench_does_not_emit_an_empty_specific_struct():
    doc = olympiad_bench_prompt(
        {
            "subject": "Math",
            "language": "English",
            "unit": None,
            "is_multiple_answer": False,
            "answer_type": "Numerical",
            "final_answer": "1",
            "question": "Compute 1.",
        },
        "olympiad_bench:OE_TO_maths_en_COMP|0",
    )

    assert doc.specific is None


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("Answer: B", "two"),
        ("<think>x</think>Answer: <B>", "two"),
        ("<think>x</think>The correct answer is B. two.", "two"),
        ("<think>x</think>正确答案是 **B**。", "two"),
        ("<think>x</think><answer>B</answer>", "two"),
        ("<think>x</think>Answer: <choice B>", "two"),
        ("<think>x</think>Answer: <final>B</final>", "two"),
        ('<think>x</think>Answer: <span class="choice">B</span>', "two"),
        ("<think>x</think>Answer: <letter>B</letter>", "two"),
        ("<think>x</think><letter>C</letter>", "three"),
        ('<think>x</think>Answer: {"answer": "B"}', "two"),
        ("<think>x</think>Answer: <function=finish>\n<parameter=message>\nB\n</parameter>\n</function>", "two"),
        ("<think>x</think>C. three", "three"),
        ("<think>x</think><<B>>", "two"),
        ("<think>x</think><<C> three>", "three"),
        ("<think>x</think>因此，不会发生的反应是选项(D)。", "four"),
        ("<think>x</think>选项A直接给出了定义，其他选项过于宽泛（C）或不够具体（D）。", "one"),
        ("<think>x</think>This aligns with option B.", "two"),
        ("<think>x</think>This makes (B) the correct effect.", "two"),
        ("<think>x</think>Answer: A\nAnswer: B", "two"),
        ("<think>x</think>Answer: A\nI choose B", "two"),
        ("<think>x</think>C.", "three"),
        ("<think>x</think>The correct answer is A, as the evidence shows.", "one"),
        ("<think>x</think>因此，正确选项为B。\nAnswer: <", "two"),
        ("<think>x</think>Answer: C, A", '["one","three"]'),
        ("<think>x</think>Answer: <letter>", ""),
        ("<think>x</think>Answer: E", ""),
        ("<think>x</think>I am unsure", ""),
    ],
)
def test_choice_extraction_handles_dashboard_formats(completion, expected):
    assert _choice_answer(completion, ["one", "two", "three", "four"]) == expected


def test_choice_extraction_uses_unique_full_choice_text_as_last_fallback():
    choices = ["store bile", "produce digestive enzymes", "filter blood"]

    assert _choice_answer("<think>x</think>The gallbladder's function is to store bile.", choices) == "store bile"
    assert _choice_answer("<think>x</think>Both store bile and filter blood are discussed.", choices) == ""


def test_choice_extraction_rejects_conflicting_direct_option_references():
    assert _choice_answer("<think>x</think>选项A正确，选项B也正确。", ["one", "two", "three"]) == ""


def test_choice_extraction_uses_query_options_for_native_letter_choices():
    query = "Question?\nA。first option\nB。collective defense\nC。third option\nAnswer:"

    assert _choice_answer("<think>x</think>Collective defense", ["A", "B", "C"], query) == "B"
    assert _choice_answer("<think>x</think>The correct answer is <collective defense>.", ["A", "B", "C"], query) == "B"


def test_choice_extraction_normalizes_numeric_range_option_text():
    query = "Question?\nA. 14-28 weeks\nB. 3-9 weeks\nC. 28-37weeks\nD. 8-14weeks\nAnswer:"

    assert (
        _choice_answer(
            "<think>x</think>The dangerous period is specifically between 8 and 14 weeks of gestation.",
            ["A", "B", "C", "D"],
            query,
        )
        == "D"
    )


def test_choice_extraction_prefers_the_longest_unique_option_text():
    query = "Question?\nA. /\nB. //\nC. %\nAnswer:"

    assert _choice_answer("<think>x</think>Python floor division uses //.", ["A", "B", "C"], query) == "B"


def test_pipeline_leaves_logprob_choices_untouched_by_default(monkeypatch):
    doc = _choice_doc()
    original_query = doc.query
    original_generation_size = doc.generation_size
    original_stop_sequences = doc.stop_sequences
    metric = Metrics.loglikelihood_acc.value
    parameters = PipelineParameters(launcher_type=ParallelismManager.NONE)

    pipeline, task, original_config, _ = _run_pipeline(
        monkeypatch,
        [doc],
        parameters,
        metrics=(metric,),
    )

    assert doc.query == original_query
    assert doc.sampling_methods == [SamplingMethod.LOGPROBS]
    assert doc.generation_size == original_generation_size
    assert doc.stop_sequences == original_stop_sequences
    assert doc.specific is None
    assert pipeline.sampling_docs[SamplingMethod.LOGPROBS] == [doc]
    assert task.metrics == (metric,)
    assert task.config is original_config

    response = ModelResponse(
        text=["<think>work</think>Answer: B"],
        output_tokens=[[1]],
    )
    pipeline._post_process_outputs({SamplingMethod.GENERATIVE: [response]})
    assert response.text_post_processed == ["Answer: B"]


@pytest.mark.parametrize("enabled", [False, True])
def test_pipeline_logs_choice_conversion_setting_before_preparation(monkeypatch, caplog, enabled):
    doc = _choice_doc()
    parameters = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        convert_logprob_choices_to_generation=enabled,
    )
    caplog.set_level(logging.INFO, logger="lighteval.pipeline")

    _run_pipeline(
        monkeypatch,
        [doc],
        parameters,
        metrics=(Metrics.loglikelihood_acc.value,),
    )

    assert (
        f"convert_logprob_choices_to_generation: {enabled}, recommended value for RWKV models: True"
    ) in caplog.messages


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
