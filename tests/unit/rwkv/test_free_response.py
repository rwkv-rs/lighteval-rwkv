import pytest

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.tasks.rwkv_free_response import RWKVFreeResponseMatch, is_rwkv_free_response


@pytest.mark.parametrize(
    ("task_full_name", "expected"),
    [
        ("arithmetic:add_or_sub|0", True),
        ("math:algebra|0", True),
        ("asdiv|0", True),
        ("gsm_plus|0", True),
        ("aimo_progress_prize_1|0", True),
        ("gsm8k|0", False),
        ("mmlu|0", False),
        ("mathqa|0", False),
    ],
)
def test_is_rwkv_free_response_gates_on_task_name(task_full_name, expected):
    assert is_rwkv_free_response(task_full_name) == expected


@pytest.mark.parametrize(
    ("gold", "prediction", "extracted"),
    [
        ("-371", "230 − 601 = −371\n\nThis means that 601 is 371 more than 230.", "-371"),
        (r"\frac{1}{2}", r"Thus $\boxed{\frac{1}{2}}$.", "1/2"),
        ("19", r"The area condition gives the result $\boxed{19}$.", "19"),
        ("19", "The final answer is 19 cm^2.", "19"),
        ("19", "Final answer: 019", "19"),
        ("19", "x = 20\n\nThe value is 19", "19"),
        ("0", "Final answer: 000", "0"),
        ("999", "Final answer: 999", "999"),
    ],
)
def test_rwkv_free_response_match_extracts_and_scores_final_answer(gold, prediction, extracted):
    metric = RWKVFreeResponseMatch()
    doc = Doc(query="question", choices=[gold], gold_index=0)
    response = ModelResponse(text=[prediction])

    assert metric.compute(doc, response) == 1.0
    assert metric.extract_answer(doc, response) == extracted


def test_rwkv_free_response_match_scores_wrong_answers_as_zero():
    metric = RWKVFreeResponseMatch()
    doc = Doc(query="question", choices=["19"], gold_index=0)
    response = ModelResponse(text=["The final answer is 20"])

    assert metric.compute(doc, response) == 0.0
    assert doc.specific["extracted_predictions"] == ["20"]


def test_rwkv_free_response_match_handles_unparsable_completions():
    metric = RWKVFreeResponseMatch()
    doc = Doc(query="question", choices=["19"], gold_index=0)
    response = ModelResponse(text=["I am unsure"])

    assert metric.compute(doc, response) == 0.0
    assert metric.extract_answer(doc, response) == ""


def test_rwkv_free_response_match_reuses_stored_prediction_without_reparsing():
    metric = RWKVFreeResponseMatch()
    doc = Doc(query="question", choices=["19"], gold_index=0, specific={"extracted_predictions": ["19"]})

    assert metric.extract_answer(doc, ModelResponse(text=["irrelevant text"])) == "19"
