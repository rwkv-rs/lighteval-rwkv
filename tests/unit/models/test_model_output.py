import pytest

from lighteval.models.model_output import ModelResponse


def test_model_response_retains_reasoning_when_sliced() -> None:
    response = ModelResponse(
        text=["first", "second"],
        output_tokens=[[10, 0], [20, 21]],
        text_post_processed=["processed first", "processed second"],
        reasonings=[None, "because"],
        finish_reasons=["stop", "length"],
        stop_reasons=["✿", None],
        terminal_token_ids=[0, 21],
    )

    selected = response[1]

    assert selected.text == ["second"]
    assert selected.output_tokens == [[20, 21]]
    assert selected.text_post_processed == ["processed second"]
    assert selected.reasonings == ["because"]
    assert selected.finish_reasons == ["length"]
    assert selected.stop_reasons == [None]
    assert selected.terminal_token_ids == [21]


def test_model_response_preserves_existing_positional_arguments() -> None:
    response = ModelResponse(
        "prompt",
        [1],
        ["answer"],
        [[2]],
        ["processed"],
        ["reasoning"],
        [-0.5],
        [True],
        [[1.0]],
        [-1.0],
        3,
        4,
    )

    assert response.logprobs == [-0.5]
    assert response.argmax_logits_eq_gold == [True]
    assert response.truncated_tokens_count == 3
    assert response.padded_tokens_count == 4


def test_model_response_rejects_misaligned_termination_metadata() -> None:
    for field_name in ("finish_reasons", "stop_reasons", "terminal_token_ids"):
        with pytest.raises(ValueError, match=field_name):
            ModelResponse(text=["first", "second"], **{field_name: [None]})
