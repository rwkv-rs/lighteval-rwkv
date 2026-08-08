from lighteval.models.model_output import ModelResponse


def test_model_response_retains_reasoning_when_sliced() -> None:
    response = ModelResponse(
        text=["first", "second"],
        output_tokens=[[10, 0], [20, 21]],
        reasonings=[None, "because"],
    )

    selected = response[1]

    assert selected.text == ["second"]
    assert selected.output_tokens == [[20, 21]]
    assert selected.reasonings == ["because"]


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
