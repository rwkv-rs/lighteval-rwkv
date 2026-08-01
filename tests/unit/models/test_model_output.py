from dataclasses import asdict

from lighteval.models.model_output import ModelResponse


def test_model_response_retains_http_termination_evidence_when_sliced() -> None:
    response = ModelResponse(
        text=["first", "second"],
        output_tokens=[[10, 0], [20, 21]],
        reasonings=[None, "because"],
        finish_reasons=["stop", "length"],
        stop_reasons=[0, None],
        terminal_token_ids=[0, 21],
    )

    selected = response[1]

    assert selected.reasonings == ["because"]
    assert selected.finish_reasons == ["length"]
    assert selected.stop_reasons == [None]
    assert selected.terminal_token_ids == [21]
    assert asdict(selected)["finish_reasons"] == ["length"]
