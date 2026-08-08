from dataclasses import asdict

from datasets import Dataset

from lighteval.models.model_output import ModelResponse


def test_model_response_retains_http_termination_evidence_when_sliced() -> None:
    response = ModelResponse(
        text=["first", "second"],
        output_tokens=[[10, 0], [20, 21]],
        reasonings=[None, "because"],
        finish_reasons=["stop", "length"],
        stop_reasons=["0", None],
        terminal_token_ids=[0, 21],
    )

    selected = response[1]

    assert selected.reasonings == ["because"]
    assert selected.finish_reasons == ["length"]
    assert selected.stop_reasons == [None]
    assert selected.terminal_token_ids == [21]
    assert asdict(selected)["finish_reasons"] == ["length"]


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
    assert response.finish_reasons == []


def test_termination_evidence_round_trips_through_dataset_parquet(tmp_path) -> None:
    response = ModelResponse(
        text=["first", "second"],
        output_tokens=[[10, 0], [20, 21]],
        finish_reasons=["stop", "length"],
        stop_reasons=["0", None],
        terminal_token_ids=[0, 21],
    )
    parquet_path = tmp_path / "responses.parquet"

    Dataset.from_list([asdict(response)]).to_parquet(parquet_path)
    restored = Dataset.from_parquet(str(parquet_path))[0]

    assert restored["finish_reasons"] == ["stop", "length"]
    assert restored["stop_reasons"] == ["0", None]
    assert restored["terminal_token_ids"] == [0, 21]
