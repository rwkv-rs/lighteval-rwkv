import gzip
import json

import pytest

from lighteval.logging.evaluation_artifact import (
    EvaluationArtifact,
    EvaluationArtifactError,
    canonical_json,
)


def test_evaluation_artifact_has_stable_json_digest_and_gzip():
    artifact = EvaluationArtifact(
        manifest_path="artifacts/model/stamp/artifact.json",
        results_path="results/model/results_stamp.json",
        details_paths=(
            "details/model/stamp/details_b_stamp.parquet",
            "details/model/stamp/details_a_stamp.parquet",
        ),
    )

    expected = canonical_json(artifact.as_dict())

    assert artifact.canonical_json() == expected
    assert artifact.canonical_gzip() == artifact.canonical_gzip()
    assert gzip.decompress(artifact.canonical_gzip()) == expected
    assert len(artifact.content_digest) == 64
    payload = json.loads(expected)
    assert payload["publication"]["status"] == "not_published"
    assert payload["details_paths"] == sorted(artifact.details_paths)


def test_canonical_json_rejects_non_finite_metrics():
    with pytest.raises(EvaluationArtifactError, match="canonical JSON"):
        canonical_json({"metric": float("nan")})
