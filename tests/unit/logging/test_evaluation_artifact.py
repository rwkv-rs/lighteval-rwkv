import gzip
import json
import subprocess
import sys

import pytest

from lighteval.logging.evaluation_artifact import (
    EvaluationArtifact,
    EvaluationArtifactError,
    atomic_write_bytes,
    build_artifact_member,
    canonical_json,
    load_evaluation_artifact,
)


def _write_artifact(tmp_path, *, publication_requested=False):
    results_path = tmp_path / "results/model/results_stamp.json"
    details_path = tmp_path / "details/model/stamp/details_task_stamp.parquet"
    atomic_write_bytes(results_path, b'{"score":1}\n')
    atomic_write_bytes(details_path, b"PAR1fixture")
    artifact = EvaluationArtifact(
        manifest_path="artifacts/model/stamp/artifact.json",
        members=(
            build_artifact_member(
                tmp_path,
                results_path,
                role="results",
                media_type="application/json",
            ),
            build_artifact_member(
                tmp_path,
                details_path,
                role="details",
                media_type="application/vnd.apache.parquet",
            ),
        ),
        publication_requested=publication_requested,
    )
    manifest_path = tmp_path / artifact.manifest_path
    atomic_write_bytes(manifest_path, artifact.canonical_json() + b"\n")
    return artifact, manifest_path


def _rewrite_manifest(manifest_path, payload):
    atomic_write_bytes(manifest_path, canonical_json(payload) + b"\n")


def test_evaluation_artifact_round_trip_binds_member_bytes_and_is_idempotent(tmp_path):
    artifact, manifest_path = _write_artifact(tmp_path)
    manifest_inode = manifest_path.stat().st_ino

    loaded = load_evaluation_artifact(tmp_path, artifact.manifest_path)
    atomic_write_bytes(manifest_path, artifact.canonical_json() + b"\n")
    loaded_again = load_evaluation_artifact(tmp_path, artifact.manifest_path)

    assert loaded == artifact
    assert loaded_again.content_digest == artifact.content_digest
    assert manifest_path.stat().st_ino == manifest_inode
    assert gzip.decompress(artifact.canonical_gzip()) == artifact.canonical_json()
    assert artifact.results_path == "results/model/results_stamp.json"
    assert artifact.details_paths == ("details/model/stamp/details_task_stamp.parquet",)
    assert len(artifact.content_digest) == 64
    assert artifact.as_dict()["publication"] == {
        "requested": False,
        "status": "not_published",
    }


def test_evaluation_artifact_loads_in_fresh_process(tmp_path):
    artifact, _ = _write_artifact(tmp_path, publication_requested=True)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from lighteval.logging.evaluation_artifact import load_evaluation_artifact; "
                "import sys; "
                "print(load_evaluation_artifact(Path(sys.argv[1]), sys.argv[2]).content_digest)"
            ),
            str(tmp_path),
            artifact.manifest_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == artifact.content_digest


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "unavailable"),
        ("mutated", "digest mismatch"),
        ("truncated", "size mismatch"),
    ],
)
def test_evaluation_artifact_rejects_missing_or_changed_members(tmp_path, mutation, message):
    artifact, _ = _write_artifact(tmp_path)
    member_path = tmp_path / artifact.results_path
    if mutation == "missing":
        member_path.unlink()
    elif mutation == "mutated":
        member_path.write_bytes(b'{"score":2}\n')
    else:
        member_path.write_bytes(b"{")

    with pytest.raises(EvaluationArtifactError, match=message):
        load_evaluation_artifact(tmp_path, artifact.manifest_path)


def test_evaluation_artifact_rejects_path_escape(tmp_path):
    artifact, manifest_path = _write_artifact(tmp_path)
    payload = artifact.as_dict()
    payload["members"][0]["path"] = "../outside.json"
    _rewrite_manifest(manifest_path, payload)

    with pytest.raises(EvaluationArtifactError, match="relative POSIX"):
        load_evaluation_artifact(tmp_path, artifact.manifest_path)


def test_evaluation_artifact_rejects_invalid_publication_fields(tmp_path):
    artifact, manifest_path = _write_artifact(tmp_path)
    payload = artifact.as_dict()
    payload["publication"]["status"] = "published"
    _rewrite_manifest(manifest_path, payload)

    with pytest.raises(EvaluationArtifactError, match="publication fields"):
        load_evaluation_artifact(tmp_path, artifact.manifest_path)


def test_evaluation_artifact_rejects_noncanonical_manifest(tmp_path):
    artifact, manifest_path = _write_artifact(tmp_path)
    manifest_path.write_text(json.dumps(artifact.as_dict(), indent=2), encoding="utf-8")

    with pytest.raises(EvaluationArtifactError, match="canonical JSON"):
        load_evaluation_artifact(tmp_path, artifact.manifest_path)


def test_canonical_json_rejects_non_finite_metrics():
    with pytest.raises(EvaluationArtifactError, match="canonical JSON"):
        canonical_json({"metric": float("nan")})
