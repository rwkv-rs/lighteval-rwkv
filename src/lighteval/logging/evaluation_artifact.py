# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gzip
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Literal


class EvaluationArtifactError(ValueError):
    """Raised when an evaluation artifact fails its publication contract."""


def canonical_json(value: object) -> bytes:
    """Serialize an artifact value to stable, finite JSON bytes."""
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise EvaluationArtifactError("evaluation artifact is not canonical JSON") from error


def canonical_gzip(value: object) -> bytes:
    """Compress canonical JSON reproducibly for an external publisher."""
    output = BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as archive:
        archive.write(canonical_json(value))
    return output.getvalue()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Commit bytes by same-directory fsync and atomic replacement.

    A failed call can leave only an unreferenced temporary file, never a
    partially replaced destination. Repeating an identical write is a no-op.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == content:
        return

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class EvaluationArtifactMember:
    """One byte-addressed member of an evaluation publication artifact."""

    path: str
    role: Literal["results", "details"]
    media_type: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "media_type": self.media_type,
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class EvaluationArtifact:
    """Content-addressed index for standard LightEval results and details.

    Creating this local artifact never publishes it. ``publication_requested``
    records an explicit caller decision while the initial status remains
    ``not_published`` until an external publisher returns its own receipt.
    """

    manifest_path: str
    members: tuple[EvaluationArtifactMember, ...]
    publication_requested: bool = False

    @property
    def results_path(self) -> str:
        return next(member.path for member in self.members if member.role == "results")

    @property
    def details_paths(self) -> tuple[str, ...]:
        return tuple(member.path for member in self.members if member.role == "details")

    def as_dict(self) -> dict[str, object]:
        """Return the stable payload written to ``manifest_path``."""
        return {
            "schema_version": "lighteval-evaluation-artifact-v1",
            "publication": {
                "requested": self.publication_requested,
                "status": "not_published",
            },
            "manifest_path": self.manifest_path,
            "members": [member.as_dict() for member in self.members],
        }

    def canonical_json(self) -> bytes:
        """Return the artifact as stable JSON bytes."""
        return canonical_json(self.as_dict())

    def canonical_gzip(self) -> bytes:
        """Return a reproducible gzip body for an explicit publisher."""
        return canonical_gzip(self.as_dict())

    @property
    def content_digest(self) -> str:
        """Return an identity transitively bound to every member's bytes."""
        return hashlib.sha256(self.canonical_json()).hexdigest()


def build_artifact_member(
    root: Path,
    path: Path,
    *,
    role: Literal["results", "details"],
    media_type: str,
) -> EvaluationArtifactMember:
    """Describe one already committed publication member."""
    root = Path(root).resolve()
    path = Path(path).resolve()
    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError as error:
        raise EvaluationArtifactError("artifact member path escapes output directory") from error
    _resolve_relative_path(root, relative_path)
    if not path.is_file():
        raise EvaluationArtifactError(f"artifact member does not exist: {relative_path}")
    content = path.read_bytes()
    return EvaluationArtifactMember(
        path=relative_path,
        role=role,
        media_type=media_type,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def load_evaluation_artifact(root: Path, manifest_path: str) -> EvaluationArtifact:
    """Load and fully verify an artifact without trusting its producer."""
    root = Path(root).resolve()
    resolved_manifest = _resolve_relative_path(root, manifest_path)
    try:
        manifest_bytes = resolved_manifest.read_bytes()
    except OSError as error:
        raise EvaluationArtifactError(f"artifact manifest is unavailable: {manifest_path}") from error

    payload = _strict_json_loads(manifest_bytes)
    artifact = _artifact_from_payload(payload, expected_manifest_path=manifest_path)
    if manifest_bytes != artifact.canonical_json() + b"\n":
        raise EvaluationArtifactError("artifact manifest is not canonical JSON")

    seen_paths: set[str] = set()
    for member in artifact.members:
        if member.path in seen_paths:
            raise EvaluationArtifactError(f"duplicate artifact member: {member.path}")
        seen_paths.add(member.path)
        member_path = _resolve_relative_path(root, member.path)
        try:
            content = member_path.read_bytes()
        except OSError as error:
            raise EvaluationArtifactError(f"artifact member is unavailable: {member.path}") from error
        if len(content) != member.size_bytes:
            raise EvaluationArtifactError(f"artifact member size mismatch: {member.path}")
        if hashlib.sha256(content).hexdigest() != member.sha256:
            raise EvaluationArtifactError(f"artifact member digest mismatch: {member.path}")

    return artifact


def _strict_json_loads(content: bytes) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise EvaluationArtifactError(f"non-finite JSON constant is forbidden: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationArtifactError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            content,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise EvaluationArtifactError("artifact manifest is not valid JSON") from error
    if not isinstance(payload, dict):
        raise EvaluationArtifactError("artifact manifest must be a JSON object")
    return payload


def _artifact_from_payload(payload: dict[str, object], *, expected_manifest_path: str) -> EvaluationArtifact:
    if set(payload) != {"schema_version", "publication", "manifest_path", "members"}:
        raise EvaluationArtifactError("artifact manifest has an unsupported schema")
    if payload["schema_version"] != "lighteval-evaluation-artifact-v1":
        raise EvaluationArtifactError("artifact manifest has an unsupported schema version")
    if payload["manifest_path"] != expected_manifest_path:
        raise EvaluationArtifactError("artifact manifest path does not match its location")
    _validate_relative_path(expected_manifest_path)

    publication_requested = _parse_publication(payload["publication"])
    members = _parse_members(payload["members"])
    return EvaluationArtifact(
        manifest_path=expected_manifest_path,
        members=tuple(members),
        publication_requested=publication_requested,
    )


def _parse_publication(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"requested", "status"}:
        raise EvaluationArtifactError("artifact publication fields are invalid")
    if not isinstance(value["requested"], bool) or value["status"] != "not_published":
        raise EvaluationArtifactError("artifact publication fields are invalid")
    return value["requested"]


def _parse_members(value: object) -> list[EvaluationArtifactMember]:
    if not isinstance(value, list):
        raise EvaluationArtifactError("artifact members must be a JSON array")
    members = [_parse_member(raw_member) for raw_member in value]
    if sum(member.role == "results" for member in members) != 1:
        raise EvaluationArtifactError("artifact must contain exactly one results member")
    expected_order = sorted(members, key=lambda member: (member.role != "results", member.path))
    if members != expected_order:
        raise EvaluationArtifactError("artifact members are not canonically ordered")
    return members


def _parse_member(value: object) -> EvaluationArtifactMember:
    expected_fields = {"path", "role", "media_type", "size_bytes", "sha256"}
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise EvaluationArtifactError("artifact member fields are invalid")

    path = value["path"]
    role = value["role"]
    media_type = value["media_type"]
    size_bytes = value["size_bytes"]
    sha256 = value["sha256"]
    if not isinstance(path, str):
        raise EvaluationArtifactError("artifact member path must be a string")
    _validate_relative_path(path)
    if role not in {"results", "details"}:
        raise EvaluationArtifactError("artifact member role is invalid")
    expected_media_type = "application/json" if role == "results" else "application/vnd.apache.parquet"
    if media_type != expected_media_type:
        raise EvaluationArtifactError("artifact member media type does not match its role")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        raise EvaluationArtifactError("artifact member size is invalid")
    if not _is_sha256(sha256):
        raise EvaluationArtifactError("artifact member digest is invalid")
    return EvaluationArtifactMember(
        path=path,
        role=role,
        media_type=media_type,
        size_bytes=size_bytes,
        sha256=sha256,
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_relative_path(path: str) -> None:
    if not path or "\\" in path:
        raise EvaluationArtifactError("artifact paths must be canonical relative POSIX paths")
    candidate = PurePosixPath(path)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != path
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or (candidate.parts and candidate.parts[0].endswith(":"))
    ):
        raise EvaluationArtifactError("artifact paths must be canonical relative POSIX paths")


def _resolve_relative_path(root: Path, path: str) -> Path:
    _validate_relative_path(path)
    resolved = (root / Path(*PurePosixPath(path).parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise EvaluationArtifactError("artifact path escapes output directory") from error
    return resolved
