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
from dataclasses import dataclass
from io import BytesIO


class EvaluationArtifactError(ValueError):
    """Raised when an evaluation artifact cannot be serialized canonically."""


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


@dataclass(frozen=True)
class EvaluationArtifact:
    """Canonical index for standard LightEval results and details.

    Creating this local artifact never publishes it. ``publication_requested``
    records an explicit caller decision while the initial status remains
    ``not_published`` until an external publisher returns its own receipt.
    """

    manifest_path: str
    results_path: str
    details_paths: tuple[str, ...]
    publication_requested: bool = False

    def as_dict(self) -> dict[str, object]:
        """Return the stable payload written to ``manifest_path``."""
        return {
            "schema_version": "lighteval-evaluation-artifact-v1",
            "publication": {
                "requested": self.publication_requested,
                "status": "not_published",
            },
            "manifest_path": self.manifest_path,
            "results_path": self.results_path,
            "details_paths": sorted(self.details_paths),
        }

    def canonical_json(self) -> bytes:
        """Return the artifact as stable JSON bytes."""
        return canonical_json(self.as_dict())

    def canonical_gzip(self) -> bytes:
        """Return a reproducible gzip body for an explicit publisher."""
        return canonical_gzip(self.as_dict())

    @property
    def content_digest(self) -> str:
        """Return the SHA-256 identity of the canonical artifact payload."""
        return hashlib.sha256(self.canonical_json()).hexdigest()
