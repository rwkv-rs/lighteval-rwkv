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

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum

from lighteval.tasks.requests import Doc


class TaskPromptMode(str, Enum):
    """Evaluation-level task instruction override modes."""

    REPLACE = "replace"
    PREPEND = "prepend"
    APPEND = "append"
    INHERIT = "inherit"


@dataclass(frozen=True)
class TaskPromptIdentity:
    """Stable upstream and effective task-instruction provenance."""

    upstream_instruction: str
    effective_instruction: str
    mode: str
    digest: str

    def as_dict(self) -> dict[str, str]:
        """Return JSON-compatible prompt provenance."""
        return asdict(self)


def apply_task_prompt_override(
    doc: Doc,
    task_prompt: str,
    mode: str | TaskPromptMode,
) -> TaskPromptIdentity:
    """Apply one evaluation-level task prompt override before request creation.

    ``prepend`` and ``append`` concatenate the configured prompt verbatim so
    callers retain control over whitespace. The upstream instruction prefix is
    removed from ``Doc.query`` before the effective instruction is installed,
    matching :class:`PromptManager`'s upstream rendering behavior.
    """
    if not isinstance(task_prompt, str):
        raise TypeError("task_prompt must be a string")
    try:
        selected_mode = TaskPromptMode(mode)
    except ValueError as error:
        choices = ", ".join(item.value for item in TaskPromptMode)
        raise ValueError(f"task_prompt mode must be one of: {choices}") from error

    upstream = doc.instruction or ""
    match selected_mode:
        case TaskPromptMode.REPLACE:
            effective = task_prompt
        case TaskPromptMode.PREPEND:
            effective = task_prompt + upstream
        case TaskPromptMode.APPEND:
            effective = upstream + task_prompt
        case TaskPromptMode.INHERIT:
            effective = upstream

    if doc.instruction and doc.query.startswith(doc.instruction):
        query_without_instruction = doc.query[len(doc.instruction) :].strip()
        if selected_mode is not TaskPromptMode.REPLACE or query_without_instruction:
            doc.query = query_without_instruction

    canonical = json.dumps(
        {
            "effective_instruction": effective,
            "mode": selected_mode.value,
            "upstream_instruction": upstream,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    identity = TaskPromptIdentity(
        upstream_instruction=upstream,
        effective_instruction=effective,
        mode=selected_mode.value,
        digest=hashlib.sha256(canonical.encode()).hexdigest(),
    )
    doc.instruction = effective or None
    doc.specific = dict(doc.specific or {}, rwkv_task_prompt=identity.as_dict())
    return identity
