import hashlib
import json

import pytest

from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc
from lighteval.tasks.rwkv_prompt import (
    TaskPromptMode,
    apply_task_prompt_override,
)


def _render_plain_prompt(doc: Doc) -> str:
    return PromptManager(use_chat_template=False).prepare_prompt(doc)


@pytest.mark.parametrize(
    ("mode", "task_prompt", "expected"),
    [
        ("replace", "Evaluation instruction: ", "Evaluation instruction: "),
        ("prepend", "Evaluation instruction: ", "Evaluation instruction: Upstream instruction: "),
        ("append", " Evaluation instruction", "Upstream instruction:  Evaluation instruction"),
        ("inherit", "ignored", "Upstream instruction: "),
    ],
)
def test_task_prompt_modes(mode: str, task_prompt: str, expected: str):
    doc = Doc(
        query="Upstream instruction: Question?",
        choices=["Answer"],
        gold_index=0,
        instruction="Upstream instruction: ",
    )

    identity = apply_task_prompt_override(doc, task_prompt, mode)

    assert doc.query == "Question?"
    assert doc.instruction == expected
    assert identity.upstream_instruction == "Upstream instruction: "
    assert identity.effective_instruction == expected
    assert identity.mode == mode
    assert doc.specific == {"rwkv_task_prompt": identity.as_dict()}
    canonical = json.dumps(
        {
            "effective_instruction": expected,
            "mode": mode,
            "upstream_instruction": "Upstream instruction: ",
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert identity.digest == hashlib.sha256(canonical.encode()).hexdigest()


def test_explicit_empty_replace_clears_instruction():
    doc = Doc(
        query="Upstream instruction: Question?",
        choices=["Answer"],
        gold_index=0,
        instruction="Upstream instruction: ",
    )

    identity = apply_task_prompt_override(doc, "", TaskPromptMode.REPLACE)

    assert identity.effective_instruction == ""
    assert _render_plain_prompt(doc) == "Question?"


@pytest.mark.parametrize("task_prompt", ["", "Answer with one choice only.\n"])
def test_replace_preserves_gpqa_style_query_when_instruction_is_complete_task_input(
    task_prompt: str,
):
    doc = Doc(
        query="Which explanation is correct?",
        choices=["first", "second", "third", "fourth"],
        gold_index=2,
        instruction="Which explanation is correct?",
    )

    identity = apply_task_prompt_override(doc, task_prompt, TaskPromptMode.REPLACE)

    assert identity.upstream_instruction == "Which explanation is correct?"
    assert doc.instruction == (task_prompt or None)
    assert doc.query == "Which explanation is correct?"
    assert doc.choices == ["first", "second", "third", "fourth"]
    assert _render_plain_prompt(doc) == (f"{task_prompt}\n\n" if task_prompt else "") + "Which explanation is correct?"


@pytest.mark.parametrize(
    ("mode", "task_prompt"),
    [
        ("prepend", "Answer with one choice only.\n"),
        ("append", "\nAnswer with one choice only."),
        ("inherit", "ignored"),
    ],
)
def test_non_replace_modes_keep_gpqa_task_content(mode: str, task_prompt: str):
    question = "Which explanation is correct?"
    doc = Doc(
        query=question,
        choices=["first", "second", "third", "fourth"],
        gold_index=2,
        instruction=question,
    )

    apply_task_prompt_override(doc, task_prompt, mode)
    rendered = _render_plain_prompt(doc)

    assert question in rendered
    assert rendered.count(question) == 1
    assert doc.choices == ["first", "second", "third", "fourth"]


def test_plain_prompt_preserves_effective_instruction_fewshots_query_and_choices():
    fewshot = Doc(
        query="Example?\n\nA. No\nB. Yes",
        choices=["No", "Yes"],
        gold_index=1,
    )
    doc = Doc(
        query="Question?\n\nA. Left\nB. Right",
        choices=["Left", "Right"],
        gold_index=1,
        instruction="Upstream instruction: ",
        fewshot_samples=[fewshot],
        stop_sequences=["<END>"],
    )
    apply_task_prompt_override(doc, "Evaluation instruction:\n", "replace")

    assert _render_plain_prompt(doc) == (
        "Evaluation instruction:\n\n\nExample?\n\nA. No\nB. Yes Yes\n\nQuestion?\n\nA. Left\nB. Right"
    )
    assert doc.stop_sequences == ["<END>"]


def test_invalid_task_prompt_contract_fails_closed():
    doc = Doc(query="Question?", choices=["Answer"], gold_index=0)

    with pytest.raises(ValueError, match="task_prompt mode must be one of"):
        apply_task_prompt_override(doc, "", "invalid")
    with pytest.raises(TypeError, match="task_prompt must be a string"):
        apply_task_prompt_override(doc, None, "replace")
