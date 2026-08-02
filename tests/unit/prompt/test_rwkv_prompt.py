import hashlib
import json

import pytest

from lighteval.tasks.requests import Doc
from lighteval.tasks.rwkv_prompt import (
    RWKV_LIGHTEVAL_REPOSITORY,
    RWKV_LIGHTEVAL_UPSTREAM_COMMIT,
    RWKV_LIGHTEVAL_UPSTREAM_RELEASE,
    RWKV_LIGHTEVAL_UPSTREAM_RELEASE_COMMIT,
    TaskPromptMode,
    apply_task_prompt_override,
    render_naive_prompt,
)


@pytest.mark.parametrize(
    ("mode", "task_prompt", "expected"),
    [
        ("replace", "Campaign instruction: ", "Campaign instruction: "),
        ("prepend", "Campaign instruction: ", "Campaign instruction: Upstream instruction: "),
        ("append", " Campaign instruction", "Upstream instruction:  Campaign instruction"),
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
    assert identity.experimental is (mode != "inherit")
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


def test_default_empty_replace_clears_instruction():
    doc = Doc(
        query="Upstream instruction: Question?",
        choices=["Answer"],
        gold_index=0,
        instruction="Upstream instruction: ",
    )

    identity = apply_task_prompt_override(doc, "", TaskPromptMode.REPLACE)

    assert identity.effective_instruction == ""
    assert render_naive_prompt(doc) == "Question?"


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
    assert render_naive_prompt(doc) == (f"{task_prompt}\n\n" if task_prompt else "") + "Which explanation is correct?"


def test_naive_prompt_preserves_effective_instruction_fewshots_query_and_choices():
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
    )
    apply_task_prompt_override(doc, "Campaign instruction:\n", "replace")

    assert render_naive_prompt(doc) == (
        "Campaign instruction:\n\n\nExample?\n\nA. No\nB. Yes Yes\n\nQuestion?\n\nA. Left\nB. Right"
    )


def test_invalid_task_prompt_contract_fails_closed():
    doc = Doc(query="Question?", choices=["Answer"], gold_index=0)

    with pytest.raises(ValueError, match="task_prompt mode must be one of"):
        apply_task_prompt_override(doc, "", "invalid")
    with pytest.raises(TypeError, match="task_prompt must be a string"):
        apply_task_prompt_override(doc, None, "replace")


def test_fork_source_identity_is_immutable():
    assert RWKV_LIGHTEVAL_REPOSITORY == "https://github.com/rwkv-rs/lighteval-rwkv"
    assert RWKV_LIGHTEVAL_UPSTREAM_RELEASE == "v0.13.0"
    assert RWKV_LIGHTEVAL_UPSTREAM_RELEASE_COMMIT == "7c1cd62716b0a198a630c26d781430c54726cd02"
    assert RWKV_LIGHTEVAL_UPSTREAM_COMMIT == "64f4f5ae173626509fad6e477ca4ee56ebb26129"
