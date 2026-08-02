import pytest

from lighteval.models.endpoints.litellm_model import prepare_openai_compatible_request
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc
from lighteval.tasks.rwkv_prompt import apply_task_prompt_override


def _document() -> Doc:
    fewshot = Doc(
        query="Example?\n\nA. No\nB. Yes",
        choices=["No", "Yes"],
        gold_index=1,
    )
    return Doc(
        query="Upstream instruction: Question?\n\nA. Left\nB. Right",
        choices=["Left", "Right"],
        gold_index=1,
        instruction="Upstream instruction: ",
        fewshot_samples=[fewshot],
    )


def test_naive_request_uses_v1_completions_and_preserves_complete_task_input():
    doc = _document()
    apply_task_prompt_override(doc, "Campaign instruction:\n", "replace")

    request = prepare_openai_compatible_request(
        doc,
        prompt_manager=PromptManager(use_chat_template=False),
        use_chat_template=False,
    )

    assert request.endpoint == "/v1/completions"
    assert request.as_payload() == {
        "prompt": (
            "Campaign instruction:\n\n\n"
            "Example?\n\nA. No\nB. Yes Yes\n\n"
            "Question?\n\nA. Left\nB. Right"
        )
    }


def test_chat_request_remains_the_default_baseline():
    doc = _document()

    request = prepare_openai_compatible_request(
        doc,
        prompt_manager=PromptManager(use_chat_template=True),
        use_chat_template=True,
    )

    assert request.endpoint == "/v1/chat/completions"
    assert request.as_payload() == {
        "messages": [
            {
                "role": "user",
                "content": "Upstream instruction: Example?\n\nA. No\nB. Yes",
            },
            {"role": "assistant", "content": "Yes"},
            {
                "role": "user",
                "content": "Question?\n\nA. Left\nB. Right",
            },
        ]
    }


def test_naive_request_rejects_model_system_prompt():
    with pytest.raises(ValueError, match="model system prompt"):
        prepare_openai_compatible_request(
            _document(),
            prompt_manager=PromptManager(
                use_chat_template=False,
                system_prompt="model instruction",
            ),
            use_chat_template=False,
        )
