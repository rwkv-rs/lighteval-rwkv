import threading
from types import SimpleNamespace

import pytest

from lighteval.models.rwkv.http_model import PROMPT_TEMPLATES, RWKVHttpModel
from lighteval.models.rwkv.http_pool import Completion
from lighteval.tasks.prompt_manager import PromptManager


def _document(query, num_samples=1, generation_size=9000, stops=None):
    return SimpleNamespace(
        query=query,
        instruction=None,
        fewshot_samples=[],
        use_logits=False,
        num_samples=num_samples,
        generation_size=generation_size,
        stop_sequences=stops or [],
    )


@pytest.mark.parametrize(
    ("template", "prefix", "template_stop"),
    [
        ("bot", "\nBot✿", "✿"),
        ("assistant", "\n\nAssistant: ", "\nUser:"),
        ("function_calling", "\n### Assistant", "\n### User"),
    ],
)
def test_model_uses_prompt_template_stops_and_preserves_document_order(template, prefix, template_stop):
    calls = []
    lock = threading.Lock()

    class Pool:
        http_worker_limit = 5

        def complete(self, messages, parameters):
            with lock:
                calls.append((messages, parameters))
                token = len(calls)
            return Completion(
                text=messages[-1]["content"],
                reasoning="reasoning",
                finish_reason="stop",
                stop_reason=template_stop,
                terminal_token_id=0,
                prompt_text=f"rendered-{messages[-1]['content']}",
                prompt_token_ids=(1, 2),
                output_token_ids=(token,),
            )

    model = RWKVHttpModel.__new__(RWKVHttpModel)
    model.pool = Pool()
    model.prompt_manager = PromptManager(use_chat_template=True, tokenizer=None)
    model._prompt_template = template
    model._assistant_prefix = prefix
    model._template_stop = template_stop
    model._cot_mode = "open_think"
    model._generation_parameters = {
        "temperature": 0.96,
        "top_p": 0.76,
        "top_k": 32,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.1,
        "penalty_decay": 0.988,
    }
    model._cache = None

    responses = model.greedy_until(
        [_document("first", num_samples=2, stops=[template_stop, "task-stop"]), _document("second")]
    )

    assert [response.text for response in responses] == [["first", "first"], ["second"]]
    assert [response.reasonings for response in responses] == [
        ["reasoning", "reasoning"],
        ["reasoning"],
    ]
    assert [response.finish_reasons for response in responses] == [["stop", "stop"], ["stop"]]
    assert [response.stop_reasons for response in responses] == [
        [template_stop, template_stop],
        [template_stop],
    ]
    assert [response.terminal_token_ids for response in responses] == [[0, 0], [0]]
    assert len(calls) == 3
    first_parameters = calls[0][1]
    assert first_parameters == {
        **model._generation_parameters,
        "max_completion_tokens": 8192,
        "stop": [template_stop, "task-stop"],
        "chat_template_kwargs": {
            "rwkv_prompt_template": template,
            "rwkv_generation_prompt": "open_think",
        },
        "ignore_eos": False,
        "return_token_ids": True,
    }


def test_model_fake_think_parameters_and_provenance(tmp_path, monkeypatch):
    class Cache:
        def __init__(self, config):
            self.config = config

    class Pool:
        model_id = "served"

        def __init__(self):
            self.manifest = SimpleNamespace(max_model_len=10240)

        def close(self):
            pass

    manifest = SimpleNamespace(
        model_name="RWKV7-g1h-7.2B-20260710-ctx10240",
        served_model_name="served",
        model_revision="weight-sha",
        wkv_mode="fp32io16",
        vllm_version="0.11.0",
        max_model_len=10240,
        fingerprint="f" * 64,
    )
    monkeypatch.setattr("lighteval.models.rwkv.http_model.SampleCache", Cache)

    model = RWKVHttpModel(
        manifest=manifest,
        prompt_template="bot",
        cot_mode="fake_think",
        cache_dir=tmp_path,
        max_samples=3,
        pool=Pool(),
    )

    assert model._generation_parameters == {
        "temperature": 1.0,
        "top_p": 0.28,
        "top_k": 32,
    }
    assert model.config.model_name == manifest.model_name
    assert model.config.model_revision == "weight-sha"
    assert model.config.wkv_mode == "fp32io16"
    assert model.config.prompt_template == "bot"
    assert model.config.cot_mode == "fake_think"
    assert model.config.pool_fingerprint == "f" * 64
    assert model.config.max_samples == 3
    assert model.config.generation_parameters.max_new_tokens == 8192
    model.cleanup()


def test_model_rejects_generation_logits():
    model = RWKVHttpModel.__new__(RWKVHttpModel)
    model._cache = None
    document = _document("question")
    document.use_logits = True

    with pytest.raises(ValueError, match="generation logits"):
        model.greedy_until([document])


def test_prompt_template_contract_is_exact():
    assert PROMPT_TEMPLATES == {
        "bot": ("\nBot✿", "✿"),
        "assistant": ("\n\nAssistant: ", "\nUser:"),
        "function_calling": ("\n### Assistant", "\n### User"),
    }
