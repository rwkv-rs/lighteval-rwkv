import asyncio
from types import SimpleNamespace

import pytest

from lighteval.models.rwkv.http_model import PROMPT_TEMPLATES, RWKVHttpModel
from lighteval.models.rwkv.http_pool import Completion
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.utils.cache_management import TaskID


def _document(query, num_samples=1, generation_size=9000, stops=None):
    return SimpleNamespace(
        id=query,
        task_name="task|0",
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

    class Pool:
        http_worker_limit = 5

        async def start(self):
            pass

        async def complete(self, messages, parameters):
            calls.append((messages, parameters))
            token = len(calls)
            await asyncio.sleep(0.001 * (4 - token))
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

    responses = asyncio.run(
        model.greedy_until(
            [_document("first", num_samples=2, stops=[template_stop, "task-stop"]), _document("second")]
        )
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
    assert [response.output_tokens for response in responses] == [[[1], [2]], [[3]]]
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
        "return_prompt_text": True,
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
        asyncio.run(model.greedy_until([document]))


def test_async_model_cache_preserves_document_order_and_skips_completed_requests():
    calls = []

    class Pool:
        async def start(self):
            pass

        async def complete(self, messages, _parameters):
            calls.append(messages[-1]["content"])
            return Completion(
                text=messages[-1]["content"],
                reasoning=None,
                finish_reason="stop",
                stop_reason=None,
                terminal_token_id=1,
                prompt_text=messages[-1]["content"],
                prompt_token_ids=(1,),
                output_token_ids=(2,),
            )

    class Cache:
        def __init__(self):
            self.results = {}

        def get_task_id(self, task_name, sampling_method):
            return TaskID(task_name, "hash", sampling_method)

        def get_samples_to_process_and_cache(self, docs, sampling_method):
            missing = [doc for doc in docs if doc.id not in self.results]
            cached = {self.get_task_id(doc.task_name, sampling_method) for doc in docs if doc.id in self.results}
            return missing, cached

        def cache_samples(self, docs, results, **_kwargs):
            self.results.update((doc.id, result) for doc, result in zip(docs, results))

        def get_samples_from_cache(self, docs, _task_ids, _sampling_method):
            return [self.results[doc.id] for doc in docs]

    model = RWKVHttpModel.__new__(RWKVHttpModel)
    model.pool = Pool()
    model.prompt_manager = PromptManager(use_chat_template=True, tokenizer=None)
    model._prompt_template = "bot"
    model._template_stop = "✿"
    model._cot_mode = "open_think"
    model._generation_parameters = {}
    model._cache = Cache()
    docs = [_document("first"), _document("second")]

    first = asyncio.run(model.greedy_until(docs))
    second = asyncio.run(model.greedy_until(list(reversed(docs))))

    assert calls == ["first", "second"]
    assert [response.text for response in first] == [["first"], ["second"]]
    assert [response.text for response in second] == [["second"], ["first"]]


def test_prompt_template_contract_is_exact():
    assert PROMPT_TEMPLATES == {
        "bot": ("\nBot✿", "✿"),
        "assistant": ("\n\nAssistant: ", "\nUser:"),
        "function_calling": ("\n### Assistant", "\n### User"),
    }
