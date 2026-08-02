import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from lighteval.models.endpoints.litellm_model import (
    LiteLLMClient,
    OpenAICompatibleRequest,
    prepare_openai_compatible_request,
)
from lighteval.models.model_input import GenerationParameters
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
    assert request.tokenization == "server"
    assert request.as_payload() == {
        "prompt": ("Campaign instruction:\n\n\nExample?\n\nA. No\nB. Yes Yes\n\nQuestion?\n\nA. Left\nB. Right")
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


@pytest.mark.parametrize(
    "openai_request",
    [
        OpenAICompatibleRequest(endpoint="/v1/completions", model_input=[{"role": "user", "content": "x"}]),
        OpenAICompatibleRequest(endpoint="/v1/chat/completions", model_input="x"),
    ],
)
def test_endpoint_payload_cannot_mix_text_and_chat_input(openai_request):
    with pytest.raises(TypeError, match="text completions|chat completions"):
        openai_request.as_payload()


class _OpenAICompatibleServer:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers["content-length"])
                owner.requests.append((self.path, json.loads(self.rfile.read(content_length))))
                if not owner.responses:
                    raise AssertionError("unexpected endpoint request")
                status, payload = owner.responses.pop(0)
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


def _completion_response(*choices):
    return {
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 1,
        "model": "test-model",
        "choices": list(choices),
        "usage": {"prompt_tokens": 1, "completion_tokens": len(choices), "total_tokens": len(choices) + 1},
    }


def _choice(text, index=0, *, finish_reason="stop", stop_reason="END", token_id=7, logprob=-0.25):
    return {
        "text": text,
        "index": index,
        "logprobs": {
            "tokens": [text],
            "token_logprobs": [logprob],
            "top_logprobs": [{text: logprob}],
            "text_offset": [0],
        },
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "token_ids": [token_id],
    }


def _transport_client(base_url, *, retries=1, generation_parameters=None):
    client = object.__new__(LiteLLMClient)
    client.model = "openai/test-model"
    client.provider = "openai"
    client.base_url = base_url
    client.api_key = "test-key"
    client.generation_parameters = generation_parameters or GenerationParameters()
    client.concurrent_requests = 1
    client.API_MAX_RETRY = retries
    client.API_RETRY_SLEEP = 0
    client.API_RETRY_MULTIPLIER = 1
    client.timeout = 2
    client.use_chat_template = False
    client.prompt_manager = PromptManager(use_chat_template=False)
    client._cache = None
    return client


def _call_text_completion(client, *, num_samples=1, return_logits=True):
    return client._LiteLLMClient__call_api(
        OpenAICompatibleRequest(endpoint="/v1/completions", model_input="transport prompt"),
        return_logits,
        17,
        num_samples,
        ["END"],
    )


def test_text_completion_uses_real_litellm_transport_and_preserves_payload_and_evidence():
    response = _completion_response(
        _choice("first", 0, token_id=11, logprob=-0.1),
        _choice("second", 1, finish_reason="length", stop_reason=None, token_id=12, logprob=-0.2),
    )
    parameters = GenerationParameters(
        temperature=0.4,
        top_p=0.8,
        seed=123,
        stop_tokens=["MODEL_DEFAULT_STOP"],
        frequency_penalty=0.2,
        presence_penalty=0.3,
    )
    with _OpenAICompatibleServer([(200, response)]) as server:
        client = _transport_client(server.base_url, generation_parameters=parameters)
        model_responses = client.greedy_until(
            [
                Doc(
                    query="transport prompt",
                    choices=["answer"],
                    gold_index=0,
                    generation_size=17,
                    stop_sequences=["END"],
                    use_logits=True,
                    num_samples=2,
                )
            ]
        )

    assert len(server.requests) == 1
    path, payload = server.requests[0]
    assert path == "/v1/completions"
    assert payload == {
        "model": "test-model",
        "prompt": "transport prompt",
        "frequency_penalty": 0.2,
        "logprobs": 1,
        "max_tokens": 17,
        "n": 2,
        "presence_penalty": 0.3,
        "seed": 123,
        "stop": ["END"],
        "temperature": 0.4,
        "top_p": 0.8,
    }
    assert model_responses[0].text == ["first", "second"]
    assert model_responses[0].token_logprobs == [[-0.1], [-0.2]]
    assert model_responses[0].finish_reasons == ["stop", "length"]
    assert model_responses[0].stop_reasons == ["END", None]
    assert model_responses[0].terminal_token_ids == [11, 12]


def test_greedy_until_isolates_generation_groups_without_duplicate_requests_and_restores_order(monkeypatch):
    client = _transport_client(
        "http://unused.invalid/v1",
        generation_parameters=GenerationParameters(temperature=0.2),
    )
    calls = []

    def fake_parallel(requests, return_logits, max_new_tokens, num_samples, stop_sequence):
        calls.append(
            {
                "prompts": [request.model_input for request in requests],
                "return_logits": return_logits,
                "max_new_tokens": max_new_tokens,
                "num_samples": num_samples,
                "stop_sequence": stop_sequence,
            }
        )
        return [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text=f"answer:{request.model_input}:{sample_index}",
                        reasoning_content=None,
                        logprobs={"token_logprobs": [-0.5]} if return_logits else None,
                        finish_reason="stop",
                        stop_reason=stop_sequence[0],
                        token_ids=[sample_index],
                    )
                    for sample_index in range(num_samples)
                ]
            )
            for request in requests
        ]

    monkeypatch.setattr(client, "_LiteLLMClient__call_api_parallel", fake_parallel)
    docs = [
        Doc(
            query="first",
            choices=["x"],
            gold_index=0,
            generation_size=3,
            stop_sequences=["GROUP_A"],
            use_logits=False,
            num_samples=1,
        ),
        Doc(
            query="second",
            choices=["x"],
            gold_index=0,
            generation_size=7,
            stop_sequences=["GROUP_B"],
            use_logits=True,
            num_samples=2,
        ),
        Doc(
            query="third is longer",
            choices=["x"],
            gold_index=0,
            generation_size=3,
            stop_sequences=["GROUP_A"],
            use_logits=False,
            num_samples=1,
        ),
    ]

    responses = client.greedy_until(docs)

    assert {
        (call["return_logits"], call["max_new_tokens"], call["num_samples"], tuple(call["stop_sequence"]))
        for call in calls
    } == {
        (False, 3, 1, ("GROUP_A",)),
        (True, 7, 2, ("GROUP_B",)),
    }
    requested_prompts = [prompt for call in calls for prompt in call["prompts"]]
    assert sorted(requested_prompts) == sorted(doc.query for doc in docs)
    assert len(requested_prompts) == len(set(requested_prompts)) == len(docs)
    assert [response.input for response in responses] == [doc.query for doc in docs]
    assert [len(response.text) for response in responses] == [1, 2, 1]


def test_greedy_until_rejects_group_response_count_mismatch(monkeypatch):
    client = _transport_client("http://unused.invalid/v1")
    monkeypatch.setattr(client, "_LiteLLMClient__call_api_parallel", lambda *_args: [])
    doc = Doc(query="one", choices=["x"], gold_index=0, generation_size=2)

    with pytest.raises(ValueError, match="0 responses for 1 requests"):
        client.greedy_until([doc])


@pytest.mark.parametrize(
    "payload",
    [
        _completion_response(),
        _completion_response({"index": 0, "text": "", "finish_reason": "stop"}),
    ],
)
def test_text_completion_rejects_empty_or_malformed_success_schema(payload):
    with _OpenAICompatibleServer([(200, payload)]) as server:
        client = _transport_client(server.base_url)
        with pytest.raises(ValueError, match="choices|empty or malformed"):
            _call_text_completion(client, return_logits=False)

    assert len(server.requests) == 1


@pytest.mark.parametrize(
    ("choice", "return_logits", "message"),
    [
        (_choice("bad-logprob", logprob=True), True, "token log probabilities"),
        (_choice("bad-finish", finish_reason={"reason": "stop"}), False, "finish reason"),
        (_choice("bad-stop-list", stop_reason=["END"]), False, "stop reason"),
        (_choice("bad-stop-bool", stop_reason=True), False, "stop reason"),
        (_choice("bad-token-id", token_id=True), False, "terminal token id"),
    ],
)
def test_text_completion_retries_and_rejects_malformed_response_evidence(choice, return_logits, message):
    payload = _completion_response(choice)
    with _OpenAICompatibleServer([(200, payload), (200, payload)]) as server:
        client = _transport_client(server.base_url, retries=2)
        with pytest.raises(ValueError, match=message):
            _call_text_completion(client, return_logits=return_logits)

    assert len(server.requests) == 2


@pytest.mark.parametrize(
    ("status", "expected_exception", "expected_requests"),
    [
        (400, "BadRequestError", 1),
        (429, "RateLimitError", 2),
        (500, "InternalServerError", 2),
    ],
)
def test_text_completion_preserves_http_error_after_retry_exhaustion(status, expected_exception, expected_requests):
    error_payload = {
        "error": {
            "message": f"server-marker-{status}",
            "type": "test_error",
            "param": None,
            "code": str(status),
        }
    }
    with _OpenAICompatibleServer([(status, error_payload)] * expected_requests) as server:
        client = _transport_client(server.base_url, retries=2)
        with pytest.raises(Exception) as error:
            _call_text_completion(client, return_logits=False)

    assert type(error.value).__name__ == expected_exception
    assert f"server-marker-{status}" in str(error.value)
    assert len(server.requests) == expected_requests
