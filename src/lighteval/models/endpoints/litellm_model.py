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

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Literal

import httpx
import requests
from openai import OpenAI
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.rwkv_prompt import render_naive_prompt
from lighteval.utils.cache_management import SampleCache, cached
from lighteval.utils.imports import is_package_available, requires


logger = logging.getLogger(__name__)


class _OpenAIResponseCapture:
    """Capture successful raw JSON before LiteLLM coerces response fields."""

    def __init__(self, *, base_url: str | None, api_key: str | None, timeout: float | None):
        self.status_code: int | None = None
        self.payload: object = None
        self._http_client = httpx.Client(timeout=timeout, event_hooks={"response": [self._capture]})
        client_kwargs = {
            "api_key": api_key,
            "http_client": self._http_client,
            "max_retries": 0,
        }
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def reset(self) -> None:
        self.status_code = None
        self.payload = None

    def close(self) -> None:
        self.client.close()

    def _capture(self, response: httpx.Response) -> None:
        self.status_code = response.status_code
        if response.is_success:
            response.read()
            try:
                self.payload = response.json()
            except ValueError:
                self.payload = None


@dataclass(frozen=True)
class OpenAICompatibleRequest:
    """Rendered input for an endpoint that performs its own tokenization."""

    endpoint: Literal["/v1/chat/completions", "/v1/completions"]
    model_input: str | list[dict[str, str]]
    tokenization: Literal["server"] = "server"

    def as_payload(self) -> dict[str, object]:
        """Return the endpoint-specific input field."""
        if self.endpoint == "/v1/completions":
            if not isinstance(self.model_input, str):
                raise TypeError("text completions require one plain-text prompt")
            return {"prompt": self.model_input}
        if not isinstance(self.model_input, list):
            raise TypeError("chat completions require a message list")
        return {"messages": self.model_input}


def prepare_openai_compatible_request(
    doc: Doc,
    *,
    prompt_manager: PromptManager,
    use_chat_template: bool,
) -> OpenAICompatibleRequest:
    """Render a standard chat request or the RWKV naive text-completion path."""
    if use_chat_template:
        return OpenAICompatibleRequest(
            endpoint="/v1/chat/completions",
            model_input=prompt_manager.prepare_prompt_api(doc),
        )
    if prompt_manager.system_prompt is not None:
        raise ValueError("naive completions do not accept a model system prompt")
    return OpenAICompatibleRequest(
        endpoint="/v1/completions",
        model_input=render_naive_prompt(doc),
    )


if is_package_available("litellm"):
    import litellm
    from litellm import encode, supports_reasoning
    from litellm.utils import get_max_tokens

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").handlers.clear()

else:
    from unittest.mock import Mock

    litellm = Mock()
    encode = Mock()


class LiteLLMModelConfig(ModelConfig):
    """Configuration class for LiteLLM unified API client.

    This configuration is used to connect to various LLM providers through the LiteLLM
    unified API. LiteLLM provides a consistent interface to multiple providers including
    OpenAI, Anthropic, Google, and many others.

    litellm doc: https://docs.litellm.ai/docs/

    Attributes:
        model_name (str):
            Model identifier. Can include provider prefix (e.g., "gpt-4", "claude-3-sonnet")
            or use provider/model format (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet").
        provider (str | None):
            Optional provider name override. If None, inferred from model_name.
            Examples: "openai", "anthropic", "google", "cohere", etc.
        base_url (str | None):
            Custom base URL for the API. If None, uses provider's default URL.
            Useful for using custom endpoints or local deployments.
        api_key (str | None):
            API key for authentication. If None, reads from environment variables.
            Environment variable names are provider-specific (e.g., OPENAI_API_KEY).
        concurrent_requests (int):
            Maximum number of concurrent API requests to execute in parallel.
            Higher values can improve throughput for batch processing but may hit rate limits
            or exhaust API quotas faster. Default is 10.
        verbose (bool):
            Whether to enable verbose logging. Default is False.
        max_model_length (int | None):
            Maximum context length for the model. If None, infers the model's default max length.
        api_max_retry (int):
            Maximum number of retries for API requests. Default is 8.
        api_retry_sleep (float):
            Initial sleep time (in seconds) between retries. Default is 1.0.
        api_retry_multiplier (float):
            Multiplier for increasing sleep time between retries. Default is 2.0.
        timeout (float):
            Request timeout in seconds. Default is None (no timeout).
        use_chat_template (bool):
            Send chat-template messages to ``/v1/chat/completions`` when true.
            When false, render the complete task input as plain text and use
            ``/v1/completions``. Default is true for upstream compatibility.
        generation_parameters (GenerationParameters, optional, defaults to empty GenerationParameters):
            Configuration parameters that control text generation behavior, including
            temperature, top_p, max_new_tokens, etc.
        system_prompt (str | None, optional, defaults to None): Optional system prompt to be used with chat models.
            This prompt sets the behavior and context for the model during evaluation.
        cache_dir (str, optional, defaults to "~/.cache/huggingface/lighteval"): Directory to cache the model.

    Example:
        ```python
        config = LiteLLMModelConfig(
            model_name="gpt-4",
            provider="openai",
            base_url="https://api.openai.com/v1",
            concurrent_requests=5,
            generation_parameters=GenerationParameters(
                temperature=0.7,
                max_new_tokens=100
            )
        )
        ```
    """

    model_name: str
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    concurrent_requests: int = 10
    verbose: bool = False
    max_model_length: int | None = None

    api_max_retry: int = 8
    api_retry_sleep: float = 1.0
    api_retry_multiplier: float = 2.0
    timeout: float | None = None
    use_chat_template: bool = True


@requires("litellm")
class LiteLLMClient(LightevalModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, config: LiteLLMModelConfig) -> None:
        """IMPORTANT: Your API keys should be set in the environment variables.
        If a base_url is not set, it will default to the public API.
        """
        self.config = config
        self.model = config.model_name
        self.provider = config.provider or config.model_name.split("/")[0]
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.generation_parameters = config.generation_parameters
        self.concurrent_requests = config.concurrent_requests
        self._max_length = config.max_model_length

        self.API_MAX_RETRY = config.api_max_retry
        self.API_RETRY_SLEEP = config.api_retry_sleep
        self.API_RETRY_MULTIPLIER = config.api_retry_multiplier
        self.timeout = config.timeout
        self.use_chat_template = config.use_chat_template

        if not self.use_chat_template and config.system_prompt is not None:
            raise ValueError("naive completions do not accept a model system prompt")

        self._tokenizer = encode
        self.pairwise_tokenization = False
        litellm.drop_params = True
        litellm.verbose = config.verbose
        self.prompt_manager = PromptManager(
            use_chat_template=self.use_chat_template,
            tokenizer=self.tokenizer,
            system_prompt=config.system_prompt,
        )

        # Initialize cache for tokenization and predictions
        self._cache = SampleCache(config)

    def _prepare_stop_sequence(self, stop_sequence):
        """Prepare and validate stop sequence."""
        if self.provider == "anthropic":
            # Filter out whitespace-only stop sequences
            if stop_sequence:
                stop_sequence = [s for s in stop_sequence if s and s.strip()]
        return stop_sequence

    def _prepare_max_new_tokens(self, max_new_tokens) -> int | None:
        """Calculate completion tokens based on max_new_tokens."""
        if not max_new_tokens or max_new_tokens <= 0:
            return None

        if supports_reasoning(self.model):
            # We need to allow more tokens to include reasoning tokens
            max_new_tokens = min(max_new_tokens * 10, self.max_length)

            logger.warning(
                f"Reasoning model detected, increasing max_new_tokens to {max_new_tokens} to allow for reasoning tokens",
            )

        return max_new_tokens

    def __call_api(
        self,
        request: OpenAICompatibleRequest,
        return_logits,
        max_new_tokens,
        num_samples,
        stop_sequence,
    ):  # noqa: C901
        """Make API call with retries."""
        stop_sequence = self._prepare_stop_sequence(stop_sequence)
        max_new_tokens = self._prepare_max_new_tokens(max_new_tokens)

        if return_logits and self.provider != "openai":
            raise ValueError("token log probabilities require an OpenAI-compatible provider")

        # Prepare kwargs for completion call
        kwargs = {
            "model": self.model,
            "max_tokens": max_new_tokens,
            "logprobs": int(return_logits) if return_logits and self.provider == "openai" else None,
            "stop": stop_sequence,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "n": num_samples,
            # LightEval owns sample caching. A second transport-level cache can
            # replay malformed/error responses and bypass the explicit retry contract.
            "caching": False,
            "num_retries": 0,
            "timeout": self.timeout,
        }
        kwargs.update(request.as_payload())
        if request.endpoint == "/v1/chat/completions":
            kwargs["response_format"] = {"type": "text"}

        if "o1" in self.model:
            logger.warning("O1 models do not support temperature, top_p, stop sequence. Disabling.")
        else:
            kwargs.update(self.generation_parameters.to_litellm_dict())
            # Task stop sequences define the evaluation contract. Model-level
            # defaults are used only when the task does not provide one.
            if stop_sequence:
                kwargs["stop"] = stop_sequence

        if request.endpoint == "/v1/chat/completions":
            if kwargs.get("max_completion_tokens", None) is None:
                kwargs["max_completion_tokens"] = max_new_tokens
            completion = litellm.completion
        else:
            configured_max_tokens = kwargs.pop("max_completion_tokens", None)
            if configured_max_tokens is not None:
                kwargs["max_tokens"] = configured_max_tokens
            completion = litellm.text_completion

        response_capture = None
        if self.provider == "openai":
            response_capture = _OpenAIResponseCapture(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
            kwargs["client"] = response_capture.client
        try:
            return self._invoke_with_retries(
                completion,
                kwargs,
                endpoint=request.endpoint,
                num_samples=num_samples,
                return_logits=return_logits,
                response_capture=response_capture,
            )
        finally:
            if response_capture is not None:
                response_capture.close()

    def _invoke_with_retries(
        self,
        completion,
        kwargs,
        *,
        endpoint,
        num_samples,
        return_logits,
        response_capture,
    ):
        if self.API_MAX_RETRY < 1:
            raise ValueError("api_max_retry must be at least one")
        for attempt in range(self.API_MAX_RETRY):
            if response_capture is not None:
                response_capture.reset()
            try:
                response = completion(**kwargs)
                self._validate_captured_response(response_capture, num_samples, return_logits)
                self._validate_response(response, endpoint, num_samples, return_logits)
                return response
            except Exception as original_error:
                schema_error = self._captured_schema_error(
                    response_capture,
                    num_samples,
                    return_logits,
                )
                if isinstance(original_error, litellm.BadRequestError) and schema_error is None:
                    raise
                error = schema_error or original_error
                if attempt + 1 == self.API_MAX_RETRY:
                    raise error
                wait_time = min(
                    64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt)
                )  # Exponential backoff with max 64s
                logger.warning(
                    f"Error in API call: {error}, waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                time.sleep(wait_time)

        raise AssertionError("unreachable API retry state")

    @classmethod
    def _captured_schema_error(cls, response_capture, num_samples, return_logits):
        try:
            cls._validate_captured_response(response_capture, num_samples, return_logits)
        except ValueError as error:
            return error
        return None

    @classmethod
    def _validate_captured_response(cls, response_capture, num_samples, return_logits) -> None:
        if response_capture is None or response_capture.status_code is None or response_capture.status_code >= 300:
            return
        payload = response_capture.payload
        if not isinstance(payload, dict):
            raise ValueError("endpoint returned malformed raw JSON")
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != num_samples:
            raise ValueError("endpoint returned malformed raw choices")
        for choice in choices:
            cls._validate_raw_choice(choice, return_logits)

    @classmethod
    def _validate_raw_choice(cls, choice, return_logits) -> None:
        if not isinstance(choice, dict):
            raise ValueError("endpoint returned a malformed raw completion choice")
        cls._validate_raw_logprobs(choice.get("logprobs"), required=return_logits)
        cls._validate_finish_reason(choice.get("finish_reason"))
        cls._validate_stop_reason(choice.get("stop_reason"))
        token_ids = choice.get("token_ids")
        if token_ids is not None:
            if not isinstance(token_ids, list):
                raise ValueError("endpoint returned a malformed terminal token id")
            if token_ids:
                cls._validate_terminal_token_id(token_ids[-1])

    @staticmethod
    def _validate_raw_logprobs(logprobs, *, required: bool) -> None:
        if logprobs is None:
            if required:
                raise ValueError("endpoint omitted requested token log probabilities")
            return
        if not isinstance(logprobs, dict) or not isinstance(logprobs.get("token_logprobs"), list):
            raise ValueError("endpoint returned malformed token log probabilities")
        for value in logprobs["token_logprobs"]:
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value)
            ):
                raise ValueError("endpoint returned malformed token log probabilities")

    @classmethod
    def _validate_response(cls, response, endpoint: str, num_samples: int, return_logits: bool) -> None:
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or len(choices) != num_samples:
            raise ValueError(
                f"endpoint returned {0 if not isinstance(choices, list) else len(choices)} choices, expected {num_samples}"
            )
        for choice in choices:
            content = cls._choice_content(choice, endpoint)
            if not isinstance(content, str):
                raise ValueError("endpoint returned a missing or malformed completion choice")
            if return_logits:
                token_logprobs = cls._choice_token_logprobs(choice)
                if not token_logprobs:
                    raise ValueError("endpoint omitted requested token log probabilities")
            cls._choice_finish_reason(choice)
            cls._choice_stop_reason(choice)
            cls._choice_terminal_token_id(choice)

    @staticmethod
    def _choice_content(choice, endpoint: str) -> str | None:
        if endpoint == "/v1/completions":
            return LiteLLMClient._response_field(choice, "text")
        message = LiteLLMClient._response_field(choice, "message")
        return LiteLLMClient._response_field(message, "content")

    @staticmethod
    def _response_field(value, name: str):
        if isinstance(value, dict):
            return value.get(name)
        field = getattr(value, name, None)
        if field is not None:
            return field
        model_extra = getattr(value, "model_extra", None)
        return model_extra.get(name) if isinstance(model_extra, dict) else None

    @classmethod
    def _choice_token_logprobs(cls, choice) -> list[float | None]:
        logprobs = cls._response_field(choice, "logprobs")
        values = cls._response_field(logprobs, "token_logprobs")
        if not isinstance(values, list):
            return []
        parsed: list[float | None] = []
        for value in values:
            if value is None:
                parsed.append(None)
            elif isinstance(value, bool):
                raise ValueError("endpoint returned malformed token log probabilities")
            elif isinstance(value, int | float) and math.isfinite(value):
                parsed.append(float(value))
            else:
                raise ValueError("endpoint returned malformed token log probabilities")
        return parsed

    @classmethod
    def _choice_finish_reason(cls, choice) -> str | None:
        return cls._validate_finish_reason(cls._response_field(choice, "finish_reason"))

    @staticmethod
    def _validate_finish_reason(finish_reason) -> str | None:
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ValueError("endpoint returned a malformed finish reason")
        return finish_reason

    @classmethod
    def _choice_stop_reason(cls, choice) -> str | int | None:
        return cls._validate_stop_reason(cls._response_field(choice, "stop_reason"))

    @staticmethod
    def _validate_stop_reason(stop_reason) -> str | int | None:
        if isinstance(stop_reason, bool) or (stop_reason is not None and not isinstance(stop_reason, str | int)):
            raise ValueError("endpoint returned a malformed stop reason")
        return stop_reason

    @classmethod
    def _choice_terminal_token_id(cls, choice) -> int | None:
        token_ids = cls._response_field(choice, "token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            return None
        return cls._validate_terminal_token_id(token_ids[-1])

    @staticmethod
    def _validate_terminal_token_id(terminal_token_id) -> int | None:
        if terminal_token_id is None:
            return None
        if not isinstance(terminal_token_id, int) or isinstance(terminal_token_id, bool):
            raise ValueError("endpoint returned a malformed terminal token id")
        return terminal_token_id

    def __call_api_parallel(
        self,
        requests,
        return_logits: bool | list[bool],
        max_new_tokens: int | list[int] | None,
        num_samples: int | list[int],
        stop_sequence: list[str] | None = None,
    ):
        results = []

        return_logitss = [return_logits for _ in requests] if not isinstance(return_logits, list) else return_logits
        max_new_tokenss = (
            [max_new_tokens for _ in requests] if not isinstance(max_new_tokens, list) else max_new_tokens
        )
        num_sampless = [num_samples for _ in requests] if not isinstance(num_samples, list) else num_samples
        stop_sequencess = [stop_sequence for _ in requests]
        assert (
            len(requests) == len(return_logitss) == len(max_new_tokenss) == len(num_sampless) == len(stop_sequencess)
        ), (
            "Length of requests, return_logitss, max_new_tokenss, "
            "num_sampless, stop_sequences should be the same but are "
            f"{len(requests)}, {len(return_logitss)}, {len(max_new_tokenss)}, "
            f"{len(num_sampless)}, {len(stop_sequencess)}"
        )

        with ThreadPoolExecutor(self.concurrent_requests) as executor:
            for entry in tqdm(
                executor.map(
                    self.__call_api,
                    requests,
                    return_logitss,
                    max_new_tokenss,
                    num_sampless,
                    stop_sequencess,
                ),
                total=len(requests),
            ):
                results.append(entry)

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")

        if len(results) != len(requests):
            raise ValueError(f"endpoint returned {len(results)} responses for {len(requests)} requests")
        return results

    def estimate_context_length(self) -> int:
        def fallback():
            logger.warning("Failed to fetch model endpoint info from OpenRouter, returning default max length.")
            return self._DEFAULT_MAX_LENGTH

        # If the model is used through openrouter, the actual model name comes after the prefix
        model_name = self.model.removeprefix("openrouter/")
        endpoint_info_response = requests.get(
            f"https://openrouter.ai/api/v1/models/{model_name}/endpoints",
            headers={},
        )
        if endpoint_info_response.ok:
            try:
                endpoint_info = endpoint_info_response.json()
                context_lengths = {
                    endpoint["provider_name"]: endpoint["context_length"]
                    for endpoint in endpoint_info["data"]["endpoints"]
                }

                if self.provider in context_lengths:
                    return context_lengths[self.provider]

                min_length = min(context_lengths.values())
                logger.warning(
                    f"Estimating model context length as the minimum context length from available OpenRouter providers: {min_length}"
                )
                return min_length
            except (KeyError, TypeError, ValueError, JSONDecodeError):
                return fallback()

        return fallback()

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            docs (list[Doc]): List of documents containing the context for generation.

        Returns:
            list[ModelResponse]: list of generated responses.
        """
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            split_docs = list(split)
            requests = [
                prepare_openai_compatible_request(
                    doc,
                    prompt_manager=self.prompt_manager,
                    use_chat_template=self.use_chat_template,
                )
                for doc in split_docs
            ]
            contexts = [request.model_input for request in requests]
            max_new_tokens = split[0].generation_size  # could be none
            return_logits = split[0].use_logits
            num_samples = split[0].num_samples
            stop_sequence = split[0].stop_sequences

            if num_samples > 1 and self.generation_parameters.temperature == 0:
                raise ValueError(
                    "num_samples > 1 is not supported with temperature=0, please set temperature > 0 or use non sampling metrics."
                )

            responses = self.__call_api_parallel(requests, return_logits, max_new_tokens, num_samples, stop_sequence)
            if len(responses) != len(requests):
                raise ValueError(f"endpoint returned {len(responses)} responses for {len(requests)} requests")

            for response, context in zip(responses, contexts):
                if self.use_chat_template:
                    result: list[str] = [
                        self._choice_content(choice, "/v1/chat/completions") for choice in response.choices
                    ]
                    reasonings: list[str | None] = [
                        self._response_field(self._response_field(choice, "message"), "reasoning_content")
                        for choice in response.choices
                    ]
                else:
                    result = [self._choice_content(choice, "/v1/completions") for choice in response.choices]
                    reasonings = [self._response_field(choice, "reasoning_content") for choice in response.choices]

                token_logprobs = [self._choice_token_logprobs(choice) for choice in response.choices]
                finish_reasons = [self._choice_finish_reason(choice) for choice in response.choices]
                stop_reasons = [self._choice_stop_reason(choice) for choice in response.choices]
                terminal_token_ids = [self._choice_terminal_token_id(choice) for choice in response.choices]

                cur_response = ModelResponse(
                    text=result,
                    reasonings=reasonings,
                    input=context,
                    token_logprobs=token_logprobs,
                    finish_reasons=finish_reasons,
                    stop_reasons=stop_reasons,
                    terminal_token_ids=terminal_token_ids,
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        if self._max_length is not None:
            return self._max_length

        try:
            max_tokens = get_max_tokens(self.model)
        except Exception:
            logger.error(
                f"Unable to get the maximum sequence length for model {self.model} from litellm. Fetching information from OpenRouter instead."
            )
            max_tokens = self.estimate_context_length()

        # Avoid future requests
        self._max_length = max_tokens

        return max_tokens

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError

    @cached(SamplingMethod.PERPLEXITY)
    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        raise NotImplementedError
