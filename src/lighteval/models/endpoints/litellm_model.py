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
import time
from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from typing import Any

import requests
from pydantic import SecretStr
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached
from lighteval.utils.imports import is_package_available, requires


logger = logging.getLogger(__name__)


def _strip_question_answer_completion_template(prompt: Any) -> Any:
    """Remove a completion-only wrapper before applying an RWKV chat template.

    LightEval completion tasks commonly wrap the entire user request as
    ``Question: ... Answer:``.  In generated RWKV modes the server adds its own
    User/Assistant boundary, so retaining both wrappers gives the model two
    competing answer cues.  Only rewrite a user message when both exact edge
    markers are present; embedded labels and all other prompts stay unchanged.
    """
    if not isinstance(prompt, (list, tuple)):
        return prompt

    messages = list(prompt)
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        stripped = content.strip()
        prefix, suffix = "Question: ", "Answer:"
        if not stripped.startswith(prefix) or not stripped.endswith(suffix):
            break
        cleaned = stripped[len(prefix) : -len(suffix)].strip()
        messages[index] = {**message, "content": cleaned}
        break
    return messages


def _response_prompt_text(response: Any) -> str | None:
    """Read an optional server-rendered prompt without assuming provider fields."""
    candidates: list[Any] = [response]
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            candidates.append(dumped)
    hidden_params = getattr(response, "_hidden_params", None)
    if isinstance(hidden_params, dict):
        candidates.append(hidden_params)
    for candidate in candidates:
        for name in ("prompt_text", "rendered_prompt"):
            value = candidate.get(name) if isinstance(candidate, dict) else getattr(candidate, name, None)
            if isinstance(value, str):
                return value
    return None


if is_package_available("litellm"):
    import litellm
    from litellm import encode, supports_reasoning
    from litellm.caching.caching import Cache, LiteLLMCacheType
    from litellm.utils import ModelResponse as LitellmModelResponse
    from litellm.utils import get_max_tokens

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").handlers.clear()

    # Include provider-specific request parameters (for example RWKV's
    # top_k and extra_body fields) in LiteLLM's response-cache key.
    litellm.enable_caching_on_provider_specific_optional_params = True
    litellm.cache = Cache(type=LiteLLMCacheType.DISK)
else:
    from unittest.mock import Mock

    litellm = Mock()
    encode = Mock()
    LitellmModelResponse = Mock()


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
        api_key (SecretStr | None):
            API key for authentication. If None, reads from environment variables.
            Environment variable names are provider-specific (e.g., OPENAI_API_KEY).
            Stored as a SecretStr so it is masked in logs, reprs, and serialized configs.
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
    api_key: SecretStr | None = None
    concurrent_requests: int = 10
    verbose: bool = False
    max_model_length: int | None = None
    scoreboard_ctx_len: int | None = None

    api_max_retry: int = 8
    api_retry_sleep: float = 1.0
    api_retry_multiplier: float = 2.0
    timeout: float | None = None
    # Parameters accepted by an OpenAI-compatible server but not normalized by
    # LiteLLM are forwarded through the provider's extra_body mechanism.
    extra_body: dict[str, Any] | None = None
    generation_only: bool = False
    target_completions: int = 0
    minimum_completions: int = 0
    maximum_completions: int = 0


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
        self.api_key = config.api_key.get_secret_value() if config.api_key is not None else None
        self.generation_parameters = config.generation_parameters
        self.concurrent_requests = config.concurrent_requests
        self._max_length = config.max_model_length

        self.API_MAX_RETRY = config.api_max_retry
        self.API_RETRY_SLEEP = config.api_retry_sleep
        self.API_RETRY_MULTIPLIER = config.api_retry_multiplier
        self.timeout = config.timeout

        self._tokenizer = encode
        self.pairwise_tokenization = False
        litellm.drop_params = True
        litellm.verbose = config.verbose
        self.prompt_manager = PromptManager(
            use_chat_template=True, tokenizer=self.tokenizer, system_prompt=config.system_prompt
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

    def _prepare_max_new_tokens(self, max_new_tokens, prompt: Any = None) -> int | None:
        """Fit the requested completion budget inside the configured context."""
        if not max_new_tokens or max_new_tokens <= 0:
            return None

        if supports_reasoning(self.model):
            max_new_tokens = min(max_new_tokens * 10, self.max_length)
            logger.warning(
                f"Reasoning model detected, increasing max_new_tokens to {max_new_tokens} to allow for reasoning tokens",
            )

        if self.max_length and prompt is not None and max_new_tokens >= self.max_length:
            messages = prompt if isinstance(prompt, (list, tuple)) else [prompt]
            text = "\n".join(
                message.get("content", "")
                if isinstance(message, dict) and isinstance(message.get("content"), str)
                else str(message)
                for message in messages
            )
            prompt_tokens = len(self.tokenizer(model=self.model, text=text))
            max_new_tokens = min(max_new_tokens, max(1, self.max_length - prompt_tokens - 1))

        return max_new_tokens

    def __call_api(self, prompt, return_logits, max_new_tokens, num_samples, stop_sequence):  # noqa: C901
        """Make API call with retries."""
        response = LitellmModelResponse()
        stop_sequence = self._prepare_stop_sequence(stop_sequence)
        if isinstance(self.config.extra_body, dict):
            chat_template_kwargs = self.config.extra_body.get("chat_template_kwargs", {})
        else:
            chat_template_kwargs = {}
        rwkv_generation_prompt = (
            chat_template_kwargs.get("rwkv_generation_prompt") if isinstance(chat_template_kwargs, dict) else None
        )
        if rwkv_generation_prompt in {"fake_think", "open_think"}:
            prompt = _strip_question_answer_completion_template(prompt)
            # A task's newline stop sequence is valid for a one-token answer,
            # but it terminates the RWKV reasoning trace before the answer.
            stop_sequence = []
        max_new_tokens = self._prepare_max_new_tokens(max_new_tokens, prompt)

        if return_logits and not self.provider == "openai":
            logger.warning("Returning logits is not supported for this provider, ignoring.")

        # Prepare kwargs for completion call
        kwargs = {
            "model": self.model,
            # A bare model name cannot be routed by LiteLLM.  The provider is
            # deliberately kept in the LightEval config (rather than baked
            # into model_name), so forward it on every request.
            "custom_llm_provider": self.provider,
            "messages": prompt,
            "response_format": {"type": "text"},
            "max_tokens": max_new_tokens,
            "logprobs": return_logits if self.provider == "openai" else None,
            "stop": stop_sequence,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "n": num_samples,
            "caching": True,
            "timeout": self.timeout,
        }

        if "o1" in self.model:
            logger.warning("O1 models do not support temperature, top_p, stop sequence. Disabling.")
        else:
            kwargs.update(self.generation_parameters.to_litellm_dict())
        if rwkv_generation_prompt in {"fake_think", "open_think"}:
            kwargs["max_tokens"] = max_new_tokens
            kwargs["max_completion_tokens"] = max_new_tokens
            kwargs["stop"] = stop_sequence
            logger.info(
                "RWKV request: model=%s endpoint=%s/chat/completions mode=%s max_tokens=%s context=%s stop=%r",
                self.model,
                self.base_url,
                rwkv_generation_prompt,
                max_new_tokens,
                self.max_length,
                stop_sequence,
            )
        elif kwargs.get("max_completion_tokens") is None:
            kwargs["max_completion_tokens"] = max_new_tokens
        if self.config.extra_body:
            extra_body = dict(self.config.extra_body)
            chat_template_kwargs = extra_body.get("chat_template_kwargs")
            if isinstance(chat_template_kwargs, dict) and (
                "rwkv_prompt_template" in chat_template_kwargs or "rwkv_generation_prompt" in chat_template_kwargs
            ):
                # vLLM's RWKV endpoint exposes the exact post-template text
                # through this provider-specific flag.  Preserve an explicit
                # caller value, while making RWKV configs self-describing for
                # details saved by the LightEval tracker.
                extra_body.setdefault("return_prompt_text", True)
            kwargs["extra_body"] = extra_body

        last_error: BaseException | None = None
        for attempt in range(self.API_MAX_RETRY):
            try:
                response = litellm.completion(**kwargs)
                content = response.choices[0].message.content

                # If response is empty, retry without caching (maybe the error is recoverable and solved with a retry)
                if not content:
                    logger.info("Response is empty, retrying without caching")
                    kwargs["caching"] = False
                    response = litellm.completion(**kwargs)
                    content = response.choices[0].message.content

                return response
            except litellm.BadRequestError as e:
                if rwkv_generation_prompt in {"fake_think", "open_think"}:
                    raise
                if "message" in e.__dict__:
                    error_string = (
                        "The response was filtered due to the prompt triggering Microsoft's content management policy"
                    )
                    if error_string in e.__dict__["message"]:
                        logger.warning(f"{error_string}. Returning empty response.")
                        return LitellmModelResponse()
            except Exception as e:
                last_error = e
                wait_time = min(
                    64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt)
                )  # Exponential backoff with max 64s
                logger.warning(
                    f"Error in API call: {e}, waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                time.sleep(wait_time)

        message = f"API call failed after {self.API_MAX_RETRY} attempts"
        logger.error(message)
        if self.config.generation_only or rwkv_generation_prompt in {"fake_think", "open_think"}:
            raise RuntimeError(message) from last_error
        return LitellmModelResponse()

    def __call_api_parallel(
        self,
        prompts,
        return_logits: bool | list[bool],
        max_new_tokens: int | list[int] | None,
        num_samples: int | list[int],
        stop_sequence: list[str] | None = None,
    ):
        results = []

        return_logitss = [return_logits for _ in prompts] if not isinstance(return_logits, list) else return_logits
        max_new_tokenss = [max_new_tokens for _ in prompts] if not isinstance(max_new_tokens, list) else max_new_tokens
        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
        stop_sequencess = [stop_sequence for _ in prompts]
        assert (
            len(prompts) == len(return_logitss) == len(max_new_tokenss) == len(num_sampless) == len(stop_sequencess)
        ), (
            f"Length of prompts, return_logitss, max_new_tokenss, num_sampless, stop_sequences, system_prompts should be the same but are {len(prompts)}, {len(return_logitss)}, {len(max_new_tokenss)}, {len(num_sampless)}, {len(stop_sequencess)}"
        )

        with ThreadPoolExecutor(self.concurrent_requests) as executor:
            for entry in tqdm(
                executor.map(
                    self.__call_api,
                    prompts,
                    return_logitss,
                    max_new_tokenss,
                    num_sampless,
                    stop_sequencess,
                ),
                total=len(prompts),
            ):
                results.append(entry)

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")

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
            contexts = [self.prompt_manager.prepare_prompt_api(doc) for doc in dataset]
            max_new_tokens = self.generation_parameters.max_new_tokens or split[0].generation_size
            return_logits = split[0].use_logits
            num_samples = split[0].num_samples
            stop_sequence = split[0].stop_sequences

            if num_samples > 1 and self.generation_parameters.temperature == 0:
                raise ValueError(
                    "num_samples > 1 is not supported with temperature=0, please set temperature > 0 or use non sampling metrics."
                )

            responses = self.__call_api_parallel(contexts, return_logits, max_new_tokens, num_samples, stop_sequence)

            for response, context in zip(responses, contexts):
                rendered_prompt = _response_prompt_text(response)
                result: list[str] = [choice.message.content for choice in response.choices]
                reasonings: list[str | None] = [
                    getattr(choice.message, "reasoning_content", None) for choice in response.choices
                ]
                finish_reasons = [getattr(choice, "finish_reason", "stop") or "stop" for choice in response.choices]

                cur_response = ModelResponse(
                    # In empty responses, the model should return an empty string instead of None
                    text=result if result and result[0] else [""],
                    reasonings=reasonings,
                    finish_reasons=finish_reasons,
                    input=rendered_prompt if rendered_prompt is not None else context,
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

    def _prompt_logprobs_request(self, prompt: str | list[str], *, top_logprobs: int) -> list[dict[str, Any]]:
        """Request prompt token logprobs from the OpenAI-compatible completions API."""
        request_url = f"{self.base_url.rstrip('/')}/completions"
        request_headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        request_body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 0,
            "echo": True,
            "prompt_logprobs": top_logprobs,
            "return_token_ids": True,
        }
        for attempt in range(6):
            response = requests.post(
                request_url,
                headers=request_headers,
                json=request_body,
                timeout=self.timeout or 120.0,
            )
            if response.status_code < 500 or attempt == 5:
                break
            time.sleep(min(2**attempt, 16))
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices")
        if not isinstance(choices, list) or not all(isinstance(choice, dict) for choice in choices):
            raise RuntimeError("RWKV prompt_logprobs response did not contain choices")
        return choices

    @staticmethod
    def _candidate_logprob(choice: dict[str, Any], prefix_token_ids: list[int]) -> float:
        token_ids = choice.get("prompt_token_ids")
        prompt_logprobs = choice.get("prompt_logprobs")
        if not isinstance(token_ids, list) or not isinstance(prompt_logprobs, list):
            raise RuntimeError("RWKV prompt_logprobs response did not contain token ids and logprobs")
        if len(prompt_logprobs) != len(token_ids):
            raise RuntimeError("RWKV prompt_logprobs response has an invalid candidate span")
        # Tokenizers may merge the final prefix token with the first candidate
        # bytes.  The candidate span therefore starts at the longest common
        # token prefix, not necessarily len(tokenize(prefix)).
        candidate_start = 0
        for prefix_id, token_id in zip(prefix_token_ids, token_ids):
            if prefix_id != token_id:
                break
            candidate_start += 1
        if candidate_start >= len(token_ids):
            raise RuntimeError("RWKV prompt_logprobs response contains no candidate tokens")
        score = 0.0
        for index in range(candidate_start, len(token_ids)):
            entries = prompt_logprobs[index]
            if not isinstance(entries, dict):
                raise RuntimeError(f"RWKV prompt logprob is missing at token position {index}")
            value = entries.get(str(token_ids[index]))
            if not isinstance(value, dict) or not isinstance(value.get("logprob"), (int, float)):
                raise RuntimeError(
                    "RWKV prompt_logprobs did not return the selected candidate token; "
                    "increase the server prompt_logprobs limit"
                )
            score += float(value["logprob"])
        return score

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        """Score each choice with native prompt logprobs; never generate a fallback answer."""
        results: list[ModelResponse] = []
        for doc in docs:
            if not isinstance(doc.choices, list) or not doc.choices:
                raise ValueError("RWKV loglikelihood requires a non-empty choice list")
            prefix = doc.query
            prefix_choice = self._prompt_logprobs_request(prefix, top_logprobs=1)[0]
            prefix_token_ids = prefix_choice.get("prompt_token_ids")
            if not isinstance(prefix_token_ids, list):
                raise RuntimeError("RWKV prompt_logprobs response did not contain prefix token ids")
            prompts = [prefix + str(choice) for choice in doc.choices]
            candidate_choices = self._prompt_logprobs_request(prompts, top_logprobs=20)
            if len(candidate_choices) != len(prompts):
                raise RuntimeError("RWKV prompt_logprobs response count did not match choices")
            scores = [self._candidate_logprob(choice, prefix_token_ids) for choice in candidate_choices]
            results.append(
                ModelResponse(
                    text=[""],
                    input=prefix,
                    input_tokens=prefix_token_ids,
                    logprobs=scores,
                    output_tokens=[[] for _ in scores],
                    finish_reasons=["stop"],
                )
            )
        return results

    @cached(SamplingMethod.PERPLEXITY)
    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        raise NotImplementedError
