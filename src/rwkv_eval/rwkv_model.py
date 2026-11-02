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

"""RWKV HTTP model adapter built on LightEval's native LiteLLM client."""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Sequence

import httpx
import requests

from lighteval.models.endpoints.litellm_model import (
    LiteLLMClient,
    LiteLLMModelConfig,
    LitellmModelResponse,
    litellm,
)
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import cached
from src.rwkv_eval.configs import ModelEndpoint, SamplingConfig


LOGGER = logging.getLogger(__name__)


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


def _sampling_config(cot_mode: str, max_tokens: int, seed: int) -> SamplingConfig:
    values = {
        "NoCoT": (0.0, 0, 1.0, 0.0, 0.0, 1.0),
        "FakeCoT": (1.0, 32, 0.28, 0.0, 0.0, 1.0),
        "CoT": (0.96, 0, 0.90, 1.0, 0.09, 0.994),
    }
    try:
        temp, top_k, top_p, presence, frequency, decay = values[cot_mode]
    except KeyError as error:
        raise ValueError(f"unsupported cot_mode: {cot_mode!r}") from error
    return SamplingConfig(max_tokens, temp, top_k, top_p, presence, frequency, decay, seed)


def _litelm_model(
    endpoint: ModelEndpoint,
    replicas: Sequence[ModelEndpoint],
    cot_mode: str,
    template: str | None,
    max_tokens: int,
    seed: int,
    cache_dir: str | None = None,
):
    """Build the shared LiteLLM configuration for one RWKV endpoint pool."""
    from lighteval.models.model_input import GenerationParameters

    if any(item.url != endpoint.url for item in replicas):
        raise ValueError("LiteLLM endpoint pooling requires replicas to share one base URL")
    sampling = _sampling_config(cot_mode, max_tokens, seed)
    generation_prompt = {"FakeCoT": "fake_think", "CoT": "open_think"}.get(cot_mode)
    extra_body: dict[str, Any] = {"penalty_decay": sampling.penalty_decay}
    if cot_mode != "NoCoT":
        extra_body.update(
            {
                "return_prompt_text": True,
                "chat_template_kwargs": {
                    "rwkv_prompt_template": template,
                    "rwkv_generation_prompt": generation_prompt,
                },
            }
        )
    return RWKVModelConfig(
        model_name=endpoint.model_name,
        provider="openai",
        base_url=f"{endpoint.url.rstrip('/')}/v1",
        api_key=endpoint.api_key,
        concurrent_requests=sum(item.max_num_seqs for item in replicas),
        max_model_length=endpoint.ctx_len,
        scoreboard_ctx_len=endpoint.ctx_len,
        cache_dir=cache_dir or "~/.cache/huggingface/lighteval",
        api_max_retry=6,
        timeout=14400.0,
        extra_body=extra_body,
        generation_only=cot_mode != "NoCoT",
        # Restore the RWKV sampling policy from 59b3974f. Native sampling
        # metrics choose a power-of-two k; max_samples keeps smoke tests at 1.
        target_completions=4096 if cot_mode != "NoCoT" else 0,
        minimum_completions=3000 if cot_mode != "NoCoT" else 0,
        maximum_completions=6000 if cot_mode != "NoCoT" else 0,
        generation_parameters=GenerationParameters(
            temperature=sampling.temp,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            presence_penalty=sampling.presence_penalty,
            frequency_penalty=sampling.frequency_penalty,
            max_new_tokens=sampling.max_generated_tokens,
            seed=sampling.seed,
        ),
    )


class RWKVModelConfig(LiteLLMModelConfig):
    """RWKV endpoint options, following the native LiteLLM model config."""

    scoreboard_ctx_len: int | None = None
    extra_body: dict[str, Any] | None = None
    generation_only: bool = False
    target_completions: int = 0
    minimum_completions: int = 0
    maximum_completions: int = 0


class RWKVClient(LiteLLMClient):
    """RWKV requests and rollout persistence, following LiteLLMClient."""

    def __call_api(self, prompt, return_logits, max_new_tokens, num_samples, stop_sequence, seed_offset=0):  # noqa: C901
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
        if rwkv_generation_prompt not in {"fake_think", "open_think"}:
            max_new_tokens = self._prepare_max_new_tokens(max_new_tokens)
        # RWKV budgets limit generated tokens, not prompt + completion.
        # The inference backend owns its independently configured context.

        if return_logits and not self.provider == "openai":
            LOGGER.warning("Returning logits is not supported for this provider, ignoring.")

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
            LOGGER.warning("O1 models do not support temperature, top_p, stop sequence. Disabling.")
        else:
            kwargs.update(self.generation_parameters.to_litellm_dict())
        request_id = None
        if rwkv_generation_prompt in {"fake_think", "open_think"}:
            # SampleCache already persists RWKV responses with the complete
            # model-config key.  Disable only this request's redundant provider
            # cache instead of changing LiteLLM's global cache-key policy.
            kwargs["caching"] = False
            # LiteLLM's synchronous post-call handler can still write when
            # caching=False; no-store prevents a SQLite connection per rollout.
            kwargs["cache"] = {"no-store": True}
            # Native vLLM ParentRequest seeds child i with parent seed + i.
            # Separate HTTP rollouts retain that sampling contract on resume.
            if kwargs.get("seed") is not None:
                kwargs["seed"] += seed_offset
            # Prototype: OpenAI's httpx.Timeout support. Bound a non-streaming
            # read by the requested output budget so a lost gateway response
            # cannot leave an otherwise idle pool waiting for four hours.
            timeout = self.timeout or 14400
            read_timeout = min(timeout, max(1800, max_new_tokens / 2))
            kwargs["timeout"] = httpx.Timeout(timeout, connect=30.0, read=read_timeout)
            kwargs["max_tokens"] = max_new_tokens
            kwargs["max_completion_tokens"] = max_new_tokens
            kwargs["stop"] = stop_sequence
            request_id = uuid.uuid4().hex
            # LiteLLM proxies have their own upstream timeout; direct OpenAI-compatible
            # backends ignore this header. Keep it aligned with the SDK read timeout.
            kwargs["extra_headers"] = {
                "X-Request-ID": request_id,
                "X-Litellm-Timeout": str(read_timeout),
            }
            LOGGER.info(
                "RWKV request: model=%s request_id=%s endpoint=%s/chat/completions request_type=GENERATIVE mode=%s max_new_tokens=%s seed=%s timeout=%s stop=%r",
                self.model,
                request_id,
                self.base_url,
                rwkv_generation_prompt,
                max_new_tokens,
                kwargs.get("seed"),
                kwargs["timeout"],
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
            started = time.monotonic()
            try:
                response = litellm.completion(**kwargs)
                content = response.choices[0].message.content

                # If response is empty, retry without caching (maybe the error is recoverable and solved with a retry)
                if not content:
                    LOGGER.info("Response is empty, retrying without caching")
                    kwargs["caching"] = False
                    response = litellm.completion(**kwargs)
                    content = response.choices[0].message.content

                if rwkv_generation_prompt in {"fake_think", "open_think"}:
                    response._hidden_params["rwkv_prompt_token_count"] = response.usage.prompt_tokens
                    response._hidden_params["rwkv_effective_max_new_tokens"] = max_new_tokens
                    response._hidden_params["rwkv_seed"] = kwargs.get("seed")
                    gateway_id = (response._hidden_params.get("additional_headers") or {}).get(
                        "llm_provider-x-litellm-call-id"
                    )
                    LOGGER.info(
                        "RWKV GENERATIVE response: model=%s request_id=%s gateway_call_id=%s elapsed=%.2fs",
                        self.model,
                        request_id,
                        gateway_id,
                        time.monotonic() - started,
                    )
                return response
            except litellm.BadRequestError as e:
                if rwkv_generation_prompt in {"fake_think", "open_think"}:
                    raise
                if "message" in e.__dict__:
                    error_string = (
                        "The response was filtered due to the prompt triggering Microsoft's content management policy"
                    )
                    if error_string in e.__dict__["message"]:
                        LOGGER.warning(f"{error_string}. Returning empty response.")
                        return LitellmModelResponse()
            except Exception as e:
                if rwkv_generation_prompt in {"fake_think", "open_think"} and getattr(e, "status_code", None) in {
                    401,
                    403,
                }:
                    raise
                last_error = e
                wait_time = min(
                    64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt)
                )  # Exponential backoff with max 64s
                if request_id:
                    headers = getattr(getattr(e, "response", None), "headers", {}) or {}
                    LOGGER.warning(
                        "RWKV GENERATIVE transport failure: model=%s request_id=%s gateway_call_id=%s attempt=%d elapsed=%.2fs error=%s status=%s retry_in=%s",
                        self.model,
                        request_id,
                        headers.get("x-litellm-call-id"),
                        attempt + 1,
                        time.monotonic() - started,
                        type(e).__name__,
                        getattr(e, "status_code", None),
                        wait_time,
                    )
                else:
                    LOGGER.warning(
                        f"Error in API call: {e}, waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                    )
                time.sleep(wait_time)

        message = f"API call failed after {self.API_MAX_RETRY} attempts"
        LOGGER.error(message)
        if self.config.generation_only or rwkv_generation_prompt in {"fake_think", "open_think"}:
            raise RuntimeError(message) from last_error
        return LitellmModelResponse()

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
        template_kwargs = (self.config.extra_body or {}).get("chat_template_kwargs", {})
        if isinstance(template_kwargs, dict) and template_kwargs.get("rwkv_generation_prompt") in {
            "fake_think",
            "open_think",
        }:
            return self._greedy_until_rwkv(docs)
        return super().greedy_until(docs)

    @staticmethod
    def _generation_response(response, context):
        """Prototype: greedy_until's native response conversion."""
        candidates: list[Any] = [response]
        if callable(model_dump := getattr(response, "model_dump", None)):
            dumped = model_dump()
            if isinstance(dumped, dict):
                candidates.append(dumped)
        hidden = getattr(response, "_hidden_params", {})
        if isinstance(hidden, dict):
            candidates.append(hidden)
        rendered_prompt = next(
            (
                value
                for candidate in candidates
                for name in ("prompt_text", "rendered_prompt")
                if isinstance(
                    value := candidate.get(name) if isinstance(candidate, dict) else getattr(candidate, name, None),
                    str,
                )
            ),
            None,
        )
        effective_budget = hidden.get("rwkv_effective_max_new_tokens")
        if effective_budget is not None and rendered_prompt is None:
            raise RuntimeError("RWKV generation is missing its captured server-rendered prompt")
        texts = [choice.message.content for choice in response.choices]
        return ModelResponse(
            text=texts if texts and texts[0] else [""],
            rendered_prompt=rendered_prompt,
            effective_max_new_tokens=effective_budget,
            seeds=[hidden["rwkv_seed"]] if "rwkv_seed" in hidden else [],
            prompt_token_count=hidden.get("rwkv_prompt_token_count"),
            reasonings=[getattr(choice.message, "reasoning_content", None) for choice in response.choices],
            finish_reasons=[getattr(choice, "finish_reason", "stop") or "stop" for choice in response.choices],
            input=rendered_prompt if rendered_prompt is not None else context,
        )

    def _greedy_until_rwkv(self, docs):  # noqa: C901
        """Prototype: __call_api_parallel and native SampleCache.

        Harvest independent HTTP rollouts in completion order and fsync each
        before awaiting another. Original document/rollout order is restored
        for native metrics; vLLM's seed + child-index policy is unchanged.
        """
        outputs = [[None] * doc.num_samples for doc in docs]
        rollout_docs = [
            [
                replace(doc, id=doc.id if doc.num_samples == 1 else f"{doc.id}#rwkv-{index}", num_samples=1)
                for index in range(doc.num_samples)
            ]
            for doc in docs
        ]
        cached = {}
        if self._cache is not None:
            # The native cache reads each task's journal per call, so restore
            # all available rollouts together before submitting missing ones.
            all_rollouts = [rollout for group in rollout_docs for rollout in group]
            missing, task_ids = self._cache.get_samples_to_process_and_cache(all_rollouts, SamplingMethod.GENERATIVE)
            missing_keys = {(item.task_name, item.id) for item in missing}
            present = [item for item in all_rollouts if (item.task_name, item.id) not in missing_keys]
            if present:
                cached = dict(
                    zip(
                        ((item.task_name, item.id) for item in present),
                        self._cache.get_samples_from_cache(present, task_ids, SamplingMethod.GENERATIVE),
                        strict=True,
                    )
                )
        jobs, errors = {}, []
        with ThreadPoolExecutor(self.concurrent_requests) as executor:
            for doc_index, (doc, children) in enumerate(zip(docs, rollout_docs, strict=True)):
                context = self.prompt_manager.prepare_prompt_api(doc)
                for index, rollout_doc in enumerate(children):
                    key = (rollout_doc.task_name, rollout_doc.id)
                    if key in cached:
                        response = cached[key]
                        seed = self.generation_parameters.seed
                        expected_seed = seed + index if seed is not None else None
                        if (
                            len(response.text) != 1
                            or response.seeds != [expected_seed]
                            or len(response.finish_reasons) != 1
                            or response.rendered_prompt is None
                            or response.effective_max_new_tokens
                            != (self.generation_parameters.max_new_tokens or doc.generation_size)
                        ):
                            raise ValueError(f"unverifiable cached RWKV rollout {key}")
                        outputs[doc_index][index] = response
                        continue
                    future = executor.submit(
                        self.__call_api,
                        context,
                        doc.use_logits,
                        self.generation_parameters.max_new_tokens or doc.generation_size,
                        1,
                        doc.stop_sequences,
                        index,
                    )
                    jobs[future] = (doc_index, index, rollout_doc, context)
            for future in as_completed(jobs):
                doc_index, index, rollout_doc, context = jobs[future]
                try:
                    response = self._generation_response(future.result(), context)
                except CancelledError:
                    continue
                except Exception as error:
                    errors.append(error)
                    for queued in jobs:
                        queued.cancel()
                    continue
                if self._cache is not None:
                    self._cache.cache_sample(rollout_doc, response, SamplingMethod.GENERATIVE)
                outputs[doc_index][index] = response
        if errors:
            raise errors[0]
        results = []
        for responses in outputs:
            first = responses[0]
            results.append(
                ModelResponse(
                    text=[response.text[0] for response in responses],
                    input=first.input,
                    rendered_prompt=first.rendered_prompt,
                    input_tokens=first.input_tokens,
                    prompt_token_count=first.prompt_token_count,
                    effective_max_new_tokens=first.effective_max_new_tokens,
                    reasonings=[response.reasonings[0] for response in responses],
                    finish_reasons=[response.finish_reasons[0] for response in responses],
                    seeds=[response.seeds[0] for response in responses],
                )
            )
        return results

    def _prompt_logprobs_request(
        self, prompt: str | list[str], *, session: requests.Session | None = None
    ) -> list[dict[str, Any]]:
        """Request prompt token logprobs from the OpenAI-compatible completions API."""
        request_url = f"{self.base_url.rstrip('/')}/completions"
        LOGGER.info(
            "RWKV request: model=%s endpoint=%s request_type=LOGPROBS prompts=%d",
            self.model,
            request_url,
            len(prompt) if isinstance(prompt, list) else 1,
        )
        request_headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        request_id = uuid.uuid4().hex if self.model.startswith("rwkv7") else None
        if request_id:
            request_headers["X-Request-ID"] = request_id
        request_body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 0,
            "echo": True,
            # Sequence scoring needs only the observed prompt token, not top-k alternatives.
            "prompt_logprobs": 0,
            "return_token_ids": True,
        }
        # Prototype: __call_api. Retry transient transport failures as well
        # as upstream 5xx, without changing the scoring request or its budget.
        for attempt in range(6):
            started = time.monotonic()
            try:
                response = (session or requests).post(
                    request_url,
                    headers=request_headers,
                    json=request_body,
                    timeout=300.0 if self.model.startswith("rwkv7") else self.timeout or 120.0,
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
                LOGGER.warning(
                    "RWKV LOGPROBS transport failure: model=%s request_id=%s attempt=%d elapsed=%.2fs error=%s",
                    self.model,
                    request_id,
                    attempt + 1,
                    time.monotonic() - started,
                    type(error).__name__,
                )
                if attempt == 5:
                    raise
            else:
                elapsed = time.monotonic() - started
                if response.status_code >= 500 or elapsed >= 30:
                    LOGGER.warning(
                        "RWKV LOGPROBS response: model=%s request_id=%s gateway_call_id=%s attempt=%d elapsed=%.2fs status=%d",
                        self.model,
                        request_id,
                        response.headers.get("x-litellm-call-id"),
                        attempt + 1,
                        elapsed,
                        response.status_code,
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
                raise RuntimeError("RWKV prompt_logprobs did not return the selected candidate token")
            score += float(value["logprob"])
        return score

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        """Score each choice with native prompt logprobs; never generate a fallback answer."""
        # Prototype: _greedy_until_rwkv. Checkpoint each completed RWKV
        # sequence score before a slow/failed peer; restore document order.
        if self._cache is not None and self.model.startswith("rwkv7"):
            results = [None] * len(docs)
            errors = []
            with ThreadPoolExecutor(self.concurrent_requests) as executor:
                requests_to_process = iter(enumerate(docs))
                next_doc = next(requests_to_process, None)
                jobs = {}
                in_flight = 0
                while jobs or next_doc is not None:
                    while next_doc is not None:
                        index, doc = next_doc
                        # One candidate is one backend completion sequence.
                        # Oversized documents occupy the entire pool alone.
                        weight = max(1, min(len(doc.choices), self.concurrent_requests))
                        if in_flight + weight > self.concurrent_requests:
                            break
                        jobs[executor.submit(self._loglikelihood, doc)] = (index, weight)
                        in_flight += weight
                        next_doc = next(requests_to_process, None)
                    for future in as_completed(jobs):
                        index, weight = jobs.pop(future)
                        in_flight -= weight
                        try:
                            response = future.result()
                            self._cache.cache_sample(docs[index], response, SamplingMethod.LOGPROBS)
                            results[index] = response
                        except Exception as error:
                            errors.append(error)
                            # Drain and checkpoint the requests already in flight,
                            # but do not submit the rest after a failed request.
                            next_doc = None
                        # Refill immediately, not after a slow batch barrier.
                        break
            if errors:
                raise errors[0]
            return results
        # Native non-RWKV requests retain their existing batching behavior.
        with ThreadPoolExecutor(self.concurrent_requests) as executor:
            return list(executor.map(self._loglikelihood, docs))

    def _loglikelihood(self, doc: Doc) -> ModelResponse:
        if not isinstance(doc.choices, list) or not doc.choices:
            raise ValueError("RWKV loglikelihood requires a non-empty choice list")
        prefix = doc.query
        with requests.Session() if self.model.startswith("rwkv7") else nullcontext() as session:
            prefix_choice = self._prompt_logprobs_request(prefix, session=session)[0]
            prefix_token_ids = prefix_choice.get("prompt_token_ids")
            if not isinstance(prefix_token_ids, list):
                raise RuntimeError("RWKV prompt_logprobs response did not contain prefix token ids")
            prompts = [prefix + str(choice) for choice in doc.choices]
            candidate_choices = self._prompt_logprobs_request(prompts, session=session)
        if len(candidate_choices) != len(prompts):
            raise RuntimeError("RWKV prompt_logprobs response count did not match choices")
        scores = [self._candidate_logprob(choice, prefix_token_ids) for choice in candidate_choices]
        return ModelResponse(
            text=[""],
            input=prefix,
            # Preserve exact /v1/completions inputs; publication never recreates
            # options or an answer wrapper independently from these requests.
            candidate_inputs=prompts,
            input_tokens=prefix_token_ids,
            logprobs=scores,
            output_tokens=[[] for _ in scores],
            finish_reasons=["stop"],
        )
