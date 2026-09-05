# MIT License

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_input import GenerationParameters
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache

from .http_pool import Completion, ContextLengthError, PoolError, PoolManifest, RWKVHttpPool


logger = logging.getLogger(__name__)
MAX_NEW_TOKENS = 8192
REQUEST_CONTRACT_VERSION = "rwkv-generation-v2"
CACHE_POOL_FINGERPRINT = "transport-independent"
PROMPT_TEMPLATES: dict[str, tuple[str, str]] = {
    "bot": ("\nBot✿", "✿"),
    "assistant": ("\n\nAssistant: ", "\nUser:"),
    "function_calling": ("\n### Assistant", "\n### User"),
}
SAMPLING_PARAMETERS: dict[str, dict[str, object]] = {
    "open_think": {
        "temperature": 0.96,
        "top_p": 0.76,
        "top_k": 32,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.1,
        "penalty_decay": 0.988,
    },
    "fake_think": {
        "temperature": 1.0,
        "top_p": 0.28,
        "top_k": 32,
    },
}


class RWKVHTTPModelConfig(ModelConfig):
    """Result provenance for one immutable endpoint-pool evaluation."""

    served_model_name: str
    model_revision: str
    wkv_mode: str
    vllm_version: str
    max_model_length: int
    prompt_template: str
    cot_mode: str
    pool_fingerprint: str
    request_contract_version: str
    max_samples: int | None = None


@dataclass(frozen=True)
class _Job:
    document_index: int
    sample_index: int
    messages: list[dict[str, str]]
    parameters: dict[str, object]


class RWKVHttpModel(LightevalModel):
    """Generative LightEval adapter for an existing RWKV vLLM endpoint pool."""

    is_async = True

    def __init__(
        self,
        *,
        manifest: PoolManifest,
        prompt_template: str,
        cot_mode: str,
        cache_dir: Path,
        max_samples: int | None = None,
        api_key: str | None = None,
        pool: RWKVHttpPool | None = None,
    ) -> None:
        if prompt_template not in PROMPT_TEMPLATES:
            raise ValueError("unknown RWKV prompt template")
        if cot_mode not in SAMPLING_PARAMETERS:
            raise ValueError("unknown RWKV CoT mode")
        self.pool = pool or RWKVHttpPool(manifest, api_key=api_key)
        if pool is None:
            self.pool.preflight()
        else:
            _ = self.pool.model_id

        generation_parameters = SAMPLING_PARAMETERS[cot_mode]
        self.config = RWKVHTTPModelConfig(
            model_name=manifest.model_name,
            served_model_name=manifest.served_model_name,
            model_revision=manifest.model_revision,
            wkv_mode=manifest.wkv_mode,
            vllm_version=manifest.vllm_version,
            max_model_length=manifest.max_model_len,
            prompt_template=prompt_template,
            cot_mode=cot_mode,
            pool_fingerprint=manifest.fingerprint,
            request_contract_version=REQUEST_CONTRACT_VERSION,
            max_samples=max_samples,
            cache_dir=str(cache_dir),
            generation_parameters=GenerationParameters(
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=float(generation_parameters["temperature"]),
                top_p=float(generation_parameters["top_p"]),
                top_k=int(generation_parameters["top_k"]),
                presence_penalty=generation_parameters.get("presence_penalty"),
                frequency_penalty=generation_parameters.get("frequency_penalty"),
            ),
        )
        cache_config = self.config.model_copy(update={"pool_fingerprint": CACHE_POOL_FINGERPRINT})
        self._cache = SampleCache(cache_config)
        self.prompt_manager = PromptManager(use_chat_template=True, tokenizer=None)
        self._prompt_template = prompt_template
        self._assistant_prefix, self._template_stop = PROMPT_TEMPLATES[prompt_template]
        self._cot_mode = cot_mode
        self._generation_parameters = dict(generation_parameters)

    @property
    def tokenizer(self):
        return None

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        return self.pool.manifest.max_model_len

    async def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        if self._cache is None:
            return await self._generate(docs)
        task_ids = {self._cache.get_task_id(doc.task_name, SamplingMethod.GENERATIVE) for doc in docs}
        pending, _ = self._cache.get_samples_to_process_and_cache(docs, SamplingMethod.GENERATIVE)
        if pending:
            if os.environ.get("RWKV_EVAL_CACHE_ONLY") == "1":
                raise ValueError("cache-only evaluation found uncached RWKV rollouts")
            results = await self._generate(pending)
            self._cache.cache_samples(
                docs=pending,
                results=results,
                task_ids=task_ids,
                sampling_method=SamplingMethod.GENERATIVE,
            )
        results = list(self._cache.get_samples_from_cache(docs, task_ids, SamplingMethod.GENERATIVE))
        if any(result is None for result in results):
            raise ValueError("Problem while loading and aggregating items from cache.")
        return results

    def pending_rollouts(self, docs: list[Doc]) -> int:
        """Return the uncached rollout count used by the benchmark scheduler."""
        if self._cache is None:
            return sum(doc.num_samples for doc in docs)
        pending, _ = self._cache.get_samples_to_process_and_cache(docs, SamplingMethod.GENERATIVE)
        return sum(doc.num_samples for doc in pending)

    async def _generate(self, docs: list[Doc]) -> list[ModelResponse]:  # noqa: C901
        jobs: list[_Job] = []
        response_slots: list[list[Completion | None]] = []
        for document_index, doc in enumerate(docs):
            if doc.use_logits:
                raise ValueError("RWKV HTTP evaluation does not support generation logits")
            if not isinstance(doc.num_samples, int) or isinstance(doc.num_samples, bool) or doc.num_samples <= 0:
                raise ValueError("evaluation num_samples must be positive")

            messages = self.prompt_manager.prepare_prompt_api(doc)
            response_slots.append([None] * doc.num_samples)
            parameters = dict(self._generation_parameters)
            parameters.update(
                max_completion_tokens=self._completion_limit(doc),
                stop=self._stop_sequences(doc),
                chat_template_kwargs={
                    "rwkv_prompt_template": self._prompt_template,
                    "rwkv_generation_prompt": self._cot_mode,
                },
                ignore_eos=False,
                return_token_ids=True,
                return_prompt_text=True,
            )
            for sample_index in range(doc.num_samples):
                jobs.append(
                    _Job(
                        document_index=document_index,
                        sample_index=sample_index,
                        messages=messages,
                        parameters=parameters,
                    )
                )

        if jobs:
            await self.pool.start()

            async def execute(job: _Job) -> Completion:
                try:
                    return await self.pool.complete(job.messages, job.parameters)
                except ContextLengthError as error:
                    task_name = docs[job.document_index].task_name
                    logger.warning(
                        "RWKV context limit reached: task=%s document=%d rollout=%d error=%s",
                        task_name,
                        job.document_index,
                        job.sample_index,
                        error,
                    )
                    return Completion(
                        text="",
                        reasoning=None,
                        finish_reason="length",
                        stop_reason="context_length",
                        terminal_token_id=None,
                        prompt_text=job.messages[-1]["content"],
                        prompt_token_ids=(),
                        output_token_ids=(),
                    )
                except PoolError as error:
                    task_name = docs[job.document_index].task_name
                    raise PoolError(
                        f"{task_name} document {job.document_index} rollout {job.sample_index}: {error}"
                    ) from error

            requests = [asyncio.create_task(execute(job)) for job in jobs]
            try:
                completions = await asyncio.gather(*requests)
            except BaseException:
                for request in requests:
                    request.cancel()
                await asyncio.gather(*requests, return_exceptions=True)
                raise
            for job, completion in zip(jobs, completions):
                response_slots[job.document_index][job.sample_index] = completion

        responses: list[ModelResponse] = []
        for slots in response_slots:
            if any(completion is None for completion in slots):
                raise RuntimeError("RWKV HTTP evaluation returned incomplete samples")
            completions = [completion for completion in slots if completion is not None]
            prompt_text = completions[0].prompt_text
            prompt_tokens = completions[0].prompt_token_ids
            if any(
                completion.prompt_text != prompt_text or completion.prompt_token_ids != prompt_tokens
                for completion in completions
            ):
                raise RuntimeError("RWKV HTTP replicas rendered different model inputs")
            responses.append(
                ModelResponse(
                    input=prompt_text,
                    input_tokens=list(prompt_tokens),
                    text=[completion.text for completion in completions],
                    reasonings=[completion.reasoning for completion in completions],
                    finish_reasons=[completion.finish_reason for completion in completions],
                    stop_reasons=[completion.stop_reason for completion in completions],
                    terminal_token_ids=[completion.terminal_token_id for completion in completions],
                    output_tokens=[list(completion.output_token_ids) for completion in completions],
                    truncated_tokens_count=sum(completion.finish_reason == "length" for completion in completions),
                )
            )
        return responses

    @staticmethod
    def _completion_limit(doc: Doc) -> int:
        if (
            isinstance(doc.generation_size, int)
            and not isinstance(doc.generation_size, bool)
            and doc.generation_size > 0
        ):
            return min(doc.generation_size, MAX_NEW_TOKENS)
        return MAX_NEW_TOKENS

    def _stop_sequences(self, doc: Doc) -> list[str]:
        configured = [self._template_stop, *(doc.stop_sequences or [])]
        return list(dict.fromkeys(stop for stop in configured if isinstance(stop, str) and stop))

    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError("RWKV HTTP evaluation is generative only")

    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError("RWKV HTTP evaluation is generative only")

    def cleanup(self) -> None:
        self.pool.close()

    async def acleanup(self) -> None:
        await self.pool.aclose()
