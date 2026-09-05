# MIT License

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Mapping, Sequence
from urllib.parse import urlsplit

import httpx


RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
API_MAX_RETRY = 8
API_RETRY_SLEEP = 1.0
API_RETRY_MULTIPLIER = 2.0
logger = logging.getLogger(__name__)
_CONTEXT_LENGTH_ERROR = re.compile(r"prompt contains at least (\d+) input tokens")


class PoolError(RuntimeError):
    """Raised when a configured RWKV endpoint pool cannot be used safely."""


class ContextLengthError(PoolError):
    """Raised when a completion cannot fit after reducing its output limit."""


@dataclass(frozen=True)
class Replica:
    """One OpenAI-compatible endpoint and its declared request capacity."""

    base_url: str
    max_concurrency: int

    def as_dict(self) -> dict[str, object]:
        return {
            "base_url": self.base_url,
            "max_concurrency": self.max_concurrency,
        }


@dataclass(frozen=True)
class PoolManifest:
    """Evaluator-facing identity and capacity contract for an existing pool."""

    model_name: str
    served_model_name: str
    model_revision: str
    wkv_mode: str
    vllm_version: str
    torch_version: str
    gpu: str
    max_model_len: int
    max_num_batched_tokens: int
    replicas: tuple[Replica, ...]

    @classmethod
    def read(cls, path: Path) -> PoolManifest:
        try:
            with path.open(encoding="utf-8") as stream:
                raw = json.load(stream)
        except FileNotFoundError as error:
            raise PoolError(f"RWKV pool manifest not found: {path}") from error
        except (OSError, json.JSONDecodeError) as error:
            raise PoolError(f"invalid RWKV pool manifest: {error}") from error
        if not isinstance(raw, dict):
            raise PoolError("RWKV pool manifest must be a JSON object")

        allowed = {
            "schema_version",
            "model_name",
            "served_model_name",
            "model_revision",
            "wkv_mode",
            "vllm_version",
            "torch_version",
            "gpu",
            "max_model_len",
            "max_num_batched_tokens",
            "replicas",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise PoolError("unknown RWKV pool manifest fields: " + ", ".join(unknown))
        missing = sorted(allowed - set(raw))
        if missing:
            raise PoolError("missing RWKV pool manifest fields: " + ", ".join(missing))
        if raw["schema_version"] != 1 or isinstance(raw["schema_version"], bool):
            raise PoolError("RWKV pool manifest schema_version must be 1")

        model_name = cls._nonempty_string(raw["model_name"], "model_name")
        served_model_name = cls._nonempty_string(raw["served_model_name"], "served_model_name")
        model_revision = cls._nonempty_string(raw["model_revision"], "model_revision")
        wkv_mode = raw["wkv_mode"]
        if wkv_mode not in {"fp16", "fp32io16"}:
            raise PoolError("RWKV pool manifest wkv_mode must be fp16 or fp32io16")
        vllm_version = cls._nonempty_string(raw["vllm_version"], "vllm_version")
        torch_version = cls._nonempty_string(raw["torch_version"], "torch_version")
        gpu = cls._nonempty_string(raw["gpu"], "gpu")
        max_model_len = cls._positive_int(raw["max_model_len"], "max_model_len")
        max_num_batched_tokens = cls._positive_int(raw["max_num_batched_tokens"], "max_num_batched_tokens")

        configured_replicas = raw["replicas"]
        if not isinstance(configured_replicas, list) or not configured_replicas:
            raise PoolError("RWKV pool manifest replicas must be a non-empty array")
        replicas = tuple(cls._read_replica(value, index) for index, value in enumerate(configured_replicas))
        urls = [replica.base_url for replica in replicas]
        duplicates = sorted(url for url in set(urls) if urls.count(url) > 1)
        if duplicates:
            raise PoolError("duplicate RWKV replica base_url: " + ", ".join(duplicates))

        return cls(
            model_name=model_name,
            served_model_name=served_model_name,
            model_revision=model_revision,
            wkv_mode=wkv_mode,
            vllm_version=vllm_version,
            torch_version=torch_version,
            gpu=gpu,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            replicas=replicas,
        )

    @classmethod
    def _read_replica(cls, raw: object, index: int) -> Replica:
        if not isinstance(raw, dict):
            raise PoolError(f"RWKV pool replica {index} must be a JSON object")
        allowed = {"base_url", "max_concurrency"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise PoolError(f"unknown RWKV pool replica {index} fields: " + ", ".join(unknown))
        missing = sorted(allowed - set(raw))
        if missing:
            raise PoolError(f"missing RWKV pool replica {index} fields: " + ", ".join(missing))

        base_url = raw["base_url"]
        if not isinstance(base_url, str) or base_url != base_url.strip():
            raise PoolError(f"RWKV pool replica {index} has an invalid base_url")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise PoolError(f"RWKV pool replica {index} base_url must be an HTTP(S) origin")
        return Replica(
            base_url=base_url.rstrip("/"),
            max_concurrency=cls._positive_int(raw["max_concurrency"], f"replica {index} max_concurrency"),
        )

    @staticmethod
    def _nonempty_string(value: object, name: str) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise PoolError(f"RWKV pool manifest {name} must be a non-empty string")
        return value

    @staticmethod
    def _positive_int(value: object, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise PoolError(f"RWKV pool manifest {name} must be a positive integer")
        return value

    @property
    def aggregate_capacity(self) -> int:
        return sum(replica.max_concurrency for replica in self.replicas)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model_name": self.model_name,
            "served_model_name": self.served_model_name,
            "model_revision": self.model_revision,
            "wkv_mode": self.wkv_mode,
            "vllm_version": self.vllm_version,
            "max_model_len": self.max_model_len,
            "replicas": [replica.as_dict() for replica in self.replicas],
        }

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class CapacityScheduler:
    """Reserve replica slots using least normalized in-flight load."""

    def __init__(self, capacities: Sequence[int]) -> None:
        if not capacities or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in capacities
        ):
            raise ValueError("capacities must contain positive integers")
        self._capacities = tuple(capacities)
        self._inflight = [0] * len(capacities)
        self._peak_inflight = [0] * len(capacities)
        self._condition = asyncio.Condition()

    async def acquire(self, excluded: frozenset[int] = frozenset()) -> int:
        eligible = [index for index in range(len(self._capacities)) if index not in excluded]
        if not eligible:
            raise PoolError("all RWKV replicas have already failed this request")
        async with self._condition:
            while True:
                available = [index for index in eligible if self._inflight[index] < self._capacities[index]]
                if available:
                    selected = min(
                        available,
                        key=lambda index: (
                            self._inflight[index] / self._capacities[index],
                            self._inflight[index],
                            index,
                        ),
                    )
                    self._inflight[selected] += 1
                    self._peak_inflight[selected] = max(self._peak_inflight[selected], self._inflight[selected])
                    return selected
                await self._condition.wait()

    async def release(self, index: int) -> None:
        async with self._condition:
            if not 0 <= index < len(self._inflight):
                raise IndexError("replica index is outside the scheduler")
            if self._inflight[index] <= 0:
                raise RuntimeError("replica does not have an in-flight reservation")
            self._inflight[index] -= 1
            self._condition.notify_all()

    @asynccontextmanager
    async def lease(self, excluded: frozenset[int] = frozenset()) -> AsyncIterator[int]:
        index = await self.acquire(excluded)
        try:
            yield index
        finally:
            await self.release(index)

    @property
    def inflight(self) -> tuple[int, ...]:
        return tuple(self._inflight)

    @property
    def peak_inflight(self) -> tuple[int, ...]:
        return tuple(self._peak_inflight)


@dataclass(frozen=True)
class Completion:
    text: str
    reasoning: str | None
    finish_reason: str
    stop_reason: str | None
    terminal_token_id: int | None
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]


class RWKVHttpPool:
    """Capacity-aware client for one already-deployed RWKV endpoint pool."""

    def __init__(self, manifest: PoolManifest, api_key: str | None = None) -> None:
        self.manifest = manifest
        self._scheduler = CapacityScheduler([replica.max_concurrency for replica in manifest.replicas])
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._clients: tuple[httpx.AsyncClient, ...] | None = None
        self._model_id: str | None = None
        self._first_request_at: float | None = None

    def preflight(self) -> str:
        with ThreadPoolExecutor(max_workers=len(self.manifest.replicas)) as executor:
            model_ids = list(executor.map(self._preflight_replica, range(len(self.manifest.replicas))))
        if len(set(model_ids)) != 1:
            raise PoolError("RWKV replicas do not serve one model: " + ", ".join(model_ids))
        model_id = model_ids[0]
        if model_id != self.manifest.served_model_name:
            raise PoolError(
                "RWKV pool served model does not match manifest: "
                f"expected {self.manifest.served_model_name}, found {model_id}"
            )
        self._model_id = model_id
        return model_id

    def _preflight_replica(self, index: int) -> str:
        replica = self.manifest.replicas[index]
        client = httpx.Client(
            base_url=replica.base_url,
            headers=self._headers,
            timeout=httpx.Timeout(30.0),
            trust_env=False,
        )
        try:
            health = client.get("/health")
            health.raise_for_status()
            response = client.get("/v1/models")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise PoolError(f"RWKV replica preflight failed for {replica.base_url}: {error}") from error
        finally:
            client.close()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or not data:
            raise PoolError(f"RWKV replica {replica.base_url} returned an invalid model list")
        model_ids = [entry.get("id") for entry in data if isinstance(entry, dict)]
        if len(model_ids) != len(data) or not all(isinstance(model_id, str) and model_id for model_id in model_ids):
            raise PoolError(f"RWKV replica {replica.base_url} returned an invalid model list")
        if self.manifest.served_model_name not in model_ids:
            raise PoolError(
                f"RWKV replica {replica.base_url} does not serve manifest model {self.manifest.served_model_name}"
            )
        return self.manifest.served_model_name

    async def start(self) -> None:
        if self._clients is not None:
            return
        self._clients = tuple(
            httpx.AsyncClient(
                base_url=replica.base_url,
                headers=self._headers,
                timeout=httpx.Timeout(3600.0, connect=30.0),
                limits=httpx.Limits(
                    max_connections=replica.max_concurrency,
                    max_keepalive_connections=replica.max_concurrency,
                ),
                trust_env=False,
            )
            for replica in self.manifest.replicas
        )

    async def complete(  # noqa: C901
        self, messages: list[dict[str, str]], parameters: Mapping[str, object]
    ) -> Completion:
        if self._model_id is None:
            raise PoolError("RWKV pool must pass preflight before evaluation")
        if not isinstance(messages, list):
            raise PoolError("RWKV chat completion requires messages")
        if self._first_request_at is None:
            self._first_request_at = time.monotonic()
        await self.start()
        assert self._clients is not None

        attempted: set[int] = set()
        failures: list[str] = []
        context_failure: str | None = None
        for attempt in range(API_MAX_RETRY):
            if len(attempted) == len(self._clients):
                attempted.clear()
            async with self._scheduler.lease(frozenset(attempted)) as index:
                attempted.add(index)
                try:
                    return await self._complete_on(index, messages, parameters)
                except httpx.HTTPStatusError as error:
                    status = error.response.status_code
                    adjusted = self._context_limited_parameters(error, parameters)
                    if adjusted is not None:
                        context_failure = error.response.text[:500]
                        parameters = adjusted
                        continue
                    if self._context_error_message(error) is not None:
                        raise ContextLengthError(error.response.text[:500]) from error
                    if status not in RETRYABLE_STATUS_CODES:
                        detail = error.response.text[:500]
                        raise PoolError(f"RWKV endpoint rejected completion with HTTP {status}: {detail}") from error
                    failures.append(f"{self.manifest.replicas[index].base_url}: HTTP {status}")
                except httpx.RequestError as error:
                    failures.append(f"{self.manifest.replicas[index].base_url}: {type(error).__name__}: {error}")
            if attempt < API_MAX_RETRY - 1:
                wait_time = min(64, API_RETRY_SLEEP * (API_RETRY_MULTIPLIER**attempt))
                logger.warning(
                    "RWKV completion transport retry: attempt=%d/%d wait_seconds=%s error=%s",
                    attempt + 1,
                    API_MAX_RETRY,
                    wait_time,
                    failures[-1],
                )
                await asyncio.sleep(wait_time)
        if context_failure is not None and not failures:
            raise ContextLengthError(context_failure)
        raise PoolError("RWKV completion failed: " + "; ".join(failures))

    @staticmethod
    def _context_error_message(error: httpx.HTTPStatusError) -> str | None:
        if error.response.status_code != 400:
            return None
        try:
            message = error.response.json()["error"]["message"]
        except (KeyError, TypeError, ValueError):
            return None
        return message if isinstance(message, str) and _CONTEXT_LENGTH_ERROR.search(message) else None

    def _context_limited_parameters(
        self, error: httpx.HTTPStatusError, parameters: Mapping[str, object]
    ) -> dict[str, object] | None:
        message = self._context_error_message(error)
        match = _CONTEXT_LENGTH_ERROR.search(message) if message is not None else None
        requested = parameters.get("max_completion_tokens")
        if match is None or isinstance(requested, bool) or not isinstance(requested, int):
            return None
        available = self.manifest.max_model_len - int(match.group(1))
        if available <= 0 or available >= requested:
            return None
        adjusted = dict(parameters)
        adjusted["max_completion_tokens"] = available
        logger.info(
            "RWKV completion limited by context: prompt_tokens=%s requested=%s effective=%s",
            match.group(1),
            requested,
            available,
        )
        return adjusted

    async def _complete_on(  # noqa: C901
        self,
        index: int,
        messages: list[dict[str, str]],
        parameters: Mapping[str, object],
    ) -> Completion:
        payload = dict(parameters)
        payload.update(model=self.model_id, messages=messages, n=1, stream=False)
        assert self._clients is not None
        response = await self._clients[index].post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        try:
            raw = response.json()
            choices = raw["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("completion must return exactly one choice")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise ValueError("completion choice must be an object")
            message = choice["message"]
            if not isinstance(message, dict):
                raise ValueError("completion message must be an object")
            text = message.get("content")
            reasoning = message.get("reasoning_content")
            if not isinstance(text, str):
                raise ValueError("completion content is missing or invalid")
            if reasoning is not None and not isinstance(reasoning, str):
                raise ValueError("completion reasoning_content must be text")
            prompt_text = raw.get("prompt_text")
            if not isinstance(prompt_text, str):
                raise ValueError("completion prompt_text is missing or invalid")
            prompt_token_ids = self._token_ids(raw.get("prompt_token_ids"), "prompt")
            output_token_ids = self._token_ids(choice.get("token_ids"), "output")
            finish_reason = choice.get("finish_reason")
            if not isinstance(finish_reason, str) or not finish_reason:
                raise ValueError("completion finish_reason is missing or invalid")
            raw_stop_reason = choice.get("stop_reason")
            if raw_stop_reason is not None and (
                isinstance(raw_stop_reason, bool) or not isinstance(raw_stop_reason, (str, int))
            ):
                raise ValueError("completion stop_reason must be text or a token id")
        except (KeyError, TypeError, ValueError) as error:
            raise PoolError(f"RWKV endpoint returned an invalid completion: {error}") from error

        terminal_token_id = (
            raw_stop_reason
            if isinstance(raw_stop_reason, int) and not isinstance(raw_stop_reason, bool)
            else (output_token_ids[-1] if output_token_ids else None)
        )
        stop_reason = str(raw_stop_reason) if raw_stop_reason is not None else None

        return Completion(
            text=text,
            reasoning=reasoning,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
            terminal_token_id=terminal_token_id,
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            output_token_ids=output_token_ids,
        )

    @staticmethod
    def _token_ids(value: object, name: str) -> tuple[int, ...]:
        if not isinstance(value, list) or any(
            not isinstance(token, int) or isinstance(token, bool) for token in value
        ):
            raise ValueError(f"completion {name} token ids are missing or invalid")
        return tuple(value)

    @property
    def model_id(self) -> str:
        if self._model_id is None:
            raise PoolError("RWKV pool has not completed preflight")
        return self._model_id

    @property
    def aggregate_capacity(self) -> int:
        return self.manifest.aggregate_capacity

    @property
    def http_worker_limit(self) -> int:
        return self.aggregate_capacity + 1

    @property
    def inflight(self) -> tuple[int, ...]:
        return self._scheduler.inflight

    @property
    def peak_inflight(self) -> tuple[int, ...]:
        return self._scheduler.peak_inflight

    @property
    def first_request_at(self) -> float | None:
        return self._first_request_at

    def close(self) -> None:
        if self._clients is not None:
            raise PoolError("RWKV async pool must be closed from its event loop")

    async def aclose(self) -> None:
        clients, self._clients = self._clients, None
        if clients is not None:
            await asyncio.gather(*(client.aclose() for client in clients))
        if any(self.inflight):
            raise RuntimeError("RWKV pool closed with active requests")
        logger.info(
            "RWKV pool peak in-flight: observed=%s capacity=%s",
            self.peak_inflight,
            tuple(replica.max_concurrency for replica in self.manifest.replicas),
        )
