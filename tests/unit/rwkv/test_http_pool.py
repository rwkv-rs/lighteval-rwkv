import asyncio
import json

import httpx
import pytest

from lighteval.models.rwkv import http_pool
from lighteval.models.rwkv.http_pool import (
    CapacityScheduler,
    ContextLengthError,
    PoolError,
    PoolManifest,
    RWKVHttpPool,
)


def _manifest(tmp_path, replicas=None, served_model_name="rwkv-current"):
    path = tmp_path / "pool.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_name": "RWKV7-g1h-7.2B-20260710-ctx10240",
                "served_model_name": served_model_name,
                "model_revision": "weight-sha",
                "wkv_mode": "fp32io16",
                "vllm_version": "0.11.0",
                "torch_version": "2.10.0",
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                "max_model_len": 10240,
                "max_num_batched_tokens": 64,
                "replicas": replicas
                or [
                    {"base_url": "http://10.0.0.1:8000", "max_concurrency": 2},
                    {"base_url": "http://10.0.0.2:8000", "max_concurrency": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    return PoolManifest.read(path)


class Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)
        self.request = httpx.Request("GET", "http://test")

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            response = httpx.Response(self.status_code, text=self.text, request=self.request)
            raise httpx.HTTPStatusError("failure", request=self.request, response=response)


def _completion(token):
    return Response(
        {
            "prompt_text": "rendered",
            "prompt_token_ids": [1, 2],
            "choices": [
                {
                    "message": {"content": f"answer-{token}", "reasoning_content": "reason"},
                    "token_ids": [token],
                    "finish_reason": "stop",
                    "stop_reason": 0,
                }
            ],
        }
    )


def test_manifest_is_strict_and_fingerprint_is_stable(tmp_path):
    manifest = _manifest(tmp_path)

    assert manifest.aggregate_capacity == 3
    assert len(manifest.fingerprint) == 64
    assert manifest.fingerprint == _manifest(tmp_path).fingerprint

    raw = manifest.as_dict()
    raw["publish"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PoolError, match="unknown RWKV pool manifest fields: publish"):
        PoolManifest.read(path)


def test_scheduler_respects_heterogeneous_capacity_and_unblocks():
    async def run():
        scheduler = CapacityScheduler([2, 1])
        assert [await scheduler.acquire() for _ in range(3)] == [0, 1, 0]

        acquire = asyncio.create_task(scheduler.acquire())
        await asyncio.sleep(0)
        assert not acquire.done()
        await scheduler.release(0)
        assert await acquire == 0

        await scheduler.release(0)
        await scheduler.release(0)
        await scheduler.release(1)
        assert scheduler.inflight == (0, 0)

    asyncio.run(run())


def test_pool_preflights_every_replica_and_checks_manifest_identity(tmp_path, monkeypatch):
    gets = []

    class Client:
        def __init__(self, *, base_url, headers, **_kwargs):
            self.base_url = str(base_url)
            assert headers == {"Authorization": "Bearer secret"}

        def get(self, path):
            gets.append((self.base_url, path))
            return (
                Response({})
                if path == "/health"
                else Response({"data": [{"id": "another-model"}, {"id": "rwkv-current"}]})
            )

        def close(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    pool = RWKVHttpPool(_manifest(tmp_path), api_key="secret")

    assert pool.preflight() == "rwkv-current"
    assert len(gets) == 4
    assert pool.http_worker_limit == 4
    pool.close()

    mismatch = RWKVHttpPool(_manifest(tmp_path, served_model_name="different"), api_key="secret")
    with pytest.raises(PoolError, match="does not serve manifest model"):
        mismatch.preflight()
    mismatch.close()


def test_pool_uses_capacity_concurrently_and_preserves_completion_metadata(tmp_path, monkeypatch):
    calls = []

    class Client:
        def __init__(self, *, base_url, **_kwargs):
            self.base_url = str(base_url)

        def get(self, path):
            return Response({}) if path == "/health" else Response({"data": [{"id": "rwkv-current"}]})

        def close(self):
            pass

    class AsyncClient:
        def __init__(self, *, base_url, **_kwargs):
            self.base_url = str(base_url)

        async def post(self, path, *, json):
            assert path == "/v1/chat/completions"
            calls.append(self.base_url)
            token = len(calls)
            await asyncio.sleep(0)
            return _completion(token)

        async def aclose(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    monkeypatch.setattr(http_pool.httpx, "AsyncClient", AsyncClient)
    pool = RWKVHttpPool(_manifest(tmp_path))
    pool.preflight()

    async def run():
        completions = await asyncio.gather(*(pool.complete([{"role": "user", "content": "q"}], {}) for _ in range(2)))
        await pool.aclose()
        return completions

    completions = asyncio.run(run())

    assert set(calls) == {"http://10.0.0.1:8000", "http://10.0.0.2:8000"}
    assert [completion.prompt_text for completion in completions] == ["rendered", "rendered"]
    assert [completion.finish_reason for completion in completions] == ["stop", "stop"]
    assert [completion.terminal_token_id for completion in completions] == [0, 0]
    assert pool.peak_inflight == (1, 1)
    assert pool.inflight == (0, 0)


@pytest.mark.parametrize("failures_before_success", [1, 6])
def test_pool_fails_over_only_for_retryable_failures(tmp_path, monkeypatch, failures_before_success):  # noqa: C901
    posts = []
    delays = []

    class Client:
        def __init__(self, *, base_url, **_kwargs):
            self.base_url = str(base_url)

        def get(self, path):
            return Response({}) if path == "/health" else Response({"data": [{"id": "rwkv-current"}]})

        def close(self):
            pass

    class AsyncClient:
        def __init__(self, *, base_url, **_kwargs):
            self.base_url = str(base_url)

        async def post(self, path, *, json):
            posts.append((self.base_url, path))
            if len(posts) <= failures_before_success:
                return Response({"error": "busy"}, status_code=503)
            return _completion(7)

        async def aclose(self):
            pass

    async def no_sleep(_delay):
        assert pool.inflight == (0, 0)
        delays.append(_delay)

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    monkeypatch.setattr(http_pool.httpx, "AsyncClient", AsyncClient)
    monkeypatch.setattr(http_pool.asyncio, "sleep", no_sleep)
    pool = RWKVHttpPool(_manifest(tmp_path))
    pool.preflight()

    async def run():
        completion = await pool.complete([{"role": "user", "content": "q"}], {})
        await pool.aclose()
        return completion

    completion = asyncio.run(run())
    assert completion.text == "answer-7"
    assert posts[:2] == [
        ("http://10.0.0.1:8000", "/v1/chat/completions"),
        ("http://10.0.0.2:8000", "/v1/chat/completions"),
    ]
    assert delays == [float(2**attempt) for attempt in range(failures_before_success)]


def test_pool_retries_context_limited_completion_with_effective_output_limit(tmp_path, monkeypatch):
    posted_limits = []

    class Client:
        def __init__(self, **_kwargs):
            pass

        def get(self, path):
            return Response({}) if path == "/health" else Response({"data": [{"id": "rwkv-current"}]})

        def close(self):
            pass

    class AsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def post(self, _path, *, json):
            posted_limits.append(json["max_completion_tokens"])
            if len(posted_limits) <= 2:
                return Response(
                    {
                        "error": {
                            "message": (
                                "requested output tokens and your prompt contains at least "
                                f"{8999 + len(posted_limits)} input tokens"
                            )
                        }
                    },
                    status_code=400,
                )
            return _completion(7)

        async def aclose(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    monkeypatch.setattr(http_pool.httpx, "AsyncClient", AsyncClient)
    pool = RWKVHttpPool(_manifest(tmp_path))
    pool.preflight()

    async def run():
        completion = await pool.complete(
            [{"role": "user", "content": "q"}], {"max_completion_tokens": 2000}
        )
        await pool.aclose()
        return completion

    assert asyncio.run(run()).text == "answer-7"
    assert posted_limits == [2000, 1240, 1239]


def test_pool_reports_context_limit_when_prompt_cannot_fit(tmp_path, monkeypatch):
    class Client:
        def __init__(self, **_kwargs):
            pass

        def get(self, path):
            return Response({}) if path == "/health" else Response({"data": [{"id": "rwkv-current"}]})

        def close(self):
            pass

    class AsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def post(self, _path, *, json):
            return Response(
                {"error": {"message": "your prompt contains at least 10240 input tokens"}},
                status_code=400,
            )

        async def aclose(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    monkeypatch.setattr(http_pool.httpx, "AsyncClient", AsyncClient)
    pool = RWKVHttpPool(_manifest(tmp_path))
    pool.preflight()

    async def run():
        with pytest.raises(ContextLengthError, match="10240 input tokens"):
            await pool.complete([{"role": "user", "content": "q"}], {"max_completion_tokens": 2000})
        await pool.aclose()

    asyncio.run(run())


def test_pool_fails_closed_on_schema_errors_without_trying_another_replica(tmp_path, monkeypatch):
    posts = []

    class Client:
        def __init__(self, *, base_url, **_kwargs):
            self.base_url = str(base_url)

        def get(self, path):
            return Response({}) if path == "/health" else Response({"data": [{"id": "rwkv-current"}]})

        def close(self):
            pass

    class AsyncClient:
        def __init__(self, *, base_url, **_kwargs):
            self.base_url = str(base_url)

        async def post(self, path, *, json):
            posts.append(self.base_url)
            return Response({"choices": []})

        async def aclose(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    monkeypatch.setattr(http_pool.httpx, "AsyncClient", AsyncClient)
    pool = RWKVHttpPool(_manifest(tmp_path))
    pool.preflight()

    async def run():
        with pytest.raises(PoolError, match="invalid completion"):
            await pool.complete([{"role": "user", "content": "q"}], {})
        await pool.aclose()

    asyncio.run(run())
    assert posts == ["http://10.0.0.1:8000"]


def test_pool_rejects_missing_completion_content(tmp_path, monkeypatch):
    class Client:
        def __init__(self, **_kwargs):
            pass

        def get(self, path):
            return Response({}) if path == "/health" else Response({"data": [{"id": "rwkv-current"}]})

        def close(self):
            pass

    class AsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def post(self, path, *, json):
            assert path == "/v1/chat/completions"
            return Response(
                {
                    "prompt_token_ids": [1],
                    "choices": [{"message": {}, "token_ids": [2], "finish_reason": "stop"}],
                }
            )

        async def aclose(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    monkeypatch.setattr(http_pool.httpx, "AsyncClient", AsyncClient)
    pool = RWKVHttpPool(_manifest(tmp_path))
    pool.preflight()

    async def run():
        with pytest.raises(PoolError, match="content is missing"):
            await pool.complete([{"role": "user", "content": "q"}], {})
        await pool.aclose()

    asyncio.run(run())


def test_cancelled_request_releases_capacity_before_pool_close(tmp_path, monkeypatch):
    class Client:
        def __init__(self, **_kwargs):
            pass

        def get(self, path):
            return Response({}) if path == "/health" else Response({"data": [{"id": "rwkv-current"}]})

        def close(self):
            pass

    started = asyncio.Event()

    class AsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def post(self, _path, *, json):
            started.set()
            await asyncio.Future()

        async def aclose(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    monkeypatch.setattr(http_pool.httpx, "AsyncClient", AsyncClient)
    pool = RWKVHttpPool(_manifest(tmp_path))
    pool.preflight()

    async def run():
        request = asyncio.create_task(pool.complete([{"role": "user", "content": "q"}], {}))
        await started.wait()
        assert pool.inflight == (1, 0)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert pool.inflight == (0, 0)
        await pool.aclose()

    asyncio.run(run())
