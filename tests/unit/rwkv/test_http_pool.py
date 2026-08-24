import json
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from lighteval.models.rwkv import http_pool
from lighteval.models.rwkv.http_pool import CapacityScheduler, PoolError, PoolManifest, RWKVHttpPool


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
                "max_model_len": 10240,
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
    scheduler = CapacityScheduler([2, 1])
    assert [scheduler.acquire() for _ in range(3)] == [0, 1, 0]

    acquired = []
    ready = threading.Event()

    def acquire():
        acquired.append(scheduler.acquire())
        ready.set()

    thread = threading.Thread(target=acquire)
    thread.start()
    assert not ready.wait(0.05)
    scheduler.release(0)
    assert ready.wait(1)
    thread.join()
    assert acquired == [0]

    scheduler.release(0)
    scheduler.release(0)
    scheduler.release(1)
    assert scheduler.inflight == (0, 0)


def test_pool_preflights_every_replica_and_checks_manifest_identity(tmp_path, monkeypatch):
    gets = []

    class Client:
        def __init__(self, *, base_url, headers, **_kwargs):
            self.base_url = str(base_url)
            assert headers == {"Authorization": "Bearer secret"}

        def get(self, path):
            gets.append((self.base_url, path))
            return Response({}) if path == "/health" else Response({"data": [{"id": "rwkv-current"}]})

        def close(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    pool = RWKVHttpPool(_manifest(tmp_path), api_key="secret")

    assert pool.preflight() == "rwkv-current"
    assert len(gets) == 4
    assert pool.http_worker_limit == 4
    pool.close()

    mismatch = RWKVHttpPool(_manifest(tmp_path, served_model_name="different"), api_key="secret")
    with pytest.raises(PoolError, match="does not match manifest"):
        mismatch.preflight()
    mismatch.close()


def test_pool_uses_capacity_concurrently_and_preserves_completion_metadata(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    calls = []
    lock = threading.Lock()

    class Client:
        def __init__(self, *, base_url, **_kwargs):
            self.base_url = str(base_url)

        def get(self, path):
            return Response({}) if path == "/health" else Response({"data": [{"id": "rwkv-current"}]})

        def post(self, path, *, json):
            if path == "/detokenize":
                return Response({"prompt": "rendered"})
            with lock:
                calls.append(self.base_url)
                token = len(calls)
            barrier.wait(timeout=1)
            return _completion(token)

        def close(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    pool = RWKVHttpPool(_manifest(tmp_path))
    pool.preflight()

    with ThreadPoolExecutor(max_workers=2) as executor:
        completions = list(executor.map(lambda _: pool.complete([{"role": "user", "content": "q"}], {}), range(2)))

    assert set(calls) == {"http://10.0.0.1:8000", "http://10.0.0.2:8000"}
    assert [completion.prompt_text for completion in completions] == ["rendered", "rendered"]
    assert [completion.finish_reason for completion in completions] == ["stop", "stop"]
    assert [completion.terminal_token_id for completion in completions] == [0, 0]
    pool.close()


def test_pool_fails_over_only_for_retryable_failures(tmp_path, monkeypatch):
    posts = []

    class Client:
        def __init__(self, *, base_url, **_kwargs):
            self.base_url = str(base_url)

        def get(self, path):
            return Response({}) if path == "/health" else Response({"data": [{"id": "rwkv-current"}]})

        def post(self, path, *, json):
            posts.append((self.base_url, path))
            if path == "/detokenize":
                return Response({"prompt": "rendered"})
            if self.base_url.endswith("1:8000"):
                return Response({"error": "busy"}, status_code=503)
            return _completion(7)

        def close(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    pool = RWKVHttpPool(_manifest(tmp_path))
    pool.preflight()

    completion = pool.complete([{"role": "user", "content": "q"}], {})
    assert completion.text == "answer-7"
    assert posts[:2] == [
        ("http://10.0.0.1:8000", "/v1/chat/completions"),
        ("http://10.0.0.2:8000", "/v1/chat/completions"),
    ]
    pool.close()


def test_pool_fails_closed_on_schema_errors_without_trying_another_replica(tmp_path, monkeypatch):
    posts = []

    class Client:
        def __init__(self, *, base_url, **_kwargs):
            self.base_url = str(base_url)

        def get(self, path):
            return Response({}) if path == "/health" else Response({"data": [{"id": "rwkv-current"}]})

        def post(self, path, *, json):
            posts.append(self.base_url)
            return Response({"choices": []})

        def close(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    pool = RWKVHttpPool(_manifest(tmp_path))
    pool.preflight()

    with pytest.raises(PoolError, match="invalid completion"):
        pool.complete([{"role": "user", "content": "q"}], {})
    assert posts == ["http://10.0.0.1:8000"]
    pool.close()


def test_pool_rejects_missing_completion_content(tmp_path, monkeypatch):
    class Client:
        def __init__(self, **_kwargs):
            pass

        def get(self, path):
            return Response({}) if path == "/health" else Response({"data": [{"id": "rwkv-current"}]})

        def post(self, path, *, json):
            assert path == "/v1/chat/completions"
            return Response(
                {
                    "prompt_token_ids": [1],
                    "choices": [{"message": {}, "token_ids": [2], "finish_reason": "stop"}],
                }
            )

        def close(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    pool = RWKVHttpPool(_manifest(tmp_path))
    pool.preflight()

    with pytest.raises(PoolError, match="content is missing"):
        pool.complete([{"role": "user", "content": "q"}], {})
    pool.close()
