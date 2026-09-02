import signal
from types import SimpleNamespace

import pytest

from temp import evaluation_runner


def _args(tmp_path, models="1.5b,2.9b,7.2b,13.3b"):
    return SimpleNamespace(
        models=models,
        manifest_1_5b=tmp_path / "1.5b.json",
        manifest_2_9b=tmp_path / "2.9b.json",
        manifest_7_2b=tmp_path / "7.2b.json",
        manifest_13_3b=tmp_path / "13.3b.json",
    )


def test_evaluation_runner_selects_only_requested_models(tmp_path):
    assert [evaluation.size for evaluation in evaluation_runner._evaluations(_args(tmp_path, "2.9b,13.3b"))] == [
        "2.9B",
        "13.3B",
    ]


@pytest.mark.parametrize("models", ["", "1.5b,", "unknown", "1.5b,1.5b"])
def test_evaluation_runner_rejects_invalid_model_selection(tmp_path, models):
    with pytest.raises(ValueError):
        evaluation_runner._evaluations(_args(tmp_path, models))


def test_evaluation_runner_validates_manifest_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        evaluation_runner.PoolManifest,
        "read",
        lambda _path: SimpleNamespace(aggregate_capacity=1),
    )

    with pytest.raises(ValueError, match="pool capacity must be 1024"):
        evaluation_runner._validate(evaluation_runner._evaluations(_args(tmp_path, "1.5b")))


def test_four_model_evaluation_runner_forwards_signals_to_every_process(tmp_path, monkeypatch):
    handlers = {}
    processes = []

    class Process:
        def __init__(self, command, *, cwd, env):
            self.command = command
            self.cwd = cwd
            self.env = env
            self.done = False
            self.signals = []
            processes.append(self)

        def poll(self):
            return 0 if self.done else None

        def send_signal(self, signum):
            self.signals.append(signum)

        def wait(self):
            if self is processes[0]:
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            self.done = True
            return 0

    monkeypatch.setattr(evaluation_runner, "_validate", lambda _evaluations: None)
    monkeypatch.setattr(
        evaluation_runner.signal, "signal", lambda signum, handler: handlers.setdefault(signum, handler)
    )
    monkeypatch.setattr(evaluation_runner.subprocess, "Popen", Process)
    manifests = [tmp_path / f"{size}.json" for size in ("1.5b", "2.9b", "7.2b", "13.3b")]

    result = evaluation_runner.main(
        [
            "--manifest-1.5b",
            str(manifests[0]),
            "--manifest-2.9b",
            str(manifests[1]),
            "--manifest-7.2b",
            str(manifests[2]),
            "--manifest-13.3b",
            str(manifests[3]),
        ]
    )

    assert result == 128 + signal.SIGTERM
    assert len(processes) == 4
    assert all(process.command[-2:] == ["--max-samples", "10"] for process in processes)
    assert [process.env["RWKV_EVAL_POOL_MANIFEST"] for process in processes] == [
        str(manifest.resolve()) for manifest in manifests
    ]
    assert all(process.signals == [signal.SIGTERM] for process in processes)
