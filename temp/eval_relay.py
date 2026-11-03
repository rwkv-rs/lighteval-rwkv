"""One-run evaluation relay using existing systemd and LiteLLM deployment APIs.

Run from the evaluation checkout whose SSH configuration reaches both hosts.
The producer CLI owns evaluation; this entry point only coordinates deployment.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import pwd
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


LOGGER = logging.getLogger("eval_relay")
SIZES = ("1.5b", "2.9b", "7.2b", "13.3b")
ROOT = Path(__file__).resolve().parents[1]


def run_command(command):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=45).stdout


def unit_properties(unit, *names):
    output = run_command(["timeout", "10s", "systemctl", "show", unit, *[f"--property={name}" for name in names]])
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def snapshot_producers(request):
    root = Path(request["root"])
    if request.get("require_automatic"):
        import tomllib

        if hashlib.sha256((root / "src/rwkv_eval/rwkv_model.py").read_bytes()).hexdigest() != request["adapter_hash"]:
            raise RuntimeError("deployed adapter does not match relay checkout")
        if any(
            item.get("max_num_seqs", 0) for item in tomllib.loads((root / "configs/models.toml").read_text())["model"]
        ):
            raise RuntimeError("deploy automatic concurrency before starting relay")
    units = run_command(
        ["timeout", "10s", "systemctl", "list-units", "--all", "--no-legend", "--plain", "rwkv-eval-*.service"]
    )
    result = {}
    for size in request["sizes"]:
        model = "rwkv7-g1l-" + size
        output = root / "results" / ("g1l-official-production-" + size)
        files = sorted((output / "failures" / (model + "-ctx32768-release")).glob("*.json"))
        report = json.loads(files[-1].read_text()) if files else None
        names = [
            line.split()[0]
            for line in units.splitlines()
            if line.split()[0].startswith("rwkv-eval-" + model + "-32768-release-")
        ]
        if len(names) > 1:
            raise RuntimeError("ambiguous producer units")
        item = {
            "report": report,
            "report_path": str(files[-1]) if files else None,
            "unit": None,
            "output": str(output),
        }
        if names:
            props = unit_properties(
                names[0], "Description", "MainPID", "ControlGroup", "ActiveState", "ExecMainStatus", "Result"
            )
            pid = int(props["MainPID"])
            item.update(unit=names[0], properties=props)
            if pid:
                args = Path("/proc/" + str(pid) + "/cmdline").read_bytes().decode().strip("\x00").split("\x00")
                cgroup = Path("/proc/" + str(pid) + "/cgroup").read_text()
                if (
                    "src.rwkv_eval.main" not in args
                    or str(output) not in args
                    or props["ControlGroup"] not in cgroup
                    or (not props["Description"].startswith("rwkv-eval:"))
                ):
                    raise RuntimeError("producer identity mismatch")
                item["args"] = args[args.index("src.rwkv_eval.main") + 1 :]
                if (
                    "--scoreboard" not in item["args"]
                    or item["args"][item["args"].index("--scoreboard") + 1] != "production"
                ):
                    raise RuntimeError("expected production producer")
                item["running"] = True
            else:
                item["running"] = False
        result[size] = item
    return result


def stop_producer(request):
    expected = request["producer"]
    props = unit_properties(expected["unit"], "Description", "MainPID", "ControlGroup")
    if (
        props["Description"] != expected["properties"]["Description"]
        or props["MainPID"] != expected["properties"]["MainPID"]
    ):
        raise RuntimeError("producer changed before stop")
    args = Path("/proc/" + props["MainPID"] + "/cmdline").read_bytes().decode().split("\x00")
    if "src.rwkv_eval.main" not in args or expected["output"] not in args:
        raise RuntimeError("producer ownership mismatch")
    run_command(["timeout", "40s", "systemctl", "stop", expected["unit"]])
    return {}


def drain_backend(opener, base_url):
    deadline = time.monotonic() + 300
    while True:
        with opener.open(base_url + "/metrics", timeout=15) as response:
            metrics = response.read().decode()
        counts = [
            float(line.split()[-1])
            for line in metrics.splitlines()
            if line.startswith(("irwkv:num_requests_running{", "irwkv:num_requests_waiting{"))
        ]
        if not counts:
            raise RuntimeError("extra-pool load metrics missing")
        count = sum(counts)
        if count == 0:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("extra-pool requests did not drain")
        time.sleep(5)


def wait_for_backend(opener, base_url, served_name, model, unit):
    deadline = time.monotonic() + 900
    while True:
        properties = unit_properties(unit, "ActiveState", "SubState")
        if properties["ActiveState"] == "failed" or properties["SubState"] == "auto-restart":
            subprocess.run(
                ["timeout", "60s", "systemctl", "stop", unit],
                check=True,
                timeout=65,
            )
            raise RuntimeError("new extra-pool instance crashed")
        try:
            with opener.open(base_url + "/health", timeout=15):
                pass
            with opener.open(base_url + "/v1/models", timeout=15) as response:
                models = json.load(response)["data"]
            if not any((item["id"] == served_name and item["root"] == model for item in models)):
                raise RuntimeError("extra-pool served model mismatch")
            break
        except (OSError, ValueError):
            if time.monotonic() >= deadline:
                raise RuntimeError("extra-pool startup deadline exceeded")
            time.sleep(5)


def switch_backend(request):
    size = request["size"]
    config = Path(request["backend_config_template"].format(size=size))
    data = json.loads(config.read_text())
    model = data["model"]
    if "g1l" not in model.lower() or size not in model.lower():
        raise RuntimeError("extra-pool weight identity mismatch")
    capacity = data["max-num-seqs"] * data.get("data-parallel-size", 1)
    base_url = "http://" + data["host"] + ":" + str(data["port"])
    units = run_command(
        [
            "timeout",
            "10s",
            "systemctl",
            "list-units",
            "--no-legend",
            "--plain",
            "--state=active",
            request["backend_unit_template"].format(size="*"),
        ]
    )
    active = [line.split()[0] for line in units.splitlines()]
    expected = request["backend_unit_template"].format(size=request["old_size"])
    if active != [expected]:
        raise RuntimeError("extra-pool service changed")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    credentials = dict(
        (
            word.split("=", 1)
            for line in Path(request["backend_environment_file"]).read_text().splitlines()
            for word in shlex.split(line, comments=True)
            if "=" in word
        )
    )
    opener.addheaders = [("Authorization", "Bearer " + credentials["IRWKV_API_KEY"])]
    drain_backend(opener, base_url)
    subprocess.run(["timeout", "60s", "systemctl", "stop", expected], check=True, timeout=65)
    subprocess.run(
        ["timeout", "60s", "systemctl", "start", request["backend_unit_template"].format(size=size)],
        check=True,
        timeout=65,
    )
    wait_for_backend(
        opener, base_url, data["served-model-name"], model, request["backend_unit_template"].format(size=size)
    )
    return {"capacity": capacity, "weight_file": os.path.basename(model)}


def add_deployment(api, models, extra, request):
    if extra:
        raise RuntimeError("extra endpoint already registered")
    name = "rwkv7-g1l-" + request["size"]
    resident = [item for item in models if item["model_name"] == name and not item["model_info"].get("blocked")]
    if len(resident) != 1:
        raise RuntimeError("ambiguous resident deployment")
    item = resident[0]
    params = dict(item["litellm_params"])
    info = item["model_info"]
    resident_capacity = params.get("max_parallel_requests") or info.get("max_parallel_requests")
    if not isinstance(resident_capacity, int) or resident_capacity <= 0:
        raise RuntimeError("missing resident capacity")
    params.update(
        api_base=request["extra_api_base"],
        max_parallel_requests=request["capacity"],
        weight=max(1, round(request["capacity"] / resident_capacity * params.get("weight", 1))),
    )
    api(
        "model/new",
        {
            "model_name": name,
            "litellm_params": params,
            "model_info": {
                key: info[key] for key in ("max_tokens", "max_input_tokens", "max_output_tokens") if key in info
            },
        },
    )
    matches = [
        item
        for item in api("model/info")["data"]
        if item["litellm_params"].get("api_base") == request["extra_api_base"]
    ]
    if (
        len(matches) != 1
        or matches[0]["model_name"] != name
        or matches[0]["litellm_params"].get("max_parallel_requests") != request["capacity"]
    ):
        raise RuntimeError("new extra deployment was not published")
    return {"id": matches[0]["model_info"]["id"], "model": name}


def gateway_request(request):
    config = Path(request["config"]).read_text()
    key = re.search(r"^\s*master_key:\s*([^\s#]+)", config, re.M).group(1).strip("\"'")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def api(path, body=None):
        query = urllib.request.Request(
            "http://127.0.0.1:4000/" + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        with opener.open(query, timeout=40) as response:
            return json.load(response)

    models = api("model/info")["data"]
    extra = [item for item in models if item["litellm_params"].get("api_base") == request["extra_api_base"]]
    if request["action"] == "info":
        if len(extra) != 1:
            raise RuntimeError("expected exactly one extra deployment")
        item = extra[0]
        return {"id": item["model_info"]["id"], "model": item["model_name"]}
    elif request["action"] == "remove":
        if (
            len(extra) != 1
            or extra[0]["model_info"]["id"] != request["id"]
            or extra[0]["model_name"] != request["model"]
        ):
            raise RuntimeError("extra deployment identity changed")
        api("model/delete", {"id": request["id"]})
        if any(item["model_info"]["id"] == request["id"] for item in api("model/info")["data"]):
            raise RuntimeError("extra deployment still registered")
        return {}
    elif request["action"] == "add":
        return add_deployment(api, models, extra, request)


def remote(host, worker, request, timeout=75):
    source = Path(__file__).read_text()
    script = str(len(source)) + "\n" + source + json.dumps(request)
    # Stream this module and its JSON request over SSH without remote installation.
    bootstrap = (
        "import sys; size=int(sys.stdin.readline()); "
        "exec(compile(sys.stdin.read(size), 'eval_relay.py', 'exec'), "
        "{'__name__':'__main__','__file__':'eval_relay.py'})"
    )
    process = subprocess.run(
        [
            "timeout",
            f"{timeout}s",
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            "sudo",
            "-n",
            "timeout",
            f"{timeout - 10}s",
            "python3",
            "-c",
            shlex.quote(bootstrap),
            "--worker",
            worker,
        ],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout + 5,
    )
    if process.returncode:
        # Never forward complete API bodies or remote environment diagnostics.
        detail = process.stderr.strip().splitlines()[-1] if process.stderr.strip() else "no diagnostic"
        raise RuntimeError(f"{request['action']} failed on {host}; exit={process.returncode}: {detail[:500]}")
    return json.loads(process.stdout)


def snapshot(args):
    return remote(
        args.eval_host,
        "evaluation",
        {
            "action": "snapshot",
            "root": args.root,
            "sizes": SIZES,
            "require_automatic": not args.plan,
            "adapter_hash": hashlib.sha256((ROOT / "src/rwkv_eval/rwkv_model.py").read_bytes()).hexdigest(),
        },
    )


def completed(item):
    report = item["report"]
    if report is None:
        raise RuntimeError("missing evaluation report")
    if report["status"] in {"stopped", "completed_with_failures"} or report["failures"] or report["failed_benchmarks"]:
        raise RuntimeError(f"{report['model_name']} requires manual error handling: {item['report_path']}")
    if item.get("running"):
        return False
    if item.get("properties", {}).get("Result", "success") != "success":
        raise RuntimeError(f"producer failed: {item['unit']}")
    if report["status"] != "completed" or report["not_executed_benchmarks"] or report["finished_at"] is None:
        raise RuntimeError(f"producer stopped without completion: {item['report_path']}")
    return True


def persist(path, state):
    """Prototype: atomic result/report serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(state, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def switch(args, state, target, pools):
    gateway = {"extra_api_base": args.extra_api_base, "config": args.gateway_config}
    old = remote(args.gateway_host, "gateway", {**gateway, "action": "info"})
    old_size = old["model"].removeprefix("rwkv7-g1l-")
    if old_size not in SIZES:
        raise RuntimeError("unexpected extra-pool model identity")
    if old_size == target:
        state["current"] = target
        return
    affected = {size: pools[size] for size in {old_size, target} if pools[size].get("running")}
    state.update(phase="stopping_producers", target=target, previous=old_size)
    state["affected_producers"] = {
        size: {key: item[key] for key in ("unit", "args", "output", "report_path")} for size, item in affected.items()
    }
    persist(args.state, state)
    for size, producer in affected.items():
        remote(args.eval_host, "evaluation", {"action": "stop", "root": args.root, "producer": producer})
    state["phase"] = "removing_deployment"
    persist(args.state, state)
    remote(args.gateway_host, "gateway", {**gateway, "action": "remove", **old})
    state["phase"] = "switching_backend"
    persist(args.state, state)
    backend = remote(
        args.eval_host,
        "evaluation",
        {
            "action": "backend",
            "root": args.root,
            "size": target,
            "old_size": old_size,
            "backend_config_template": args.backend_config_template,
            "backend_unit_template": args.backend_unit_template,
            "backend_environment_file": args.backend_environment_file,
        },
        timeout=1400,
    )
    state["phase"] = "registering_deployment"
    persist(args.state, state)
    remote(args.gateway_host, "gateway", {**gateway, "action": "add", "size": target, "capacity": backend["capacity"]})
    state["phase"] = "resuming_producers"
    persist(args.state, state)
    for size, producer in affected.items():
        remote(
            args.eval_host,
            "evaluation",
            {"action": "resume", "root": args.root, "producer": producer, "service_user": args.service_user},
            timeout=85,
        )
    state.update(phase="monitoring", current=target)
    state["transitions"].append(
        {"from": old_size, "to": target, "capacity": backend["capacity"], "finished_at": time.time()}
    )
    persist(args.state, state)
    LOGGER.info("extra pool switched %s -> %s; affected producers resumed from cache", old_size, target)


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-host", required=True)
    parser.add_argument("--gateway-host", required=True)
    parser.add_argument("--gateway-config", required=True)
    parser.add_argument("--backend-config-template", required=True)
    parser.add_argument("--backend-unit-template", required=True)
    parser.add_argument("--backend-environment-file", required=True)
    parser.add_argument("--service-user", default=pwd.getpwuid(os.getuid()).pw_name)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--extra-api-base", required=True)
    parser.add_argument("--state", type=Path, default=ROOT / "results/extra-pool-relay.json")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    snapshot_request = {
        "action": "snapshot",
        "root": args.root,
        "sizes": SIZES,
        "require_automatic": not args.plan,
        "adapter_hash": hashlib.sha256((ROOT / "src/rwkv_eval/rwkv_model.py").read_bytes()).hexdigest(),
    }
    if args.plan:
        pools = remote(args.eval_host, "evaluation", snapshot_request)
        print(
            json.dumps(
                {
                    size: {"completed": completed(item), "report": item["report_path"], "unit": item["unit"]}
                    for size, item in pools.items()
                },
                indent=2,
            )
        )
        return 0
    if args.launch:
        command = [
            "timeout",
            "30s",
            "systemd-run",
            "--user",
            "--unit=rwkv-eval-extra-pool-relay",
            "--collect",
            "--property=Restart=no",
            f"--property=WorkingDirectory={ROOT}",
            str(ROOT / ".venv/bin/python"),
            str(Path(__file__).resolve()),
            *[item for item in sys.argv[1:] if item != "--launch"],
        ]
        subprocess.run(command, check=True, timeout=35)
        return 0
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.state.exists():
            state = json.loads(args.state.read_text())
            if state["status"] == "completed":
                return 0
            if state["status"] != "running" or state["phase"] != "monitoring":
                raise RuntimeError("interrupted transition requires manual inspection before resuming")
        else:
            state = {
                "status": "running",
                "phase": "monitoring",
                "started_at": time.time(),
                "current": None,
                "completed": [],
                "transitions": [],
            }
        try:
            while True:
                pools = remote(args.eval_host, "evaluation", snapshot_request)
                state["reports"] = {size: item["report_path"] for size, item in pools.items()}
                done = [size for size in SIZES if completed(pools[size])]
                if any(size not in done for size in state["completed"]):
                    raise RuntimeError("completed evaluation scope changed during relay")
                state["completed"] = done
                target = next((size for size in SIZES if size not in done), None)
                if target is None:
                    state.update(status="completed", phase="completed", finished_at=time.time())
                    persist(args.state, state)
                    LOGGER.info("all model pools completed; relay finished")
                    return 0
                switch(args, state, target, pools)
                persist(args.state, state)
                time.sleep(args.interval)
        except Exception as error:
            state.update(status="stopped", error_type=type(error).__name__, error=str(error), finished_at=time.time())
            persist(args.state, state)
            LOGGER.exception("relay stopped; inspect %s", args.state)
            return 78


def worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("evaluation", "gateway"), required=True)
    args = parser.parse_args()
    request = json.load(sys.stdin)
    if args.worker == "gateway":
        result = gateway_request(request)
    elif request["action"] == "resume":
        root = Path(request["root"])
        args = request["producer"]["args"]
        command = [
            "timeout",
            "60s",
            "sudo",
            "-H",
            "-u",
            request["service_user"],
            str(root / ".venv/bin/python"),
            "-m",
            "src.rwkv_eval.main",
            "--launch",
            "--offline",
            "--service-user",
            request["service_user"],
            *args,
        ]
        subprocess.run(command, check=True, cwd=root, stdout=subprocess.DEVNULL, timeout=65)
        result = {}
    else:
        handlers = {
            "snapshot": snapshot_producers,
            "stop": stop_producer,
            "backend": switch_backend,
        }
        result = handlers[request["action"]](request)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(worker_main() if sys.argv[1:2] == ["--worker"] else main())
