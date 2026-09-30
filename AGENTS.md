## Core Objectives
LightEval is the mainstream evaluation library in the LLM community. This repository needs to integrate RWKV models through LightEval's native pipeline, model adapter, and task interfaces.
Correspondence Principle: For every file/type/function/variable, a similar implementation must be found as a prototype. If the prototype carries a model name, replace it with `RWKV` or other case variants; otherwise keep the same name.
Non-Interference Principle: The changes we introduce should not break LightEval's native evaluation pipeline, and should not affect the evaluation results of other models.

## Evaluation Producer Contract

- Evaluation mode is executable behavior, not a label: every multiple-choice NoCoT task must use native `LOGPROBS`/loglikelihood scoring through `/v1/completions`, never generative exact-match merely tagged `NoCoT`. The LightEval task must have the native multiple-choice solver/scorer wiring (`multiple_choice(cache=True)` and `choice()` where applicable); `generation_size` or a metadata field does not select the request type.
- Benchmark-native task configuration is authoritative. Automatic mode selection may choose NoCoT, FakeCoT, or CoT, but must be verified against actual pipeline logs (`LOGPROBS` versus `GENERATIVE`) before publishing results. A mode mismatch invalidates the score even when its uploaded label looks correct.
- NoCoT details must represent the actual completion input without a CoT prompt. CoT/FakeCoT details must use the server-rendered prompt after the configured RWKV template is applied. Never manually append choices, duplicate `Answer:`, expose an internal candidate-request JSON envelope as the user message, or reconstruct a different prompt for display.
- CoT/FakeCoT generation must not inherit a single-answer newline stop sequence from the native task. Newline is valid for one-token completion tasks but truncates a reasoning trace before its final answer; thinking-mode requests must let the model finish by its own stop/EOS protocol.
- Multiple-token choices are scored by summing the conditional logprob of every candidate token; this is sequence loglikelihood, not PPL. Single-token multiple-choice tasks may use the same native path until the server exposes reliable specified-token prompt logprobs.
- Configure generated-token budgets explicitly per model and context: normal generation requests up to 4096 tokens and high-difficulty math/reasoning requests up to 32768 tokens with the current 32768-token endpoint context. The producer must subtract the actual prompt token count before sending the request, record the effective prompt-dependent budget, and never let the gateway reject a request because prompt plus completion exceeds context. Record model, effective budget, context length, prompt template, request type, and truncation rate with every result. A completion reaching the output limit without a completed answer is truncation, not a valid answer.
- Keep producer source, configuration, entrypoint, and live lifecycle aligned. Before a rerun, inspect PID argv, cgroup, and unit; stop only producer jobs confirmed to own affected results. Never stop inference, gateway, or unrelated evaluation jobs, and never run deleted legacy CLIs or configs.
- For corrections, preserve existing rows and audit details; fetch a fresh production manifest, expire only exact affected score IDs with a temporary migration script, fetch tasks again, scope the rerun to affected model/benchmark pairs, and verify replacement IDs, modes, request types, budgets, and truncation rates after upload. Never truncate or clear the Scoreboard database for a correction and do not expire unrelated benchmarks.
- Smoke validation must demonstrate actual `LOGPROBS` versus `GENERATIVE` requests on the required 1.5B and 2.9B samples. Production runs must stream completed rollouts, support interruption/resumption, and keep credentials root-only or environment-only (never in source, logs, or argv).
- Responsibility boundaries remain strict: this repository owns the LightEval producer pipeline, RWKV HTTP adapter, and Scoreboard publication; external systems own inference lifecycle, weights, wkv mode, and endpoint manifests. Evaluation-only transient units may reuse existing lifecycle operations for one-off runs, but must not become a permanent scheduler. Do not modify Scoreboard AGENTS or implementation from this repository.

## Directory Specification

```text
configs/eval/                    RWKV one-click evaluation configuration and external endpoint pool manifest schema examples
src/lighteval/main_rwkv.py       RWKV one-click evaluation CLI and configuration validation
src/lighteval/models/rwkv/       RWKV HTTP model adapter and endpoint pool client
src/lighteval/tasks/             LightEval native benchmark definitions; do not add tasks just because the target list is missing items
tests/unit/rwkv/                 Hermetic tests for RWKV CLI, configuration, HTTP pool, and model
temp/                            External deployment manifests, startup scripts, and four-model process orchestration entry points
```

Changes within `src/lighteval/`, LightEval tasks, and input contracts must apply the Correspondence Principle; `temp/` is not within LightEval's management scope and only stores deployment snapshots and operation entry points provided by external systems. It must not use LightEval internal implementations as code prototypes.

model_name must clearly specify the exact weight version number for Qwen (e.g., Qwen3.5-2B) / RWKV7 (see the `RWKV7 Weights` section for details).
Adding any new file requires user confirmation.

## Authoritative RWKV7 Implementations
(1) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py
(2) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py
(3) https://github.com/BlinkDL/Albatross -- authoritative low-level inference engine implementation repository (CUDA, for PRO6000, no scheduling, no varlen)
(4) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp -- authoritative pretraining implementation repository (CUDA, for H100)
(5) https://zhiyuan1i.github.io/posts/dplr-mathematics -- Mathematical principles of Diagonal Plus Low Rank (DPLR): parallel computation of explicit transition matrices
(6) https://github.com/rwkv-rs/transformers-rwkv/tree/rwkv -- authoritative RWKV Huggingface Transformers adaptation repository (with rust tokenizer, x10 faster than python implementation)

## RWKV7 Weights
General weight naming convention: {arch_version}-{data_version}-{param_size}-{release_date}-{ctx_len}.pth
Example: rwkv7-g1h-7.2b-20260710-ctx10240.pth
arch_version: architecture version, such as rwkv7(default), rwkv7a(experimental, rwkv7 with DeepEmbed), rwkv7b(experimental, rwkv7 with DeepEmbedAttn)
data_version: data version, such as g1a, g1b... (The further back in the alphabet, the better)
param_size: parameter scale, only includes 0.1b, 0.4b, 1.5b (often used in RL), 2.9b, 7.2b (often used in inference tests), 13.3b
(1) https://huggingface.co/BlinkDL/rwkv7-g1/tree/main -- authoritative weight Release source (updated every month)
(2) https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main -- authoritative weight Test source (updated irregularly)
(3) https://huggingface.co/rwkv-rs/rwkv7-g1-st -- authoritative weight Release source (for transformers)

## Correctness Checks
1. Whether the three sets of Prompt Templates provided in the transformers-rwkv and corresponding rwkv7-g1-st weight repositories can be correctly applied
2. Use wkv_mode=fp32io16 by default
3. When using Open Think mode, use decoding parameters temp 0.96, top_p 0.76, top_k 32, presence_penalty 1.0, frequency_penalty 0.1, penalty_decay 0.988; when using Fake Think mode, use decoding parameters temperature 1.0, top_p 0.28, top_k 32; when using Open Think + Function Call mode, use decoding parameters temp 0.96, top_p 0.76, top_k 32, and disable penalty.
4. Refer to https://github.com/BlinkDL/Albatross/blob/main/faster3a_2605/eval_gpqa_diamond.py to complete the implementation of a general grader for multiple-choice questions
5. Refer to https://github.com/BlinkDL/Albatross/blob/main/faster3a_2605/eval_math500.py to complete the implementation of a general grader for short-answer questions
6. Model scores should be similar to those of Qwen3.5 models with similar parameter counts

## Throughput Checks
1. VRAM headroom should be less than 10% of the total, and GPU utilization should reach 97%. If transformers-rwkv or FlashRWKV2 has performance issues, report to the user promptly.
2. Use HTTP protocol to support communication between the evaluation side and the inference side. Request concurrency should be slightly greater than inference concurrency (a small amount of queuing is allowed, but idling is prohibited).
3. Refer to the evaluation methodology widely used by the community for this benchmark, and ensure that the appropriate interface is used (`/v1/completions`, `/v1/chat/completions`), along with the thinking intensity (NoCoT, FakeCoT, CoT) and prompt templates (`Question: ... Answer:; User: ... Assistant: <think; etc.`).

## End-to-End Testing
Each time a benchmark design is added or modified, testing must be completed.
Use 1.5b + 2.9b models to run a ten-question sample of the benchmark, and upload to https://eval.rwkv.rs/test/dashboard.
If the wait time is long, analyze whether the wait time is reasonable, especially focusing on: whether the dataset is repeatedly read, whether total inference throughput meets the target, whether the grader is asynchronous and concurrent (beware of RACE), whether individual evaluation results are streamed to disk, and whether score upload only streams and uploads the necessary parts.
Need to analyze from the dashboard the reasons for "wrong answers" and "failed to answer". If it is a capability issue caused by abnormal model output, ignore it; if the model completed the answer normally, but the answer extractor and grader had anomalies causing incorrect grading, then it needs to be fixed.
Need to check from the dashboard whether the model output for "correct answers" actually matches the standard answer. If it is actually different but the grader is too lenient, it also needs to be fixed.

## Interruption and Resumption
Start evaluation tasks in a way that allows background execution and automatic restart.
For evaluation restarts caused by requirement changes or bug fixes, prioritize creating a temporary migration script in /tmp to reuse existing results, rather than rerunning everything from scratch.
Completed rollouts are immediately written to disk; on exception, only the currently executing rollout is lost, and missing parts are filled in on resume.
After the task ends, cleanup is required to avoid leaving residual user services, etc.

## Result Saving
Record detailed (benchmark_name, model_name, n_samples, k_metrics, cot_mode, prompt_template), [optional: complete wkv_mode fp32io16 vs fp16 comparison] corresponding to (accuracy, truncation rate), where truncation rate is defined as the number of samples that reached the output limit without completing the answer / total samples.

## Responsibility Boundaries
This repository is independently responsible for the LightEval native evaluation pipeline of currently registered benchmarks, the RWKV HTTP model adapter, standard results/details output, and Scoreboard score publishing. External systems are responsible for the inference service lifecycle, weight and wkv_mode switching, and endpoint pool manifests; other frameworks are responsible for benchmarks not registered with LightEval. This repository does not carry cross-framework scheduling or external evaluation lifecycle.

## Sync Upstream
This repository needs to periodically sync with the upstream LightEval repository. First refactor the code according to the Correspondence Principle and Non-Interference Principle, remove single-use wrappers, clean up stale static tests, then sync upstream. Finally, the sync status should show "ahead x, behind 0".

## Env
Use uv to manage the local and remote dedicated environment ./.venv. This project is strictly prohibited from using other environments, and other projects are strictly prohibited from using this project's environment, to avoid environment pollution issues.
If need to clear the database, ssh rwkv-rs-server.

## Inference API
url: api.rwkv.rs
1.5B: bsz1024
2.9B: bsz1024
7.2B: bsz512
13.3B: bsz1280
