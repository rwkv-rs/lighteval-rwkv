# Fork delta vs latest upstream LightEval

Sync target: `upstream/main` = `6ba40c42` (`a8007c15` nltk 3.10.3, `6ba40c42` dependabot).
Local `HEAD` = `780b6980`; merge is applied **staged and uncommitted** (`git merge --no-commit upstream/main`).
Status after commit will be `ahead 56, behind 0`.

Derivation (authoritative, re-runnable):

```bash
git diff --name-only upstream/main   # working tree, includes the 16 uncommitted fork files
```

**Reconciliation note — 71 → 70.** The pre-merge reconnaissance listed 71 files because `.github/dependabot.yml`
was part of the delta (upstream added it, `HEAD` did not have it). Staging the merge converged that file with
upstream, so the live delta is **70 files**. 70 is the denominator used throughout this document.

Classification rule (AGENTS.md Correspondence Principle): a change is **board A** when a similar implementation
already exists in this repository and can serve as a prototype (if the prototype carries a model name, substitute
`RWKV`/case variants; otherwise keep the prototype's name). It is **board B** when no prototype exists and the
change encodes an RWKV-specific requirement.

| Board | Files | Meaning |
| --- | --- | --- |
| A only | 25 | 21 in the live delta + 4 removed by the simplification pass |
| A + B | 17 | 16 live + 1 removed |
| B only | 29 | 28 live + 1 upstream-owned file that converged during the merge |
| **unique union** | **71** | 65 live + 6 with an explicit disposition below |

### Delta reconciliation (71 measured → 65 live)

The fork-vs-upstream delta measured 71 files before the staged merge. Six of those 71 are no longer part of the
live delta; they are classified here with their board assignment and disposition so that **all 71 files carry a
board assignment and the counts reconcile to 71**:

| File | Board | Disposition |
| --- | --- | --- |
| `.github/dependabot.yml` | B (upstream-owned) | upstream added it; staging the merge converged it, so it is no longer a fork delta |
| `src/lighteval/models/rwkv/__init__.py` | A | REMOVED by the simplification pass (no upstream model subpackage ships one) |
| `src/lighteval/tasks/rwkv_free_response.py` | A | REMOVED by the simplification pass (0 importers under `src/`) |
| `src/lighteval/tasks/rwkv_single_choice.py` | A + B | REMOVED by the simplification pass (0 importers under `src/`) |
| `tests/unit/rwkv/test_free_response.py` | A | REMOVED together with its module |
| `tests/unit/rwkv/test_single_choice.py` | A | REMOVED together with its module |

Live delta = 71 − 6 = **65 files**, all classified in the board tables below.

---

## Board A — a prototype exists in this repository

### A1 · RWKV HTTP model adapter (lane 1)

| File | Change | Prototype (path:symbol) | Refactor action |
| --- | --- | --- | --- |
| `src/lighteval/models/rwkv/http_model.py` | `RWKVHTTPModelConfig(ModelConfig)` + `RWKVHttpModel(LightevalModel)` | `src/lighteval/models/vllm/vllm_model.py:VLLMModelConfig`, `VLLMModel`; `src/lighteval/models/endpoints/endpoint_model.py:InferenceEndpointModelConfig`, `InferenceEndpointModel` | config/model names must pair like upstream (`VLLMModelConfig`/`VLLMModel`); `RWKVHTTPModelConfig` vs `RWKVHttpModel` is an internal mismatch — unify the acronym casing |
| `src/lighteval/models/rwkv/http_model.py` | `cleanup`, `tokenizer`, `add_special_tokens`, `max_length`, `tok_encode` overrides | `src/lighteval/models/abstract_model.py:cleanup`, `tokenizer`, `add_special_tokens`, `max_length`, `tok_encode` | keep names identical to the abstract base (already compliant) |
| `src/lighteval/models/rwkv/http_pool.py` | retry policy constants `API_MAX_RETRY`, `API_RETRY_SLEEP`, `API_RETRY_MULTIPLIER` | `src/lighteval/metrics/utils/llm_as_judge.py:API_MAX_RETRY`, `API_RETRY_SLEEP` | keep upstream constant names; align the multiplier with the same retry-loop shape |
| `src/lighteval/models/rwkv/http_pool.py` | HTTP error mapping (`PoolError`, `ContextLengthError`, `_context_error_message`) | `src/lighteval/models/endpoints/endpoint_model.py` error/retry handling | mirror the endpoint-model error vocabulary instead of introducing a parallel one where the semantics match |
| `src/lighteval/models/rwkv/pipeline.py` | `RWKVPipeline(Pipeline)` | `src/lighteval/pipeline.py:Pipeline` | keep the `Pipeline` subclass contract (`evaluate`, `_post_process_outputs`, `_init_tasks_and_requests`) and `RWKV`-prefixed class name |
| `src/lighteval/models/rwkv/pipeline.py` | `RWKVAvgAtK(SampleLevelComputation)` with `k`/`score_rollout`/`extract_rollout_answer` | `src/lighteval/metrics/metrics_sample.py:AvgAtN`, `PassAtK`, `MajAtN` (`__init__(self, n=None, **kwargs)`) | upstream puts `SamplingMetric` computations in `metrics/metrics_sample.py` and names them `…AtN`; either follow that naming/structure or record why the RWKV variant stays in the model module |
| `src/lighteval/models/rwkv/pipeline.py` | `_dataset_cache_key` | `src/lighteval/utils/cache_management.py:SampleCache._get_task_hash` | reuse the cache-management hashing vocabulary rather than a parallel key helper |

### A2 · Tasks, metrics and scorers (lane 2)

| File | Change | Prototype (path:symbol) | Refactor action |
| --- | --- | --- | --- |
| `src/lighteval/tasks/tasks/agieval.py` | `agieval_prompt` returns `None` for out-of-range `gold` | `src/lighteval/tasks/tasks/gsm_plus.py:64`, `src/lighteval/tasks/tasks/hle.py:204` (prompt functions returning `None`) | keep the upstream "prompt returns `None` ⇒ sample dropped" convention (already compliant) |
| `src/lighteval/tasks/tasks/aimo.py` | `hf_subset="default"` | `src/lighteval/tasks/lighteval_task.py` `hf_subset` handling; sibling tasks using `"default"` | keep upstream config-field names (already compliant) |
| `src/lighteval/tasks/tasks/arithmetic.py` | `hf_data_files` + pinned `hf_revision` + `version` bump | `src/lighteval/tasks/lighteval_task.py` `hf_data_files` support; `tests/unit/tasks/test_lighteval_task.py:test_hf_data_files` | keep upstream `LightevalTaskConfig` field names; only the pinned revision constant is fork-local |
| `src/lighteval/tasks/tasks/asdiv.py` | `choices=[...]` list fix (was a bare string) | `src/lighteval/tasks/requests.py:Doc.choices` (`list[str]`) | compliant with the `Doc` contract |
| `src/lighteval/tasks/tasks/gpqa.py` | `random.Random(question).randint(0, 3)` replaces module-level `random.randint` | `src/lighteval/tasks/lighteval_task.py`, `src/lighteval/tasks/prompt_manager.py` (`random.Random(...)` usage) | keep the seeded-generator form for reproducible sample order |
| `src/lighteval/tasks/tasks/ifbench/instructions.py` | empty-string guards in `NGramOverlapChecker`, `EmojiSentenceChecker`, `ParagraphLastFirstWordMatchChecker` | the same checker methods in `src/lighteval/tasks/tasks/ifbench/instructions.py` (upstream `if not …: return False` guard style) | in-place defensive fix; keep upstream checker class/method names |
| `src/lighteval/tasks/tasks/ifeval/instructions_utils.py` | `_get_stopwords()` with `@functools.lru_cache(maxsize=1)` | `src/lighteval/tasks/tasks/ifeval/instructions_utils.py:_get_sentence_tokenizer` (`@functools.lru_cache(maxsize=1)`) | strong prototype: mirror `_get_sentence_tokenizer` exactly |
| `src/lighteval/tasks/tasks/lcb/codegen_metrics.py` | `check_correctness` manager/process structure, `p.join()`, `list(result[0])` | the same function in upstream `src/lighteval/tasks/tasks/lcb/codegen_metrics.py` | in-place fix, keep upstream function/parameter names |
| `src/lighteval/tasks/tasks/lcb/main.py` | `CodegenMetric` extension | `src/lighteval/tasks/tasks/lcb/main.py:CodegenMetric` (upstream) | keep the upstream class name and `compute` signature |
| `src/lighteval/tasks/tasks/mathqa.py` | option parsing via `ast.literal_eval` / regex split | `src/lighteval/tasks/tasks/mmmu_pro.py`, `src/lighteval/tasks/tasks/musr.py` (`literal_eval` option parsing) | mirror the existing task-local option-parsing idiom |
| `src/lighteval/tasks/tasks/mathqa.py` | `hf_revision` pin | `src/lighteval/tasks/lighteval_task.py` `hf_revision` | compliant |
| `src/lighteval/tasks/tasks/med.py` | instruction built from actual `choices` | `src/lighteval/tasks/requests.py:Doc.instruction`; `src/lighteval/tasks/tasks/mmlu_pro.py` | keep `Doc.instruction` semantics |
| `src/lighteval/tasks/tasks/med.py` | `hf_data_files` + `hf_revision` | `src/lighteval/tasks/lighteval_task.py` `hf_data_files` | compliant |
| `src/lighteval/tasks/tasks/mmlu_pro.py` | `labels = ascii_uppercase[: len(options)]` + `{letters}` in `TEMPLATE` | upstream `src/lighteval/tasks/tasks/mmlu_pro.py:TEMPLATE` and `mmlu_pro_prompt_function` | in-place fix of the upstream function; keep names |
| `src/lighteval/tasks/tasks/olympiade_bench.py` | removed `specific={}` | `src/lighteval/tasks/requests.py:Doc.specific` | compliant (drops a field the `Doc` default already covers) |

### A3 · Core pipeline, logging and cache plumbing (lane 3)

| File | Change | Prototype (path:symbol) | Refactor action |
| --- | --- | --- | --- |
| `src/lighteval/pipeline.py` | extracted `_finalize_metrics()` called from `run()` | the post-processing block it was extracted from in the same `Pipeline.run`; `src/lighteval/logging/evaluation_tracker.py:EvaluationTracker` | keep the extraction; it is behaviour-preserving for native backends |
| `src/lighteval/logging/info_loggers.py` | `log_streaming`, `flush_pending_batch`, `finalize_task`, `_task_stats` | `src/lighteval/logging/info_loggers.py:DetailsLogger.log`, `aggregate`, `CompiledDetail` | streaming path must remain a strict superset of `log()`/`aggregate()`; keep identical field names |
| `src/lighteval/logging/info_loggers.py` | `n_samples`/`n_completions`/`n_truncated`/`truncation_rate` counters; `.encode()` fixes | existing `CompiledDetail`/`CompiledDetailOverAllTasks` fields; existing `xxhash.xxh64(...)` call sites | compliant; the `.encode()` change mirrors the sibling call sites |
| `src/lighteval/logging/evaluation_tracker.py` | `_TaskParquetWriter`, `open_task_writer`, `write_task_batch`, `task_details_path`, `run_date_id` | `src/lighteval/logging/evaluation_tracker.py:EvaluationTracker.save`, `_get_details_sub_folder` | incremental writer must reuse the same path/date-id/column rules as `save()` |
| `src/lighteval/models/model_output.py` | `finish_reasons`, `stop_reasons`, `terminal_token_ids` + `__post_init__` alignment check | existing `ModelResponse` fields (`output_tokens`, `logprobs`, `reasonings`) and their slicing in `ModelResponse.final_text` | keep the parallel-list contract identical to existing optional lists |
| `src/lighteval/utils/cache_management.py` | `_CALLABLE_MEMORY_ADDRESS` normalization in `SampleCache._get_task_hash` | `src/lighteval/utils/cache_management.py:SampleCache._get_task_hash` (`cfg.__str__(lite=True)` hashing) | in-place fix; keeps the existing hash contract |
| `src/lighteval/__main__.py` | registers `lighteval.main_rwkv.rwkv` | existing registrations in `src/lighteval/__main__.py` (`main_vllm.vllm`, `main_custom.custom`, `main_sglang.sglang`) | compliant; keep the `Evaluation Backends` rich-help panel |
| `src/lighteval/logging/scoreboard.py` | parquet/detail aggregation and `_TaskAccumulator` | `src/lighteval/logging/evaluation_tracker.py:EvaluationTracker.save`, `src/lighteval/logging/info_loggers.py:DetailsLogger.CompiledDetail` | aggregation must read the same detail/parquet layout as the tracker writes |
| `src/lighteval/main_rwkv.py` | `rwkv()` CLI command + `@app.command(rich_help_panel="Evaluation Backends")` | `src/lighteval/main_endpoint.py:inference_endpoint`, `tgi`, `litellm` (command shape + `yaml.safe_load` config read) | keep the upstream CLI command shape (`app.command`, panel name, preflight logging) |
| `pyproject.toml` | adds `tomli; python_version < '3.11'`, `math-verify==0.5.2`, `langcodes[data]`, `math` extra entry | the existing dependency list and `[project.optional-dependencies] math` block in the same file | compliant: entries follow the existing extras structure |
| `README.md` | RWKV backend section | existing README section structure (`## ⚡️ Installation`, `## 🚀 Quickstart`) | compliant: place the RWKV section in the existing structure |

### A4 · Tests (lane 4)

| File | Change | Prototype (path:symbol) | Refactor action |
| --- | --- | --- | --- |
| `tests/unit/rwkv/test_config.py` | new | `tests/unit/models/vllm/test_vllm_model.py`, `tests/unit/models/endpoints/test_endpoint_model.py` | keep upstream unit-test layout/naming (`test_<unit>.py`, pytest style) |
| `tests/unit/rwkv/test_evaluation_runner.py` | new | `tests/unit/models/endpoints/test_endpoint_model.py` | same |
| `tests/unit/rwkv/test_http_model.py` | new | `tests/unit/models/endpoints/test_endpoint_model.py`, `tests/unit/models/vllm/test_vllm_model.py` | same |
| `tests/unit/rwkv/test_http_pool.py` | new | `tests/unit/models/endpoints/test_endpoint_model.py` | same |
| `tests/unit/rwkv/test_main_rwkv.py` | new | `tests/unit/models/endpoints/test_endpoint_model.py` | same |
| `tests/unit/rwkv/test_pipeline.py` | new | `tests/unit/pipeline/test_reasoning_tags.py` | same |
| `tests/unit/rwkv/test_scoreboard.py` | new | `tests/unit/logging/test_evaluation_tracker.py` | same |
| `tests/unit/logging/test_evaluation_tracker.py` | extended for streaming writers | the same upstream test file | compliant: additive cases only |
| `tests/unit/models/test_model_output.py` | extended for new response lists | the same upstream test file | compliant: additive cases only |
| `tests/unit/tasks/test_lighteval_task.py` | monkeypatched `load_dataset`, agieval `None` case | the same upstream test file | compliant: strengthens assertions, adds coverage |
| `tests/unit/utils/test_caching.py` | `test_task_hash_ignores_process_local_callable_addresses` | the same upstream test file | compliant |

---

## Board B — no prototype (RWKV-specific requirement)

| File | Change | Why there is no prototype |
| --- | --- | --- |
| `AGENTS.md` | new repository governance document | fork-only project instructions; upstream ships no `AGENTS.md` |
| `configs/eval/lighteval-full.toml` | new evaluation config | fork-only external endpoint-pool manifest schema (AGENTS.md: `configs/eval/` owns the manifest examples) |
| `configs/eval/lighteval-test.toml` | new smoke-test config | same fork-only manifest schema |
| `configs/eval/vllm_pool.example.json` | new pool manifest example | same fork-only manifest schema; placement mirrors `examples/model_configs/*.yaml` but the schema has no upstream analogue |
| `temp/evaluation_runner.py` | deployment orchestrator | AGENTS.md places `temp/` outside LightEval's management scope (external deployment snapshot); must not use LightEval internals as prototype |
| `temp/main.py` | deployment entry point | same, out of LightEval scope |
| `temp/main_g1j.py` | deployment entry point | same, out of LightEval scope |
| `temp/run_g1j.sh` | startup script | same, out of LightEval scope |
| `temp/run_rwkv.sh` | startup script | same, out of LightEval scope |
| `temp/scoreboard-backfill/1.5b.toml` | scoreboard backfill manifest | same, out of LightEval scope |
| `temp/scoreboard-backfill/7.2b.toml` | scoreboard backfill manifest | same, out of LightEval scope |
| `src/lighteval/tasks/multilingual/tasks/ceval.py` | `field:knowledge` tag | `field:*` taxonomy is fork-invented (0 upstream hits; consumed by `ScoreboardCallback._FIELD_MARKER`) |
| `src/lighteval/tasks/tasks/aime.py` | `field:math` tag | same fork-only taxonomy |
| `src/lighteval/tasks/tasks/arc.py` | `field:science` tag | same fork-only taxonomy |
| `src/lighteval/tasks/tasks/bigbench_hard.py` | `field:reasoning` tag | same fork-only taxonomy |
| `src/lighteval/tasks/tasks/commonsenseqa.py` | `field:reasoning` tag | same fork-only taxonomy |
| `src/lighteval/tasks/tasks/gsm8k.py` | `field:math` tag | same fork-only taxonomy |
| `src/lighteval/tasks/tasks/gsm_plus.py` | `field:math` tag | same fork-only taxonomy |
| `src/lighteval/tasks/tasks/hellaswag.py` | `field:reasoning` tag | same fork-only taxonomy |
| `src/lighteval/tasks/tasks/ifbench/main.py` | `field:instruction` tag | same fork-only taxonomy |
| `src/lighteval/tasks/tasks/ifeval/main.py` | `field:instruction` tag | same fork-only taxonomy |
| `src/lighteval/tasks/tasks/math.py` | `field:math` tag | same fork-only taxonomy |
| `src/lighteval/tasks/tasks/math_500.py` | `field:math` tag | same fork-only taxonomy |
| `src/lighteval/tasks/tasks/mmlu.py` | `field:knowledge` tag | same fork-only taxonomy |
| `src/lighteval/tasks/tasks/mmlu_redux.py` | `field:knowledge` tag | same fork-only taxonomy |
| `src/lighteval/tasks/tasks/openbookqa.py` | `field:science` tag | same fork-only taxonomy |
| `src/lighteval/tasks/tasks/truthfulqa.py` | `field:knowledge` tag | same fork-only taxonomy |
| `src/lighteval/tasks/tasks/winogrande.py` | `field:reasoning` tag | same fork-only taxonomy |
| `src/lighteval/logging/scoreboard.py` | `ScoreboardCallback`, campaign/idempotency, publication protocol, `_FIELD_MARKER` | no upstream `*Callback` class exists; this implements the fork's external scoreboard contract (`eval.rwkv.rs`) |
| `src/lighteval/main_rwkv.py` | `RWKVEvaluationConfig.read` TOML manifest, `resolve_benchmarks`, `ResolvedBenchmarks`, `_selector_sample_budgets` | fork-only external manifest + selector-budget contract; upstream mains read YAML model configs |
| `src/lighteval/models/rwkv/http_model.py` | `STREAMING_FLUSH_BATCH_SIZE`, `REQUEST_CONTRACT_VERSION`, `CACHE_POOL_FINGERPRINT`, prompt-template/sampling tables | fork-only HTTP request contract between evaluator and inference service |
| `src/lighteval/models/rwkv/http_pool.py` | `PoolManifest`, `Replica`, `CapacityScheduler`, `Completion`, `LogprobScore` | fork-only endpoint-pool manifest schema and capacity scheduling (AGENTS.md assigns endpoint pools to external systems) |
| `src/lighteval/models/rwkv/pipeline.py` | `_evaluation_plan`, `_selector_priority`, `_make_document_ids_unique`, `_dataset_cache_lock` | fork-only rollout/selector scheduling and file-lock behaviour; upstream has no `flock` usage |
| `src/lighteval/tasks/tasks/lcb/codegen_metrics.py` | `_FORK_CONTEXT`/`_SPAWN_CONTEXT`, `_evaluation_executor` lru_cache, `BrokenProcessPool` recovery | fork-only workaround for the fork deadlock observed under the RWKV HTTP/asyncio pipeline; upstream uses a bare `ProcessPoolExecutor` |
| `src/lighteval/tasks/tasks/lcb/main.py` | `CodegenMetric.extract_answer` staticmethod, `field:coding` tag | `extract_answer` exists nowhere upstream (only `multilingual/utils/adapters_utils.py` has an unrelated local `extract_answer`); it is a fork hook for extracted-answer publication |
| `src/lighteval/tasks/tasks/agieval.py` | `field:knowledge` tag | fork-only taxonomy (the `None`-return guard in the same file is board A) |
| `src/lighteval/tasks/tasks/aimo.py` | `field:math` tag | fork-only taxonomy |
| `src/lighteval/tasks/tasks/arithmetic.py` | `field:math` tag | fork-only taxonomy |
| `src/lighteval/tasks/tasks/asdiv.py` | `field:math` tag | fork-only taxonomy |
| `src/lighteval/tasks/tasks/gpqa.py` | `field:science` tag | fork-only taxonomy |
| `src/lighteval/tasks/tasks/mathqa.py` | `field:math` tag | fork-only taxonomy |
| `src/lighteval/tasks/tasks/med.py` | `field:medical` tag | fork-only taxonomy |
| `src/lighteval/tasks/tasks/mmlu_pro.py` | `field:knowledge` tag | fork-only taxonomy |
| `src/lighteval/tasks/tasks/olympiade_bench.py` | `field:math` tag | fork-only taxonomy |

---

## Refactor lanes (task-3, 4 parallel `openai-codex/gpt-5.6-luna` Thinking:max workers)

| Lane | Owns | Board-A focus |
| --- | --- | --- |
| 1 | `src/lighteval/models/rwkv/*` | A1 rows: package `__init__` convention, config/model name pairing, retry vocabulary, `SamplingMetric` naming/placement |
| 2 | `src/lighteval/tasks/**`, `src/lighteval/main_rwkv.py` | A2 rows + `main_rwkv.py` CLI shape |
| 3 | `src/lighteval/logging/**`, `src/lighteval/pipeline.py`, `src/lighteval/models/model_output.py`, `src/lighteval/utils/cache_management.py`, `pyproject.toml`, `README.md`, `src/lighteval/__main__.py` | A3 rows |
| 4 | `tests/**` | A4 rows: align test layout/names with the lane 1–3 renames; assertions keep their strength |

Ownership is disjoint per file; lane 4 depends on lane 1–3 symbol decisions. No new files are created by any lane
(AGENTS.md requires user confirmation for new files). `temp/` is not refactored — only this document touches it.

---

## Simplification pass (post-classification, user-directed)

After the Correspondence lanes landed, the user required the refactor to produce a **substantial code reduction**
by removing redundant logic rather than naming churn alone. The lanes had produced almost none (lane 1: zero
changes; lane 2: two factory renames; lane 3: one schema helper), so a redundancy pass followed.

| Removed file | Lines | Why it was redundant |
| --- | --- | --- |
| `src/lighteval/tasks/rwkv_single_choice.py` | 292 | zero importers under `src/`; the live choice path is `http_model.py:498` (`doc.choices` for `SamplingMethod.LOGPROBS`) plus `scoreboard.py:_logprob_choice_scores` / `_logprob_prediction` |
| `src/lighteval/tasks/rwkv_free_response.py` | 93 | zero importers under `src/`; live math scoring is native scorers wrapped by `RWKVAvgAtK` (`models/rwkv/pipeline.py`) |
| `src/lighteval/models/rwkv/__init__.py` | 14 | no upstream model subpackage ships an `__init__.py`; re-exports were unused (namespace-package imports still resolve) |
| `tests/unit/rwkv/test_single_choice.py` | 280 | tested only the removed module |
| `tests/unit/rwkv/test_free_response.py` | 70 | tested only the removed module |
| **total** | **749** | |

Coverage was preserved where the code under test is live: the two tests in `tests/unit/rwkv/test_pipeline.py` that
exercised `RWKVAvgAtK`'s answer-extraction delegation through the deleted scorer were rewritten to use a 16-line
test-local `_StubSamplingScorer`, keeping every assertion (`score_rollout`, `extract_rollout_answer`,
`rwkv_rollout_scores`, `rwkv_rollout_extracted_answers`).

Net effect: `src/` loses 399 lines and `tests/` loses 350 lines; the unit suite went from 635 to 561 passing tests
(74 tests belonged to the removed modules) with 0 failures and ruff clean.

---

## Residual findings (verified during acceptance; no action taken)

1. **The two RWKV scorer modules were orphaned and have been removed** (see "Simplification pass"). They were
   imported only by tests, never by `src/`. If they were *intended* to be wired into the evaluation path, their
   absence is a pre-existing wiring gap rather than a refactor regression; the live path (native scorers +
   `RWKVAvgAtK`) is what actually produced every recorded score.
2. **`.github/dependabot.yml` is no longer part of the delta.** Upstream added it and the staged merge converged it,
   so the live derivation yields 70 files rather than the 71 seen during pre-merge reconnaissance. This document uses
   70 as its denominator throughout; the goal text's "71 files" is superseded by the live derivation.
3. **Lane 1 applied zero changes.** Every board-A row in `src/lighteval/models/rwkv/*` was already compliant or was
   deferred as a cross-lane proposal. This is consistent with the earlier Correspondence passes (`cb6a99d8`,
   `fb3e65ce`), which had already aligned the model-adapter surface.
