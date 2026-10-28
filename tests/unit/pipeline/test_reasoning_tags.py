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

import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics.metrics import Metrics
from lighteval.models.dummy.dummy_model import DummyModel, DummyModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.imports import is_package_available


class TestPipelineReasoningTags(unittest.TestCase):
    """Test suite for pipeline reasoning tags functionality using DummyModel."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a simple test task
        self.task_config = LightevalTaskConfig(
            name="test_reasoning_task",
            prompt_function=lambda x: x,
            hf_repo="test_repo",
            hf_subset="default",
            metrics=[Metrics.exact_match],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            generation_size=10,
            stop_sequence=["\n"],
            num_fewshots=0,
        )
        self.input_task_name = "test_reasoning_task|0"
        self.task_config_name = self.task_config.full_name

        # Create test documents with reasoning tags in expected responses
        self.test_docs = [
            Doc(
                task_name=self.input_task_name,
                query="What is 2+2?",
                choices=["4"],
                gold_index=[0],
                instruction="",
                sampling_methods=[SamplingMethod.GENERATIVE],
            ),
        ]

        # Mock dataset
        self.mock_dataset = {"test": self.test_docs}

    def _mock_task_registry(self, input_task_name, task_config, task_docs, responses_with_reasoning_tags):
        """Create a fake registry for testing."""

        class FakeTask(LightevalTask):
            def __post_init__(self):
                self._docs = task_docs

            def get_docs(self, max_samples=None):
                return task_docs

            @staticmethod
            def download_dataset_worker(task) -> None:
                # Mock dataset loading
                return task._docs

        class FakeRegistry(Registry):
            def __init__(
                self, tasks: Optional[str] = None, load_multilingual: bool = False, custom_tasks: Optional[str] = None
            ):
                self.tasks_list = [input_task_name]
                # suite_name, task_name, few_shot = input_task_name.split("|")
                self.task_to_configs = {input_task_name: [task_config]}

            def load_tasks(self):
                return {input_task_name: FakeTask(config=task_config)}

        # Create a DummyModel that returns responses with reasoning tags
        class TestDummyModel(DummyModel):
            def __init__(self, config):
                super().__init__(config)

            def greedy_until(self, docs):
                # Return responses with reasoning tags
                return responses_with_reasoning_tags

        return FakeRegistry, TestDummyModel

    def test_remove_reasoning_tags_enabled(self):
        """Test that reasoning tags are removed when remove_reasoning_tags=True."""

        # Responses with reasoning tags
        responses_with_reasoning = [
            ModelResponse(text=["<think>Let me think about this... 2+2=4</think>The answer is 4"])
        ]

        FakeRegistry, TestDummyModel = self._mock_task_registry(
            self.input_task_name, self.task_config, self.test_docs, responses_with_reasoning
        )

        # Initialize accelerator if available
        if is_package_available("accelerate"):
            from accelerate import Accelerator

            Accelerator()

        with patch("lighteval.pipeline.Registry", FakeRegistry):
            # Create pipeline with reasoning tag removal enabled
            pipeline_params = PipelineParameters(
                launcher_type=ParallelismManager.NONE,
                remove_reasoning_tags=True,
                reasoning_tags=[("<think>", "</think>")],
                max_samples=1,
            )

            evaluation_tracker = EvaluationTracker(output_dir=self.temp_dir)
            model = TestDummyModel(DummyModelConfig(seed=42))

            pipeline = Pipeline(
                tasks="test_reasoning_task|0",
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model=model,
            )

            # Run the pipeline
            pipeline.evaluate()

            # Check that reasoning tags were removed from post-processed text
            details = pipeline.evaluation_tracker.details
            self.assertEqual(
                details["test_reasoning_task|0"][0]["model_response"]["text_post_processed"], ["The answer is 4"]
            )

    def test_remove_reasoning_tags_enabled_tags_as_string(self):
        """Test that reasoning tags are removed when remove_reasoning_tags=True."""

        # Responses with reasoning tags
        responses_with_reasoning = [
            ModelResponse(text=["<think>Let me think about this... 2+2=4</think>The answer is 4"])
        ]

        FakeRegistry, TestDummyModel = self._mock_task_registry(
            self.input_task_name, self.task_config, self.test_docs, responses_with_reasoning
        )

        # Initialize accelerator if available
        if is_package_available("accelerate"):
            from accelerate import Accelerator

            Accelerator()

        with patch("lighteval.pipeline.Registry", FakeRegistry):
            # Create pipeline with reasoning tag removal enabled
            pipeline_params = PipelineParameters(
                launcher_type=ParallelismManager.NONE,
                remove_reasoning_tags=True,
                reasoning_tags='[("<think>", "</think>")]',
                max_samples=1,
            )

            evaluation_tracker = EvaluationTracker(output_dir=self.temp_dir)
            model = TestDummyModel(DummyModelConfig(seed=42))

            pipeline = Pipeline(
                tasks="test_reasoning_task|0",
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model=model,
            )

            # Run the pipeline
            pipeline.evaluate()

            # Check that reasoning tags were removed from post-processed text
            details = pipeline.evaluation_tracker.details
            self.assertEqual(
                details["test_reasoning_task|0"][0]["model_response"]["text_post_processed"], ["The answer is 4"]
            )

    def test_remove_reasoning_tags_enabled_default_tags(self):
        """Test that reasoning tags are removed when remove_reasoning_tags=True."""

        # Responses with reasoning tags
        responses_with_reasoning = [
            ModelResponse(text=["<think>Let me think about this... 2+2=4</think>The answer is 4"])
        ]

        FakeRegistry, TestDummyModel = self._mock_task_registry(
            self.input_task_name, self.task_config, self.test_docs, responses_with_reasoning
        )

        # Initialize accelerator if available
        if is_package_available("accelerate"):
            from accelerate import Accelerator

            Accelerator()

        with patch("lighteval.pipeline.Registry", FakeRegistry):
            # Create pipeline with reasoning tag removal enabled
            pipeline_params = PipelineParameters(
                launcher_type=ParallelismManager.NONE, remove_reasoning_tags=True, max_samples=1
            )

            evaluation_tracker = EvaluationTracker(output_dir=self.temp_dir)
            model = TestDummyModel(DummyModelConfig(seed=42))

            pipeline = Pipeline(
                tasks="test_reasoning_task|0",
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model=model,
            )

            # Run the pipeline
            pipeline.evaluate()

            # Check that reasoning tags were removed from post-processed text
            details = pipeline.evaluation_tracker.details
            self.assertEqual(
                details["test_reasoning_task|0"][0]["model_response"]["text_post_processed"], ["The answer is 4"]
            )

    def test_remove_reasoning_tags_disabled(self):
        """Test that reasoning tags are preserved when remove_reasoning_tags=False."""

        # Responses with reasoning tags
        responses_with_reasoning = [
            ModelResponse(text=["<think>Let me think about this... 2+2=4</think>The answer is 4"])
        ]

        FakeRegistry, TestDummyModel = self._mock_task_registry(
            self.input_task_name, self.task_config, self.test_docs, responses_with_reasoning
        )

        # Initialize accelerator if available
        if is_package_available("accelerate"):
            from accelerate import Accelerator

            Accelerator()

        with patch("lighteval.pipeline.Registry", FakeRegistry):
            # Create pipeline with reasoning tag removal disabled
            pipeline_params = PipelineParameters(
                launcher_type=ParallelismManager.NONE,
                remove_reasoning_tags=False,
                reasoning_tags=[("<think>", "</think>")],
                max_samples=1,
            )

            evaluation_tracker = EvaluationTracker(output_dir=self.temp_dir)
            model = TestDummyModel(DummyModelConfig(seed=42))

            pipeline = Pipeline(
                tasks="test_reasoning_task|0",
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model=model,
            )

            # Run the pipeline
            pipeline.evaluate()

            # Check that post-processed text is None (= no post processing happened)
            details = pipeline.evaluation_tracker.details
            self.assertIsNone(
                details["test_reasoning_task|0"][0]["model_response"]["text_post_processed"],
            )

    def test_custom_reasoning_tags(self):
        """Test that custom reasoning tags are correctly applied."""

        # Responses with custom reasoning tags
        responses_with_reasoning = [
            ModelResponse(text=["[reasoning]This is my thought process[/reasoning]Final answer: 4"])
        ]

        FakeRegistry, TestDummyModel = self._mock_task_registry(
            self.input_task_name, self.task_config, self.test_docs, responses_with_reasoning
        )

        # Initialize accelerator if available
        if is_package_available("accelerate"):
            from accelerate import Accelerator

            Accelerator()

        with patch("lighteval.pipeline.Registry", FakeRegistry):
            # Create pipeline with custom reasoning tags
            pipeline_params = PipelineParameters(
                launcher_type=ParallelismManager.NONE,
                remove_reasoning_tags=True,
                reasoning_tags=[("[reasoning]", "[/reasoning]")],
                max_samples=1,
            )

            evaluation_tracker = EvaluationTracker(output_dir=self.temp_dir)
            model = TestDummyModel(DummyModelConfig(seed=42))

            pipeline = Pipeline(
                tasks="test_reasoning_task|0",
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model=model,
            )

            # Run the pipeline
            pipeline.evaluate()

            # Check that reasoning tags were removed from post-processed text
            details = pipeline.evaluation_tracker.details
            self.assertEqual(
                details["test_reasoning_task|0"][0]["model_response"]["text_post_processed"], ["Final answer: 4"]
            )

    def test_multiple_reasoning_tags(self):
        """Test that multiple reasoning tag pairs are correctly handled."""

        # Responses with multiple reasoning tag types
        responses_with_reasoning = [
            ModelResponse(text=["<think>First thought</think>Some text<reason>Second thought</reason>Final: 4"])
        ]

        FakeRegistry, TestDummyModel = self._mock_task_registry(
            self.input_task_name, self.task_config, self.test_docs, responses_with_reasoning
        )

        # Initialize accelerator if available
        if is_package_available("accelerate"):
            from accelerate import Accelerator

            Accelerator()

        with patch("lighteval.pipeline.Registry", FakeRegistry):
            # Create pipeline with multiple reasoning tag pairs
            pipeline_params = PipelineParameters(
                launcher_type=ParallelismManager.NONE,
                remove_reasoning_tags=True,
                reasoning_tags='[("<think>", "</think>"), ("<reason>", "</reason>")]',
                max_samples=1,
            )

            evaluation_tracker = EvaluationTracker(output_dir=self.temp_dir)
            model = TestDummyModel(DummyModelConfig(seed=42))

            pipeline = Pipeline(
                tasks="test_reasoning_task|0",
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model=model,
            )

            # Run the pipeline
            pipeline.evaluate()

            # Check that reasoning tags were removed from post-processed text
            details = pipeline.evaluation_tracker.details
            self.assertEqual(
                details["test_reasoning_task|0"][0]["model_response"]["text_post_processed"],
                ["Some textFinal: 4"],
            )

    def test_reasoning_tags_validation(self):
        """Test that invalid reasoning_tags parameter raises appropriate error."""

        for test_string in ["['incorrect_format']", "invalid_format"]:
            with self.assertRaises(ValueError) as context:
                PipelineParameters(
                    launcher_type=ParallelismManager.NONE,
                    reasoning_tags=test_string,  # Should be a list of tuples
                )

            # Check that the error message mentions the expected format
            print(context.__dict__)
            self.assertIn("reasoning_tags must be a list of pair tuples", str(context.exception))

    def test_default_reasoning_tags(self):
        """Test that default reasoning tags are correctly set."""

        pipeline_params = PipelineParameters(launcher_type=ParallelismManager.NONE)

        # Check that default reasoning tags are set
        self.assertEqual(pipeline_params.reasoning_tags, [("<think>", "</think>")])
        self.assertTrue(pipeline_params.remove_reasoning_tags)


class TestRWKVNonInterference(unittest.TestCase):
    """Prototype: TestPipelineReasoningTags; no endpoints or datasets needed."""

    def test_model_pools_have_independent_concurrent_benchmark_queues(self):
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier, Event

        from src.rwkv_eval.configs import BenchmarkSpec, ModelEndpoint
        from src.rwkv_eval.main import evaluate

        models = [
            ModelEndpoint("https://vllm.rwkvos.com", "fixture", f"rwkv7-g1k-{size}", slots, 32768)
            for size, slots in (("1.5b", 1024), ("2.9b", 1024), ("7.2b", 512), ("13.3b", 1280))
        ]
        benchmarks = [BenchmarkSpec("knowledge", "arc:easy"), BenchmarkSpec("knowledge", "mmlu")]
        for fail in (False, True):
            with self.subTest(failed_pool=fail):
                started = Barrier(4)
                advanced = Event()
                completed = []

                def executor(max_workers, mp_context):
                    self.assertEqual(max_workers, 4)
                    self.assertEqual(mp_context.get_start_method(), "spawn")
                    return ThreadPoolExecutor(max_workers=max_workers)

                def worker(replicas, tasks, options, log_level):
                    self.assertEqual(len(replicas), 1)
                    self.assertEqual(tasks, benchmarks)
                    self.assertEqual(options["max_samples"], None)
                    self.assertFalse(options["test_mode"])
                    self.assertEqual(options["seed"], 42)
                    self.assertEqual(options["output_dir"], "/fixture/results")
                    started.wait(3)
                    name = replicas[0].model_name
                    if name.endswith("13.3b"):
                        self.assertTrue(advanced.wait(3), "fast pool cannot advance independently")
                        if fail:
                            raise RuntimeError("failed model pool")
                    else:
                        for task in tasks:
                            completed.append((name, task.selector))
                        advanced.set()
                    return [name], []

                with (
                    patch("src.rwkv_eval.main.ProcessPoolExecutor", side_effect=executor),
                    patch(
                        "src.rwkv_eval.main._evaluate_pool",
                        side_effect=worker,
                    ),
                ):
                    run = evaluate(models, benchmarks, no_upload=True, output_dir="/fixture/results")
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "failed model pool"):
                            asyncio.run(run)
                    else:
                        scores, failures = asyncio.run(run)
                        self.assertEqual(scores, [model.model_name for model in models])
                        self.assertEqual(failures, [])
                self.assertEqual(len(completed), 6)
                self.assertIn((models[0].model_name, "mmlu"), completed)

    def test_registry_changes_are_private(self):
        from lighteval.tasks.tasks.arc import arc_easy
        from lighteval.tasks.tasks.commonsenseqa import commonsenseqa
        from lighteval.tasks.tasks.math_500 import math_500
        from lighteval.tasks.tasks.mmlu import mmlu_abstract_algebra
        from src.rwkv_eval.main import RWKVRegistry

        originals = {config.name: config for config in (arc_easy, commonsenseqa, math_500, mmlu_abstract_algebra)}
        baseline = {name: config.metrics[0].sample_level_fn.__dict__.copy() for name, config in originals.items()}
        with patch.object(Registry, "load_all_task_configs", return_value=originals):
            native_before = Registry(tasks="commonsenseqa").load_tasks()["commonsenseqa|0"]
            rwkv = RWKVRegistry(tasks=",".join(originals)).load_tasks()
            native_after = Registry(tasks="commonsenseqa").load_tasks()["commonsenseqa|0"]
            native_mmlu = Registry(tasks="mmlu:abstract_algebra").load_tasks()["mmlu:abstract_algebra|0"]
        self.assertEqual(native_before.sampling_methods, [SamplingMethod.GENERATIVE])
        self.assertEqual(native_after.sampling_methods, [SamplingMethod.GENERATIVE])
        self.assertEqual(rwkv["commonsenseqa|0"].sampling_methods, [SamplingMethod.LOGPROBS])
        self.assertEqual(rwkv["mmlu:abstract_algebra|0"].sampling_methods, [SamplingMethod.LOGPROBS])
        self.assertEqual(native_mmlu.sampling_methods, [SamplingMethod.GENERATIVE])
        for name, config in originals.items():
            self.assertIsNot(config, rwkv[name + "|0"].config)
            self.assertEqual(rwkv[name + "|0"].config.version, config.version + (name != "math_500"))
            self.assertIsNot(config.metrics[0].sample_level_fn, rwkv[name + "|0"].metrics[0].sample_level_fn)
            self.assertEqual(config.metrics[0].sample_level_fn.__dict__, baseline[name])
        sample = {"question": "Q?", "choices": {"text": ["first", "second"], "label": ["A", "B"]}, "answerKey": "B"}
        self.assertTrue(arc_easy.prompt_function(sample).query.endswith("Answer:"))
        self.assertEqual(
            rwkv["arc:easy|0"].formatter(sample, "arc:easy").query, arc_easy.prompt_function(sample).query
        )

    def test_rwkv_answer_cue_follows_request_mode(self):
        from src.rwkv_eval.main import _NO_COT_BENCHMARKS, rwkv_prompt

        query = "Question: What is 2 + 2?\nA. 3\nB. 4\nAnswer: "
        for task_name in (*_NO_COT_BENCHMARKS, "arc:easy", "mmlu:anatomy"):
            for suffix in ("", " ", "\n", " \n"):
                existing = query.rstrip() + suffix
                with self.subTest(task_name=task_name, suffix=suffix):
                    doc = rwkv_prompt(
                        {},
                        task_name,
                        prompt_function=lambda line, name: Doc(query=existing, choices=["A", "B"], gold_index=1),
                    )
                    self.assertEqual(doc.query, existing)
                    self.assertEqual(doc.choices, ["A", "B"])
                    self.assertEqual(doc.gold_index, 1)

            for original in (
                "Question: What is 2 + 2?\nA. 3\nB. 4",
                "Question: Explain the label Answer: inside this question. ",
                "Question: Q? Answer: ",
                "Question: Q?\n答案：",
            ):
                with self.subTest(task_name=task_name, original=original):
                    doc = rwkv_prompt(
                        {},
                        task_name,
                        prompt_function=lambda line, name: Doc(query=original, choices=["A", "B"], gold_index=1),
                    )
                    expected = original.rstrip()
                    if expected.endswith("Answer:"):
                        expected = expected[: -len("Answer:")].rstrip()
                    self.assertEqual(doc.query, expected + "\nAnswer: ")
                    self.assertEqual(doc.choices, ["A", "B"])
                    self.assertEqual(doc.gold_index, 1)
                    again = rwkv_prompt({}, task_name, prompt_function=lambda line, name: doc)
                    self.assertEqual(again.query, expected + "\nAnswer: ")

        for task_name in ("gsm8k", "arithmetic:2da"):
            with self.subTest(task_name=task_name):
                doc = rwkv_prompt(
                    {}, task_name, prompt_function=lambda line, name: Doc(query=query, choices=["4"], gold_index=0)
                )
                self.assertEqual(doc.query, query.rstrip()[: -len("Answer:")].rstrip() + "\n")
                self.assertEqual(doc.choices, ["4"])
                self.assertEqual(doc.gold_index, 0)
        unchanged = "Question: Explain the label Answer: inside this question."
        doc = rwkv_prompt(
            {}, "gsm8k", prompt_function=lambda line, name: Doc(query=unchanged, choices=["4"], gold_index=0)
        )
        self.assertEqual(doc.query, unchanged)

    def test_rwkv_split_ids_are_private_and_preserve_documents(self):
        from src.rwkv_eval.main import RWKVRegistry

        config = LightevalTaskConfig(
            name="openbookqa",
            prompt_function=lambda line, name: Doc(
                task_name=name, query=line["query"], choices=["A", "B"], gold_index=line["gold"]
            ),
            hf_repo="fixture",
            hf_subset="default",
            hf_avail_splits=["validation", "test"],
            evaluation_splits=["validation", "test"],
            metrics=[Metrics.loglikelihood_acc],
        )
        dataset = {
            "validation": [{"query": "Validation question\nAnswer: ", "gold": 1}],
            "test": [{"query": "Test question\nAnswer: ", "gold": 0}],
        }
        with patch.object(Registry, "load_all_task_configs", return_value={config.name: config}):
            native_before = Registry(tasks="openbookqa").load_tasks()["openbookqa|0"]
            rwkv = RWKVRegistry(tasks="openbookqa").load_tasks()["openbookqa|0"]
            native_after = Registry(tasks="openbookqa").load_tasks()["openbookqa|0"]
        for task in (native_before, rwkv, native_after):
            task.dataset = copy.deepcopy(dataset)
        native_docs = native_before._get_docs_from_split(["validation", "test"])
        rwkv_docs = rwkv._get_docs_from_split(["validation", "test"])
        self.assertEqual([doc.id for doc in native_docs], ["0", "0"])
        self.assertEqual([doc.id for doc in rwkv_docs], ["validation:0", "test:0"])
        for native, private in zip(native_docs, rwkv_docs, strict=True):
            expected = asdict(native)
            expected["id"] = private.id
            self.assertEqual(asdict(private), expected)
        self.assertEqual([doc.id for doc in native_after._get_docs_from_split(["validation", "test"])], ["0", "0"])
        self.assertEqual(
            [doc.id for doc in rwkv._get_docs_from_split(["validation", "test"], few_shots=True)], ["0", "0"]
        )
        rwkv.evaluation_split = ["test"]
        self.assertEqual([doc.id for doc in rwkv._get_docs_from_split(["test"])], ["0"])

    def test_rwkv_split_ids_resume_without_cross_question_cache_hits(self):
        from lighteval.utils.cache_management import SampleCache
        from src.rwkv_eval.main import RWKVRegistry

        config = LightevalTaskConfig(
            name="openbookqa",
            prompt_function=lambda line, name: Doc(
                task_name=name, query=line["query"], choices=["A", "B"], gold_index=line["gold"]
            ),
            hf_repo="fixture",
            hf_subset="default",
            hf_avail_splits=["validation", "test"],
            evaluation_splits=["validation", "test"],
            metrics=[Metrics.loglikelihood_acc],
        )
        with patch.object(Registry, "load_all_task_configs", return_value={config.name: config}):
            registry = RWKVRegistry(tasks="openbookqa")
            task = registry.load_tasks()["openbookqa|0"]
        task.dataset = {
            "validation": [{"query": "Validation question\nAnswer: ", "gold": 1}],
            "test": [{"query": "Test question\nAnswer: ", "gold": 0}],
        }
        docs = task.get_docs()
        responses = [
            ModelResponse(
                input=doc.query,
                candidate_inputs=[doc.query + choice for choice in doc.choices],
                logprobs=[-1.0 if i == doc.gold_index else -5.0 for i in range(2)],
            )
            for doc in docs
        ]
        with tempfile.TemporaryDirectory() as directory:
            model_config = DummyModelConfig(model_name="fixture", cache_dir=directory)
            cache = SampleCache(model_config)
            cache._init_registry(registry)
            cache.cache_sample(docs[0], responses[0], SamplingMethod.LOGPROBS)
            resumed = SampleCache(model_config)
            resumed._init_registry(registry)
            missing, cached_tasks = resumed.get_samples_to_process_and_cache(docs, SamplingMethod.LOGPROBS)
            self.assertEqual([doc.id for doc in missing], [docs[1].id])
            self.assertEqual(
                resumed.get_samples_from_cache([docs[0]], cached_tasks, SamplingMethod.LOGPROBS)[0].input,
                docs[0].query,
            )
            resumed.cache_sample(docs[1], responses[1], SamplingMethod.LOGPROBS)
            task_ids = {resumed.get_task_id(task.full_name, SamplingMethod.LOGPROBS)}
            resumed.cache_samples(docs, responses, task_ids, SamplingMethod.LOGPROBS)
            reopened = SampleCache(model_config)
            reopened._init_registry(registry)
            self.assertEqual(reopened.get_samples_to_process_and_cache(docs, SamplingMethod.LOGPROBS)[0], [])
            ordered = reopened.get_samples_from_cache(docs[::-1], task_ids, SamplingMethod.LOGPROBS)
            self.assertEqual(
                [asdict(response) for response in ordered], [asdict(response) for response in responses[::-1]]
            )
            again = task.get_docs()
            self.assertEqual({doc.id: doc.query for doc in again}, {doc.id: doc.query for doc in docs})

    def test_rwkv_choice_expectation_is_private_and_preserves_failure_details(self):
        from lighteval.metrics.metrics_sample import PassAtK
        from lighteval.tasks.tasks.gpqa import gpqa_diamond_instruct, gpqa_main_instruct
        from src.rwkv_eval.configs import SamplingConfig
        from src.rwkv_eval.main import RWKVExactMatches, RWKVPassAtK, RWKVRegistry
        from src.rwkv_eval.upload import get_detail

        originals = {config.name: config for config in (gpqa_diamond_instruct, gpqa_main_instruct)}
        baseline = {name: config.metrics[0].sample_level_fn.__dict__.copy() for name, config in originals.items()}
        with patch.object(Registry, "load_all_task_configs", return_value=originals):
            native_before = Registry(tasks=",".join(originals)).load_tasks()
            rwkv = RWKVRegistry(tasks=",".join(originals)).load_tasks()
            native_after = Registry(tasks=",".join(originals)).load_tasks()
        sampling = SamplingConfig(24576, 0.96, 32, 0.76, 1.0, 0.1, 0.988, 42)
        for name, config in originals.items():
            with self.subTest(task=name):
                doc = Doc(query="Q", choices=["A", "B", "C", "D"], gold_index=0, task_name=name + "|0")
                response = ModelResponse(
                    text=["unfinished"],
                    text_post_processed=[""],
                    rendered_prompt="server prompt",
                    finish_reasons=["length"],
                )
                for native in (native_before, native_after):
                    self.assertEqual(native[name + "|0"].metrics[0].sample_level_fn.compute(doc, response), 0.0)
                self.assertEqual(rwkv[name + "|0"].metrics[0].sample_level_fn.compute(doc, response), 0.25)
                self.assertEqual(config.metrics[0].sample_level_fn.__dict__, baseline[name])
                detail, bucket = get_detail(doc, response, sampling, {}, 0, no_cot=False)
                self.assertEqual((detail.answer, detail.is_passed, bucket), ("-", False, 2))
                self.assertEqual(detail.messages[-1]["content"], "unfinished")
        doc = Doc(query="Q", choices=["A", "B", "C", "D"], gold_index=0)
        for predictions, expected in (
            (["", ""], 0.4375),
            (["B", ""], 0.25),
            (["A", ""], 0.5),
            (["B", "B"], 0.0),
            (["A", "A"], 1.0),
        ):
            with self.subTest(predictions=predictions):
                scorer = RWKVPassAtK(k=1, n=2, sample_scoring_function=RWKVExactMatches(choice=True).compute)
                self.assertEqual(scorer.compute(doc, ModelResponse(text=predictions)), expected)
        self.assertEqual(RWKVPassAtK(k=2, n=2).pass_at_k([0.25, 0.25]), 0.0)
        self.assertEqual(RWKVPassAtK(k=1, n=3).pass_at_k([0.25, 0.25]), 0.0)
        self.assertEqual(PassAtK(k=1, n=2).compute(doc, ModelResponse(text=["", ""])), 0.0)
        self.assertEqual(RWKVExactMatches().compute_one_item("42", ""), 0.0)

    def test_generation_uses_batch_docs(self):
        from lighteval.tasks.tasks.gpqa import gpqa_diamond_instruct
        from lighteval.tasks.tasks.math_500 import math_500
        from src.rwkv_eval.main import RWKVPipeline

        cases = [
            (math_500, ["42"], 0, "reasoning</think>\\boxed{42}", "42", 1.0),
            (gpqa_diamond_instruct, ["A", "B", "C", "D"], 1, "reasoning</think>Answer: B", "B", 1.0),
            (gpqa_diamond_instruct, ["A", "B", "C", "D"], 1, "unfinished", "", 0.25),
        ]
        for config, choices, gold_index, raw, answer, score in cases:
            for streaming in (True, False):
                name = config.name + "|0"
                doc = Doc(query="Q", choices=choices, gold_index=gold_index, task_name=name)

                class TestDummyModel(DummyModel):
                    def greedy_until(self, docs):
                        return [ModelResponse(text=[raw], rendered_prompt="server prompt", finish_reasons=["stop"])]

                def load_dataset(tasks, *args):
                    for task in tasks.values():
                        task._docs = [copy.deepcopy(doc)]
                        task.dataset = {task.evaluation_split[0]: [doc]}

                with (
                    self.subTest(task=name, streaming=streaming, answer=answer),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    with (
                        patch.object(Registry, "load_all_task_configs", return_value={config.name: config}),
                        patch.object(
                            LightevalTask,
                            "load_datasets",
                            side_effect=load_dataset,
                        ),
                    ):
                        pipeline = RWKVPipeline(
                            config.name,
                            PipelineParameters(
                                launcher_type=ParallelismManager.NONE, streaming_evaluation=streaming, max_samples=1
                            ),
                            EvaluationTracker(output_dir=directory),
                            model=TestDummyModel(DummyModelConfig(seed=42)),
                        )
                        if streaming:
                            self.assertFalse(pipeline.sampling_docs)
                        pipeline.evaluate()
                        detail = pipeline.get_details()[name][0]
                        self.assertEqual(detail.model_response.final_text, [answer])
                        if streaming:
                            self.assertEqual(detail.doc.specific["rwkv_rollout_scores"], [score])
                        self.assertEqual(next(iter(detail.metric.values())), score)
                        self.assertEqual(detail.model_response.text, [raw])

    def test_logprobs_streaming_does_not_require_generated_text(self):
        from lighteval.tasks.tasks.commonsenseqa import commonsenseqa
        from src.rwkv_eval.main import RWKVPipeline

        doc = Doc(query="Q\n", choices=["first", "second"], gold_index=1, task_name="commonsenseqa|0")

        class TestDummyModel(DummyModel):
            def loglikelihood(self, docs):
                return [ModelResponse(text=[""], logprobs=[-3.0, -1.0], candidate_inputs=["Q\nfirst", "Q\nsecond"])]

        def load_dataset(tasks, *args):
            for task in tasks.values():
                task._docs = [copy.deepcopy(doc)]
                task.dataset = {task.evaluation_split[0]: [doc]}

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(Registry, "load_all_task_configs", return_value={"commonsenseqa": commonsenseqa}),
                patch.object(
                    LightevalTask,
                    "load_datasets",
                    side_effect=load_dataset,
                ),
            ):
                pipeline = RWKVPipeline(
                    "commonsenseqa",
                    PipelineParameters(
                        launcher_type=ParallelismManager.NONE, streaming_evaluation=True, max_samples=1
                    ),
                    EvaluationTracker(output_dir=directory),
                    model=TestDummyModel(DummyModelConfig(seed=42)),
                )
                pipeline.evaluate()
                detail = pipeline.get_details()["commonsenseqa|0"][0]
                self.assertIsNone(detail.model_response.text_post_processed)
                self.assertEqual(detail.model_response.candidate_inputs, ["Q\nfirst", "Q\nsecond"])
                self.assertEqual(next(iter(detail.metric.values())), 1.0)

    def test_loglikelihood_concurrency_preserves_order_and_token_sum(self):
        from threading import Barrier

        from lighteval.models.endpoints.litellm_model import LiteLLMClient

        barrier = Barrier(2)
        model = object.__new__(LiteLLMClient)
        model.concurrent_requests = 2
        model._cache = None

        def request(prompt, *, top_logprobs):
            if isinstance(prompt, str):
                barrier.wait(timeout=3)
                return [{"prompt_token_ids": [10]}]
            return [
                {
                    "prompt_token_ids": [10, 11, 12],
                    "prompt_logprobs": [None, {"11": {"logprob": -0.25}}, {"12": {"logprob": -0.5}}],
                }
                for _ in prompt
            ]

        model._prompt_logprobs_request = request
        docs = [Doc(query=query, choices=[" first", " second"], gold_index=0) for query in ("Q", "R")]
        responses = model.loglikelihood(docs)
        self.assertEqual([response.input for response in responses], ["Q", "R"])
        self.assertEqual([response.logprobs for response in responses], [[-0.75, -0.75], [-0.75, -0.75]])
        self.assertEqual(responses[1].candidate_inputs, ["R first", "R second"])

    def test_logprobs_pool_counts_candidate_sequences_not_documents(self):
        import time
        from threading import Barrier, Lock
        from unittest.mock import Mock

        from lighteval.models.endpoints.litellm_model import LiteLLMClient

        model = LiteLLMClient.__new__(LiteLLMClient)
        model.model, model.concurrent_requests, model._cache = "rwkv7-test", 8, Mock()
        docs = [
            Doc(query=str(i), choices=[str(c) for c in range(4 if i < 4 else 20)], gold_index=0, id=str(i))
            for i in range(5)
        ]
        barrier, lock = Barrier(2), Lock()
        active = 0
        peak = 0

        def score(doc):
            nonlocal active, peak
            weight = min(len(doc.choices), model.concurrent_requests)
            with lock:
                active += weight
                peak = max(peak, active)
                self.assertLessEqual(active, 8, "document concurrency exceeded backend sequence capacity")
            if int(doc.id) < 2:
                barrier.wait(3)
                time.sleep(0.03)
            with lock:
                active -= weight
            return ModelResponse(
                input=doc.query,
                logprobs=[-0.25] * len(doc.choices),
                candidate_inputs=[doc.query + choice for choice in doc.choices],
            )

        with patch.object(model, "_loglikelihood", side_effect=score):
            results = LiteLLMClient.loglikelihood.__wrapped__(model, docs)
        self.assertEqual(peak, 8)
        self.assertEqual([response.input for response in results], [doc.query for doc in docs])
        self.assertEqual([len(response.logprobs) for response in results], [4, 4, 4, 4, 20])
        self.assertEqual(model._cache.cache_sample.call_count, 5)

    def test_logprobs_transport_retries_preserve_request_and_timeout(self):
        from unittest.mock import Mock

        import requests

        from lighteval.models.endpoints.litellm_model import LiteLLMClient
        from src.rwkv_eval.configs import ModelEndpoint
        from src.rwkv_eval.main import _litelm_model

        endpoint = ModelEndpoint("https://vllm.rwkvos.com", "fixture", "rwkv7-g1k-1.5b", 1024, 32768)
        config = _litelm_model(endpoint, [endpoint], "NoCoT", None, 4096, 42)
        self.assertEqual(config.timeout, 14400)
        model = LiteLLMClient.__new__(LiteLLMClient)
        model.model, model.base_url, model.api_key, model.timeout = (
            endpoint.model_name,
            config.base_url,
            "fixture",
            config.timeout,
        )
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"prompt_token_ids": [1], "prompt_logprobs": [None]}]}
        unavailable = Mock(status_code=503)
        errors = [requests.exceptions.SSLError("EOF"), requests.exceptions.ReadTimeout("slow peer"), unavailable]
        with (
            patch("lighteval.models.endpoints.litellm_model.requests.post", side_effect=errors + [response]) as post,
            patch(
                "lighteval.models.endpoints.litellm_model.time.sleep",
            ) as sleep,
        ):
            result = model._prompt_logprobs_request("retained prefix", top_logprobs=1)
        self.assertEqual(result, response.json.return_value["choices"])
        self.assertEqual(post.call_count, 4)
        for call in post.call_args_list:
            self.assertEqual(call.kwargs["timeout"], 14400)
            self.assertEqual(call.args[0], "https://vllm.rwkvos.com/v1/completions")
            self.assertEqual(
                call.kwargs["json"],
                {
                    "model": endpoint.model_name,
                    "prompt": "retained prefix",
                    "max_tokens": 0,
                    "echo": True,
                    "prompt_logprobs": 1,
                    "return_token_ids": True,
                },
            )
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2, 4])
        with (
            patch(
                "lighteval.models.endpoints.litellm_model.requests.post",
                side_effect=requests.exceptions.SSLError("EOF"),
            ) as post,
            patch(
                "lighteval.models.endpoints.litellm_model.time.sleep",
            ),
        ):
            with self.assertRaises(requests.exceptions.SSLError):
                model._prompt_logprobs_request("retained prefix", top_logprobs=1)
        self.assertEqual(post.call_count, 6)

    def test_loglikelihood_journal_preserves_completed_peer_on_failure(self):
        from threading import Event

        from lighteval.models.endpoints.litellm_model import LiteLLMClient, LiteLLMModelConfig

        saved = Event()
        docs = [
            Doc(query=name, choices=[" first", " second"], gold_index=0, id=name, task_name="arc:easy|0")
            for name in ("slow", "fast")
        ]
        with tempfile.TemporaryDirectory() as directory:
            config = LiteLLMModelConfig(
                model_name="rwkv7-test",
                provider="openai",
                base_url="https://example.test/v1",
                cache_dir=directory,
                concurrent_requests=4,
                max_model_length=16,
            )
            model = LiteLLMClient(config)
            store = model._cache.cache_sample

            def checkpoint(doc, response, method):
                store(doc, response, method)
                saved.set()

            def score(doc):
                if doc.id == "slow":
                    self.assertTrue(saved.wait(3), "completed LOGPROBS was blocked by a slow peer")
                    raise RuntimeError("interrupted logprob request")
                return ModelResponse(
                    input=doc.query,
                    logprobs=[-0.25, -0.75],
                    candidate_inputs=[doc.query + choice for choice in doc.choices],
                )

            with (
                patch.object(model, "_loglikelihood", side_effect=score),
                patch.object(
                    model._cache,
                    "cache_sample",
                    side_effect=checkpoint,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted logprob request"):
                    model.loglikelihood(docs)
            journal = next(Path(directory).rglob("LOGPROBS.jsonl"))
            self.assertEqual([json.loads(line)["sample_id"] for line in journal.read_text().splitlines()], ["fast"])
            resumed = LiteLLMClient(config)
            calls = []

            def complete(doc):
                calls.append(doc.id)
                return ModelResponse(
                    input=doc.query,
                    logprobs=[-0.25, -0.75],
                    candidate_inputs=[doc.query + choice for choice in doc.choices],
                )

            with patch.object(resumed, "_loglikelihood", side_effect=complete):
                results = resumed.loglikelihood(docs)
            self.assertEqual(calls, ["slow"])
            self.assertEqual([result.input for result in results], ["slow", "fast"])
            self.assertEqual([result.logprobs for result in results], [[-0.25, -0.75], [-0.25, -0.75]])
            with patch.object(LiteLLMClient, "_loglikelihood", side_effect=AssertionError("unexpected inference")):
                self.assertEqual(LiteLLMClient(config).loglikelihood(docs), results)

    def test_task_cache_identity_is_stable(self):
        from src.rwkv_eval.main import RWKVRegistry

        for name in ("mmlu_pro", "commonsenseqa", "math_500", "gpqa:diamond", "truthfulqa:mc"):
            with self.subTest(task=name):
                first = RWKVRegistry(tasks=name).load_tasks()
                second = RWKVRegistry(tasks=name).load_tasks()
                before = next(iter(first.values())).config.__str__(lite=True)
                after = next(iter(second.values())).config.__str__(lite=True)
                self.assertEqual(before, after)
                self.assertNotIn(" at 0x", before)
                config = next(iter(first.values())).config
                config.generation_size += 1
                self.assertNotEqual(before, config.__str__(lite=True))
        import subprocess
        import sys

        script = (
            "from src.rwkv_eval.main import RWKVRegistry; import hashlib; "
            "config=next(iter(RWKVRegistry(tasks='truthfulqa:mc').load_tasks().values())).config; "
            "print(hashlib.sha256(config.__str__(lite=True).encode()).hexdigest())"
        )
        hashes = [
            subprocess.run(
                [sys.executable, "-c", script], check=True, capture_output=True, text=True, timeout=30
            ).stdout
            for _ in range(2)
        ]
        self.assertEqual(hashes[0], hashes[1])

    def test_live_and_parquet_details_share_semantics(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from src.rwkv_eval.configs import SamplingConfig
        from src.rwkv_eval.upload import get_detail, get_eg_details

        sampling = SamplingConfig(4096, 0.96, 32, 0.76, 1.0, 0.1, 0.988, 42)
        cases = [
            (
                Doc(query="Q", choices=["42"], gold_index=0, task_name="math_500|0"),
                ModelResponse(text=["work</think>\\boxed{42}"], rendered_prompt="rendered", finish_reasons=["stop"]),
                False,
            ),
            (
                Doc(query="Q", choices=["42"], gold_index=0, task_name="math_500|0"),
                ModelResponse(text=["42"], rendered_prompt="Assistant: <think>\n</think>", finish_reasons=["stop"]),
                False,
            ),
            (
                Doc(query="Q", choices=["42"], gold_index=0, task_name="math_500|0"),
                ModelResponse(
                    text=["</think>42"],
                    rendered_prompt="server prompt",
                    input_tokens=[1, 2],
                    effective_max_new_tokens=14,
                    seeds=[42],
                    finish_reasons=["stop"],
                ),
                False,
            ),
            (
                Doc(query="Q", choices=["308"], gold_index=0, task_name="asdiv|0"),
                ModelResponse(
                    text=[">They spent $308.00 on tickets."],
                    rendered_prompt="Assistant: <think></think",
                    finish_reasons=["stop"],
                ),
                False,
            ),
            (
                Doc(query="Q", choices=["A", "B", "C", "D"], gold_index=1, task_name="gpqa:diamond|0"),
                ModelResponse(text=["unfinished"], rendered_prompt="rendered", finish_reasons=["stop"]),
                False,
            ),
            (
                Doc(query="Q", choices=[" first", " second"], gold_index=1, task_name="arc:easy|0"),
                ModelResponse(text=[""], candidate_inputs=["Q first", "Q second"], logprobs=[-3.0, -1.0]),
                True,
            ),
        ]
        for doc, response, no_cot in cases:
            with self.subTest(task=doc.task_name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "details" / "model" / "date" / "details.parquet"
                path.parent.mkdir(parents=True)
                sampling_path = Path(directory) / "results" / "model" / "sampling_config_date.json"
                sampling_path.parent.mkdir(parents=True)
                sampling_path.write_text(json.dumps(asdict(sampling)))
                row = {"doc": asdict(doc), "model_response": asdict(response), "metric": {"acc": 0.0}}
                pq.write_table(pa.Table.from_pylist([row]), path)
                live, bucket = get_detail(doc, response, sampling, row["metric"], 0, no_cot=no_cot)
                stored = get_eg_details(path, no_cot=no_cot, task_name=doc.task_name)
                self.assertEqual(stored[bucket], [live])
                if doc.task_name.startswith("gpqa"):
                    self.assertEqual(live.answer, "-")
                    self.assertEqual(json.loads(json.dumps(asdict(live)))["answer"], "-")
                    self.assertFalse(live.is_passed)
                    self.assertEqual(bucket, 2)
                    self.assertEqual(live.messages[-1]["content"], "unfinished")
                if doc.task_name.startswith("asdiv"):
                    self.assertEqual(live.answer, "308.00")
                    self.assertTrue(live.is_passed)
                    self.assertEqual(bucket, 0)
                    self.assertEqual(live.messages[0]["content"], response.rendered_prompt)
                    self.assertEqual(live.messages[-1]["content"], response.text[0])
                if no_cot:
                    self.assertEqual(live.messages[0]["content"], "Q second")
                    self.assertEqual(response[0].candidate_inputs, response.candidate_inputs)
                    with self.assertRaises(ValueError):
                        get_detail(doc, ModelResponse(logprobs=response.logprobs), sampling, {}, 0, no_cot=True)
                else:
                    if response.effective_max_new_tokens is not None:
                        self.assertEqual(live.sampling_config.max_generated_tokens, response.effective_max_new_tokens)
                        self.assertEqual(response[0].seeds, response.seeds)
                    without_prompt = ModelResponse(text=response.text, finish_reasons=response.finish_reasons)
                    with self.assertRaises(ValueError):
                        get_detail(doc, without_prompt, sampling, {}, 0, no_cot=False)

    def test_rwkv_generated_token_budgets_and_native_isolation(self):
        from types import SimpleNamespace

        from lighteval.models.endpoints.litellm_model import LiteLLMClient
        from src.rwkv_eval.configs import SamplingConfig
        from src.rwkv_eval.main import _auto_cot_mode, _litelm_model
        from src.rwkv_eval.upload import get_detail

        for selector in (
            "gpqa:diamond",
            "gpqa:main",
            "aime24",
            "aime25",
            "aimo_progress_prize_1",
            "olympiad_bench",
            "math",
            "math:algebra",
            "math_500",
            "lcb:codegeneration",
        ):
            with self.subTest(selector=selector):
                self.assertEqual(_auto_cot_mode(selector), ("CoT", 24576))
        self.assertEqual(_auto_cot_mode("gsm8k"), ("CoT", 4096))
        self.assertEqual(_auto_cot_mode("asdiv"), ("FakeCoT", 4096))
        self.assertEqual(_auto_cot_mode("mmlu_pro"), ("NoCoT", 4096))
        with tempfile.TemporaryDirectory() as directory:
            endpoint = SimpleNamespace(
                model_name="test-rwkv", url="https://example.test", api_key=None, ctx_len=16, max_num_seqs=2
            )
            mode, budget = _auto_cot_mode("math_500")
            config = _litelm_model(endpoint, [endpoint], mode, "bot", budget, 42, cache_dir=directory)
            self.assertEqual(config.generation_parameters.max_new_tokens, 24576)
            model = LiteLLMClient(config)
            prompt = [{"role": "user", "content": "Q"}]
            reply = SimpleNamespace(
                prompt_text="server prompt",
                usage=SimpleNamespace(prompt_tokens=40000),
                choices=[SimpleNamespace(message=SimpleNamespace(content="</think>42"), finish_reason="stop")],
                _hidden_params={},
            )
            for mode in ("open_think", "fake_think"):
                config.extra_body["chat_template_kwargs"]["rwkv_generation_prompt"] = mode
                for budget in (4096, 8192, 16384, 24576):
                    with (
                        self.subTest(mode=mode, budget=budget),
                        patch(
                            "lighteval.models.endpoints.litellm_model.requests.post",
                            side_effect=AssertionError("unexpected preflight"),
                        ),
                        patch(
                            "lighteval.models.endpoints.litellm_model.litellm.completion", return_value=reply
                        ) as call,
                    ):
                        response = model._LiteLLMClient__call_api(prompt, False, budget, 1, ["\n"])
                        self.assertEqual(call.call_args.kwargs["max_tokens"], budget)
                        self.assertEqual(call.call_args.kwargs["max_completion_tokens"], budget)
                        self.assertEqual(call.call_args.kwargs["stop"], [])
                        self.assertEqual(
                            {
                                key: call.call_args.kwargs[key]
                                for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty")
                            },
                            {"temperature": 0.96, "top_p": 0.90, "presence_penalty": 1.0, "frequency_penalty": 0.09},
                        )
                        self.assertEqual(call.call_args.kwargs["extra_body"]["top_k"], -1)
                        self.assertEqual(call.call_args.kwargs["extra_body"]["penalty_decay"], 0.994)
                        converted = model._generation_response(response, prompt)
                        self.assertEqual(converted.effective_max_new_tokens, budget)
                        self.assertEqual(converted[0].prompt_token_count, 40000)
            with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=False):
                self.assertEqual(model._prepare_max_new_tokens(4, prompt), 4)
            doc = Doc(query="Q", choices=["42"], gold_index=0, task_name="math_500|0")
            sampling = SamplingConfig(24576, 0.96, 32, 0.76, 1.0, 0.1, 0.988, 42)
            detail, _ = get_detail(doc, converted, sampling, {}, 0, no_cot=False)
            self.assertEqual(detail.sampling_config.max_generated_tokens, 24576)
            self.assertEqual(detail.sampling_config.seed, 42)

    def test_rwkv_upload_retains_generated_token_budget_ceiling(self):
        from src.rwkv_eval.configs import RwkvModel, SamplingConfig
        from src.rwkv_eval.upload import Score, get_detail, upload

        sampling = SamplingConfig(16, 0.96, 32, 0.76, 1.0, 0.1, 0.988, 42)
        doc = Doc(query="Q", choices=["42"], gold_index=0, task_name="math_500|0")
        details = [
            get_detail(
                doc,
                ModelResponse(
                    text=["</think>42"],
                    rendered_prompt="server prompt",
                    effective_max_new_tokens=budget,
                    finish_reasons=["stop"],
                ),
                sampling,
                {},
                0,
                no_cot=False,
            )[0]
            for budget in (14, 2)
        ]
        score = Score(RwkvModel("rwkv7", "G1k", "1.5b", 16), "math_500", 2, 1.0, 1.0, 0.0, details, [], [])
        with patch("src.rwkv_eval.upload.httpx.Client") as factory:
            client = factory.return_value.__enter__.return_value
            upload(score, field="maths", cot_mode="CoT", token="test-token")
        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(payload["sampling_config"]["max_generated_tokens"], 14)
        self.assertEqual([d.sampling_config.max_generated_tokens for d in score.passed_details], [14, 2])
        self.assertEqual(payload["passed_details"][0]["messages"][0]["content"], "server prompt")

    def test_rwkv_generation_client_is_owned_until_response_consumed(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier, get_ident
        from types import SimpleNamespace

        from lighteval.models.endpoints.litellm_model import LiteLLMClient, LiteLLMModelConfig
        from lighteval.models.model_input import GenerationParameters

        barrier = Barrier(2)
        clients = []

        class OwnedClient:
            def __init__(self, **kwargs):
                self.owner = get_ident()
                self.closed = False
                clients.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.assert_owner()
                self.closed = True

            def assert_owner(self):
                if self.owner != get_ident() or self.closed:
                    raise AssertionError("transport closed or used by non-owner")

        def completion(**kwargs):
            kwargs["client"].assert_owner()
            self.assertFalse(kwargs["caching"])
            barrier.wait(timeout=5)
            kwargs["client"].assert_owner()
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=2),
                choices=[SimpleNamespace(message=SimpleNamespace(content="42"))],
                _hidden_params={},
            )

        with tempfile.TemporaryDirectory() as directory:
            config = LiteLLMModelConfig(
                model_name="test-rwkv",
                provider="openai",
                base_url="https://example.test/v1",
                cache_dir=directory,
                max_model_length=16,
                extra_body={"chat_template_kwargs": {"rwkv_generation_prompt": "open_think"}},
                generation_parameters=GenerationParameters(temperature=0.96, seed=42),
            )
            model = LiteLLMClient(config)
            with (
                patch("lighteval.models.endpoints.litellm_model.OpenAI", side_effect=OwnedClient),
                patch("lighteval.models.endpoints.litellm_model.httpx.Client"),
                patch("lighteval.models.endpoints.litellm_model.litellm.completion", side_effect=completion),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                futures = [
                    executor.submit(
                        model._LiteLLMClient__call_api, [{"role": "user", "content": "Q"}], False, 16, 1, []
                    )
                    for _ in range(2)
                ]
                for future in futures:
                    self.assertEqual(future.result().choices[0].message.content, "42")
            self.assertEqual(len(clients), 2)
            self.assertIsNot(clients[0], clients[1])
            self.assertTrue(all(client.closed for client in clients))
            config.extra_body = None
            config.model_name = "gpt-test"
            with (
                patch("lighteval.models.endpoints.litellm_model.OpenAI", side_effect=AssertionError("native changed")),
                patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=False),
                patch("lighteval.models.endpoints.litellm_model.litellm.completion") as call,
            ):
                model._LiteLLMClient__call_api([{"role": "user", "content": "Q"}], False, 8, 1, [])
                self.assertNotIn("client", call.call_args.kwargs)
                self.assertTrue(call.call_args.kwargs["caching"])

    def test_rwkv_owned_transport_preserves_normalized_request(self):
        from contextlib import nullcontext
        from types import SimpleNamespace

        import httpx
        from litellm.llms.openai.common_utils import BaseOpenAILLM
        from lighteval.models.endpoints import litellm_model
        from src.rwkv_eval.main import _litelm_model

        payloads = []
        transports = []

        def handler(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "wire-test",
                    "created": 1,
                    "model": "rwkv-wire-test",
                    "object": "chat.completion",
                    "prompt_text": "server prompt",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "42"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                },
            )

        def factory(**kwargs):
            client = httpx.Client(transport=httpx.MockTransport(handler))
            transports.append(client)
            return client

        with tempfile.TemporaryDirectory() as directory:
            endpoint = SimpleNamespace(
                model_name="rwkv-wire-test",
                url="https://wire-test.invalid",
                api_key="test-key",
                ctx_len=32,
                max_num_seqs=1,
            )
            model = litellm_model.LiteLLMClient(
                _litelm_model(endpoint, [endpoint], "CoT", "bot", 16, 42, cache_dir=directory)
            )
            try:
                with (
                    patch.object(litellm_model, "httpx", SimpleNamespace(Client=factory)),
                    patch.object(BaseOpenAILLM, "_get_sync_http_client", side_effect=factory),
                ):
                    with patch.object(litellm_model, "OpenAI", return_value=nullcontext()):
                        model._LiteLLMClient__call_api([{"role": "user", "content": "Q"}], False, 16, 1, [], 1)
                    model._LiteLLMClient__call_api([{"role": "user", "content": "Q"}], False, 16, 1, [], 1)
                self.assertEqual(len(payloads), 2)
                self.assertEqual(payloads[0], payloads[1])
                self.assertEqual(payloads[1]["seed"], 43)
                self.assertEqual(payloads[1]["max_completion_tokens"], 16)
                self.assertEqual(payloads[1]["stop"], [])
                self.assertEqual(
                    {
                        key: payloads[1][key]
                        for key in (
                            "temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "penalty_decay"
                        )
                    },
                    {
                        "temperature": 0.96,
                        "top_p": 0.90,
                        "top_k": -1,
                        "presence_penalty": 1.0,
                        "frequency_penalty": 0.09,
                        "penalty_decay": 0.994,
                    },
                )
                self.assertEqual(payloads[1]["messages"], [{"role": "user", "content": "Q"}])
            finally:
                for client in transports:
                    client.close()

    def test_rwkv_journal_reader_never_truncates_an_active_writer(self):
        from lighteval.utils.cache_management import SampleCache

        with tempfile.TemporaryDirectory() as directory:
            cache = SampleCache(DummyModelConfig(model_name="test", cache_dir=directory))
            path = Path(directory) / "GENERATIVE.parquet"
            journal = path.with_suffix(".jsonl")
            first = (json.dumps({"sample_id": "0", "sample": asdict(ModelResponse(text=["first"]))}) + "\n").encode()
            second = (json.dumps({"sample_id": "1", "sample": asdict(ModelResponse(text=["second"]))}) + "\n").encode()
            journal.write_bytes(first + second[:30])
            with journal.open("ab") as writer:
                self.assertEqual(len(list(cache._load_sample_journal(path))), 1)
                self.assertEqual(journal.read_bytes(), first + second[:30])
                writer.write(second[30:])
                writer.flush()
            self.assertEqual([r["sample_id"] for r in cache._load_sample_journal(path)], ["0", "1"])
            damaged = first + b"\x17\x03\x03\x02\x7f\x85\n" + second
            journal.write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, f"Corrupt completion journal .* at byte {len(first)}"):
                list(cache._load_sample_journal(path))
            self.assertEqual(journal.read_bytes(), damaged)

    def test_rwkv_rollout_journal_recovers_only_missing_child(self):
        from types import SimpleNamespace

        from lighteval.models.endpoints.litellm_model import LiteLLMClient, LiteLLMModelConfig
        from lighteval.models.model_input import GenerationParameters

        with tempfile.TemporaryDirectory() as directory:
            config = LiteLLMModelConfig(
                model_name="test-rwkv",
                provider="openai",
                base_url="https://example.test/v1",
                max_model_length=16,
                cache_dir=directory,
                concurrent_requests=2,
                api_max_retry=1,
                api_retry_sleep=0,
                extra_body={"chat_template_kwargs": {"rwkv_generation_prompt": "open_think"}},
                generation_parameters=GenerationParameters(temperature=0.96, seed=42, max_new_tokens=16),
            )
            doc = Doc(query="Q", choices=["42"], gold_index=0, id="0", task_name="math_500|0", num_samples=2)
            model = LiteLLMClient(config)
            model.prompt_manager.prepare_prompt_api = lambda doc: [{"role": "user", "content": doc.query}]

            def reply(**kwargs):
                self.assertEqual(kwargs["n"], 1)
                self.assertEqual(kwargs["max_tokens"], 16)
                self.assertEqual(kwargs["stop"], [])
                self.assertEqual(kwargs["timeout"], 14400)
                if kwargs["seed"] == 43:
                    raise RuntimeError("interrupted child")
                return SimpleNamespace(
                    prompt_text="server prompt",
                    usage=SimpleNamespace(prompt_tokens=2),
                    choices=[SimpleNamespace(message=SimpleNamespace(content="</think>42"), finish_reason="stop")],
                    _hidden_params={},
                )

            with (
                patch(
                    "lighteval.models.endpoints.litellm_model.requests.post",
                    side_effect=AssertionError("unexpected preflight"),
                ),
                patch(
                    "lighteval.models.endpoints.litellm_model.litellm.completion",
                    side_effect=reply,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "API call failed"):
                    model.greedy_until([doc])
            journal = next(Path(directory).rglob("*.jsonl"))
            self.assertEqual(len(journal.read_text().splitlines()), 1)
            with journal.open("ab") as stream:
                stream.write(b'{"sample_id":"incomplete')
            interrupted = journal.read_bytes()
            resumed = LiteLLMClient(config)
            resumed.prompt_manager.prepare_prompt_api = model.prompt_manager.prepare_prompt_api
            self.assertEqual(journal.read_bytes(), interrupted)
            calls = []

            def complete(**kwargs):
                calls.append(kwargs["seed"])
                return SimpleNamespace(
                    prompt_text="server prompt",
                    usage=SimpleNamespace(prompt_tokens=2),
                    choices=[SimpleNamespace(message=SimpleNamespace(content="</think>43"), finish_reason="stop")],
                    _hidden_params={},
                )

            with (
                patch(
                    "lighteval.models.endpoints.litellm_model.requests.post",
                    side_effect=AssertionError("unexpected preflight"),
                ),
                patch(
                    "lighteval.models.endpoints.litellm_model.litellm.completion",
                    side_effect=complete,
                ),
            ):
                response = resumed.greedy_until([doc])[0]
            self.assertEqual(calls, [43])
            self.assertEqual(response.text, ["</think>42", "</think>43"])
            self.assertEqual(response.effective_max_new_tokens, 16)
            self.assertEqual(response.input_tokens, [])
            self.assertEqual(response.prompt_token_count, 2)
            self.assertEqual(response.seeds, [42, 43])
            with patch(
                "lighteval.models.endpoints.litellm_model.requests.post", side_effect=AssertionError("cache miss")
            ):
                self.assertEqual(LiteLLMClient(config).greedy_until([doc])[0], response)

    def test_rwkv_completion_order_does_not_block_durable_fast_response(self):
        from threading import Event
        from types import SimpleNamespace

        from lighteval.models.endpoints.litellm_model import LiteLLMClient, LiteLLMModelConfig
        from lighteval.models.model_input import GenerationParameters

        saved = Event()
        with tempfile.TemporaryDirectory() as directory:
            model = LiteLLMClient(
                LiteLLMModelConfig(
                    model_name="test-rwkv",
                    provider="openai",
                    base_url="https://example.test/v1",
                    max_model_length=16,
                    cache_dir=directory,
                    concurrent_requests=2,
                    api_max_retry=1,
                    api_retry_sleep=0,
                    extra_body={"chat_template_kwargs": {"rwkv_generation_prompt": "open_think"}},
                    generation_parameters=GenerationParameters(temperature=0.96, seed=42, max_new_tokens=16),
                )
            )
            model.prompt_manager.prepare_prompt_api = lambda doc: [{"role": "user", "content": doc.query}]
            store = model._cache.cache_sample

            def checkpoint(doc, response, method):
                store(doc, response, method)
                saved.set()

            def reply(**kwargs):
                if kwargs["messages"][0]["content"] == "slow":
                    self.assertTrue(saved.wait(3), "fast response was blocked behind slow response")
                    raise RuntimeError("slow request failed")
                return SimpleNamespace(
                    prompt_text="server prompt",
                    usage=SimpleNamespace(prompt_tokens=2),
                    choices=[SimpleNamespace(message=SimpleNamespace(content="</think>42"), finish_reason="stop")],
                    _hidden_params={},
                )

            docs = [
                Doc(query=name, choices=["42"], gold_index=0, id=name, task_name="math_500|0")
                for name in ("slow", "fast")
            ]
            with (
                patch(
                    "lighteval.models.endpoints.litellm_model.requests.post",
                    side_effect=AssertionError("unexpected preflight"),
                ),
                patch(
                    "lighteval.models.endpoints.litellm_model.litellm.completion",
                    side_effect=reply,
                ),
                patch.object(model._cache, "cache_sample", side_effect=checkpoint),
            ):
                with self.assertRaisesRegex(RuntimeError, "API call failed"):
                    model.greedy_until(docs)
            rows = [json.loads(line) for line in next(Path(directory).rglob("*.jsonl")).read_text().splitlines()]
            self.assertEqual([row["sample_id"] for row in rows], ["fast"])

    def test_truthfulqa_details_follow_native_mc1_group(self):
        import numpy as np

        from lighteval.metrics.harness_compatibility.truthful_qa import TruthfulqaMCMetrics
        from src.rwkv_eval.configs import SamplingConfig
        from src.rwkv_eval.upload import _stored_metric_score, get_detail

        doc = Doc(
            query="Q",
            choices=[" best", " false", " other true", " false"],
            gold_index=[0, 2],
            specific={"len_mc1": 2},
            task_name="truthfulqa:mc|0",
        )
        sampling = SamplingConfig(4096, 0.0, 0, 1.0, 0.0, 0.0, 1.0, 42)
        for scores, expected_answer, expected_passed in (
            ([-2.0, -3.0, -1.0, -4.0], "A", True),
            ([-3.0, -2.0, -1.0, -4.0], "B", False),
        ):
            response = ModelResponse(logprobs=scores, candidate_inputs=[doc.query + c for c in doc.choices])
            metrics = TruthfulqaMCMetrics().compute(doc, response)
            detail, bucket = get_detail(doc, response, sampling, metrics, 0, no_cot=True)
            self.assertEqual(detail.answer, expected_answer)
            self.assertEqual(detail.ground_truth, "A")
            self.assertEqual(detail.is_passed, expected_passed)
            self.assertEqual(bucket, 0 if expected_passed else 1)
            self.assertEqual(detail.messages[0]["content"], response.candidate_inputs[ord(expected_answer) - 65])
            self.assertEqual(_stored_metric_score(metrics), float(metrics["truthfulqa_mc1"]))
        self.assertEqual(_stored_metric_score({"mc1": np.bool_(True), "mc2": np.float64(0.42)}), 1.0)

    def test_asdiv_preserves_native_textual_golds(self):
        from lighteval.tasks.tasks.asdiv import asdiv
        from src.rwkv_eval.answer_extract.free_response import extract_answer
        from src.rwkv_eval.main import RWKVExactMatches, RWKVRegistry

        with patch.object(Registry, "load_all_task_configs", return_value={"asdiv": asdiv}):
            task = RWKVRegistry(tasks="asdiv").load_tasks()["asdiv|0"]
        for gold in ("Yes", "Mrs. Hilt", "Cesar's toothbrush", "5, 1", "5:30", "42"):
            with self.subTest(gold=gold):
                doc = task.formatter({"body": "Question", "question": "Q?", "answer": gold}, "asdiv")
                self.assertEqual(doc.get_golds(), [gold])
                answer = extract_answer(
                    f"Answer: {gold}", "short_answer", rendered_prompt="Assistant: <think>\n</think>"
                )
                self.assertEqual(answer, gold)
                self.assertEqual(RWKVExactMatches(allow_text=True).compute_one_item(gold, answer), 1)
        self.assertEqual(extract_answer(r"</think>\boxed{\text{Yes}}", "short_answer"), "Yes")
        self.assertEqual(
            extract_answer(
                ">They spent $308.00 on tickets.",
                "short_answer",
                rendered_prompt="User: Q\n\nAssistant: <think></think",
            ),
            "308.00",
        )
        self.assertIsNone(
            extract_answer(
                "They spent $308.00 on tickets.",
                "short_answer",
                rendered_prompt="User: Q\n\nAssistant: <think></think",
            )
        )
        self.assertIsNone(extract_answer("unfinished", "short_answer"))
        self.assertIsNone(extract_answer("-", "short_answer", rendered_prompt="Assistant: <think>\n</think>"))
        self.assertEqual(RWKVExactMatches(allow_text=True).compute_one_item("Bryan", "Ryanb"), 0)
        doc = task.formatter(
            {
                "body": "Ages",
                "question": "Find the ages of George, Christopher and Ford.",
                "answer": "26 (years old); 18 (years old); 16 (years old)",
            },
            "asdiv",
        )
        self.assertEqual(doc.get_golds(), ["26; 18; 16"])
        prompt = "User: " + doc.query + "\n\nAssistant: <think></think"
        for raw, score in (
            (r">\boxed{Ford: 16, George: 26, Christopher: 18}", 1),
            (r">\boxed{George=26, Christopher=16, Ford=18}", 0),
            (r">\boxed{26}", 0),
        ):
            answer = extract_answer(raw, "short_answer", rendered_prompt=prompt)
            self.assertEqual(RWKVExactMatches(allow_text=True).compute_one_item(doc.get_golds()[0], answer), score)

    def test_olympiad_requires_all_answer_components(self):
        from src.rwkv_eval.main import RWKVExactMatches, RWKVRegistry

        task = RWKVRegistry(tasks="olympiad_bench:OE_TO_maths_en_COMP").load_tasks()[
            "olympiad_bench:OE_TO_maths_en_COMP|0"
        ]
        row = {
            "subject": "Math",
            "language": "English",
            "unit": None,
            "is_multiple_answer": True,
            "answer_type": "Numerical",
            "question": "Q?",
            "final_answer": ["2", "3"],
        }
        doc = task.formatter(row, task.name)
        self.assertEqual(doc.get_golds(), ["2, 3"])
        self.assertEqual(RWKVExactMatches().compute_one_item(doc.get_golds()[0], "2"), 0)
        self.assertEqual(RWKVExactMatches().compute_one_item(doc.get_golds()[0], "2, 3"), 1)

    def test_math_comparator_and_extractor(self):
        from src.rwkv_eval.answer_extract.free_response import extract_answer, extract_math_answer
        from src.rwkv_eval.main import RWKVExactMatches

        expression = r"\frac{1}{\sqrt{2}}"
        self.assertEqual(extract_math_answer(r"\boxed{" + expression + "}"), expression)
        self.assertEqual(extract_math_answer(expression), expression)
        for gold, expected in (
            ("$m-n$", "m-n"),
            ("$|PM| = |PN|$", "|PM| = |PN|"),
            ("$y=bx+c$", "y=bx+c"),
            (r"[-e，+\infty )", r"[-e,+\infty )"),
            ("$c, a, b$", "c, a, b"),
        ):
            self.assertEqual(extract_math_answer(gold), expected)
        self.assertEqual(RWKVExactMatches().compute_one_item("2,3", "2"), 0)
        self.assertIsNone(extract_answer("unclosed reasoning with 42", "math"))
        self.assertEqual(extract_answer("42", "math", rendered_prompt="Assistant: <think>\n</think>"), "42")
        self.assertIsNone(extract_answer("42", "math", rendered_prompt="Assistant: <think>"))
        self.assertIsNone(extract_answer("</think>42<think>unfinished", "math"))
        self.assertIsNone(extract_math_answer(r"\boxed{42"))
        self.assertIsNone(extract_math_answer(r"\boxed{41}, finally \boxed{42"))
        self.assertIsNone(extract_math_answer("The answer could not be found"))
        self.assertEqual(extract_math_answer("Hence x = 42."), "42")
        self.assertEqual(RWKVExactMatches().compute_one_item("1/2", r"\frac{1}{2}"), 1)
        self.assertEqual(RWKVExactMatches().compute_one_item("1/3", r"\frac{1}{2}"), 0)
        self.assertEqual(RWKVExactMatches().compute_one_item("", ""), 0)
        for text, expected in (
            ("The remainder when 54 is divided by 6 is 0, because 54 is exactly divisible by 6.", "0"),
            ("The remainder when 54 is divided by 6 is 0, since 54 is divisible by 6.", "0"),
            ("Thus, there are 90,720 different ways to arrange the letters.", "90720"),
            ("Thus, there are 60,480 ways to arrange the letters.", "60480"),
            ("ANSWER: cot x", r"\cot x"),
            ("ANSWER: sec x", r"\sec x"),
            ("ANSWER: csc x", r"\csc x"),
        ):
            with self.subTest(text=text):
                self.assertEqual(extract_answer("reasoning</think>" + text, "math"), expected)
        self.assertEqual(RWKVExactMatches().compute_one_item(r"\cot x", "cot x"), 1)
        self.assertEqual(RWKVExactMatches().compute_one_item(r"\cot x", "tan x"), 0)
        self.assertEqual(extract_math_answer("[-2,300]"), "[-2,300]")

    def test_rwkv_math_details_preserve_metric_gold_on_live_and_parquet_paths(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from lighteval.metrics import apply_metric
        from src.rwkv_eval.answer_extract.free_response import extract_answer, extract_math_answer
        from src.rwkv_eval.main import RWKVRegistry, _native_score, _sampling_config
        from src.rwkv_eval.upload import get_detail, get_eg_details

        native_before = Registry(tasks="math").load_tasks()
        task = RWKVRegistry(tasks="math").load_tasks()["math:prealgebra|0"]
        sampling = _sampling_config("CoT", 24576, 42)
        for gold, prediction, passed in (
            (r"\mbox{Saturday}", "Saturday", True),
            (r"\mbox{Saturday}", "Thursday", False),
            ("90\\text{ square\nunits}", "90", True),
            ("90\\text{ square\nunits}", "91", False),
            ("90\\text{ square\nunits}", None, False),
            (r"2k", "2", False),
            (r"3032_5", "3032", True),
        ):
            with self.subTest(gold=gold, prediction=prediction):
                line = {"problem": "Q?", "solution": r"Solution: \boxed{" + gold + "}"}
                doc = task.formatter(line, task.name)
                self.assertEqual(doc.get_golds(), [gold])
                raw = "unfinished" if prediction is None else r"reasoning</think>\boxed{" + prediction + "}"
                response = ModelResponse(
                    text=[raw] * 4,
                    text_post_processed=[extract_answer(raw, "math") or ""] * 4,
                    input="User: Q?\n\nAssistant: <think>",
                    rendered_prompt="User: Q?\n\nAssistant: <think>",
                    finish_reasons=["stop"] * 4,
                    effective_max_new_tokens=24576,
                    seeds=[42, 43, 44, 45],
                )
                metric = apply_metric([response], [doc], task.metrics)[0]
                self.assertEqual(_native_score(metric), float(passed))
                detail, bucket = get_detail(doc, response, sampling, metric, 0, no_cot=False)
                self.assertEqual(detail.ground_truth, gold)
                self.assertEqual(detail.answer, prediction if prediction is not None else "-")
                self.assertEqual(detail.is_passed, passed)
                self.assertEqual(bucket, 2 if prediction is None else 0 if passed else 1)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "details" / "model" / "date" / "details.parquet"
                    path.parent.mkdir(parents=True)
                    sampling_path = Path(directory) / "results" / "model" / "sampling_config_date.json"
                    sampling_path.parent.mkdir(parents=True)
                    sampling_path.write_text(json.dumps(asdict(sampling)))
                    row = {"doc": asdict(doc), "model_response": asdict(response), "metric": metric}
                    pq.write_table(pa.Table.from_pylist([row]), path)
                    replay = get_eg_details(path, task_name=doc.task_name)
                    self.assertEqual(replay[bucket], [detail] * 4)
        # Keep the shared extractor and every native task untouched.
        self.assertIsNone(extract_math_answer(r"\mbox{Saturday}"))
        self.assertIsNone(extract_math_answer("90\\text{ square\nunits}"))
        native_after = Registry(tasks="math").load_tasks()
        for name, native in native_before.items():
            self.assertEqual(str(native.config), str(native_after[name].config))

    def test_rwkv_math_timeout_falls_back_to_full_string_comparison(self):
        from src.rwkv_eval.answer_extract import free_response

        free_response.verify_math_answer.cache_clear()
        with (
            patch.object(free_response, "_get_math_worker") as get_worker,
            patch.object(free_response, "_stop_math_worker") as stop_worker,
        ):
            connection = get_worker.return_value
            connection.poll.return_value = False
            self.assertTrue(free_response.verify_math_answer(" 100 ", "100"))
            get_worker.assert_not_called()
            for gold, pred in (("100", "100+0"), ("100", "100 or 200"), ("1/2", r"\frac{1}{2}")):
                with self.subTest(gold=gold, pred=pred):
                    self.assertFalse(free_response.verify_math_answer(gold, pred))
                    args = connection.send.call_args.args[0]
                    self.assertEqual(args[:3], (gold, pred, False))
                    self.assertLessEqual(connection.poll.call_args.args[0], 5)
            self.assertEqual(stop_worker.call_count, 3)
            connection.poll.return_value = True
            connection.recv.side_effect = EOFError
            self.assertFalse(free_response.verify_math_answer("2", "2+0"))
            self.assertEqual(stop_worker.call_count, 4)
        free_response.verify_math_answer.cache_clear()

    def test_rwkv_math_worker_does_not_reimport_callers_main(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-"],
            input=(
                "from src.rwkv_eval.answer_extract.free_response import verify_math_answer\n"
                "assert verify_math_answer('1/4', r'\\frac{1}{4}')\n"
                "print('verified')\n"
            ),
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "verified")

    def test_rwkv_math_verification_bounds_huge_power_and_recovers(self):
        import concurrent.futures
        import time

        from src.rwkv_eval.answer_extract import free_response

        free_response.verify_math_answer.cache_clear()
        try:
            self.assertTrue(free_response.verify_math_answer("1/2", r"\frac{1}{2}"))
            worker = free_response._math_worker
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                value = executor.submit(
                    free_response.verify_math_answer, "100", r"3 \cdot 8^{\frac{\binom{200}{100}}{2} - 1}"
                ).result(timeout=10)
            self.assertFalse(value)
            self.assertLess(time.monotonic() - started, 8)
            self.assertIsNone(free_response._math_worker)
            self.assertIsNone(free_response._math_connection)
            self.assertTrue(free_response.verify_math_answer("1/3", r"\frac{1}{3}"))
            self.assertIsNot(free_response._math_worker, worker)
        finally:
            free_response._stop_math_worker()
            free_response.verify_math_answer.cache_clear()

    def test_rwkv_emoji_checker_preserves_native_valid_results(self):
        from lighteval.tasks.tasks.ifbench.instructions import EmojiSentenceChecker, RWKVEmojiSentenceChecker

        for sentences in (
            ["Done. 😊"],
            ["A sentence.", "😊 Another sentence. 😄"],
            ["Missing emoji."],
            ["Done. 😊", "Missing emoji."],
            [],
        ):
            with (
                self.subTest(sentences=sentences),
                patch(
                    "lighteval.tasks.tasks.ifbench.instructions.instructions_util.split_into_sentences",
                    return_value=sentences,
                ),
            ):
                self.assertEqual(
                    RWKVEmojiSentenceChecker("emoji").check_following("response"),
                    EmojiSentenceChecker("emoji").check_following("response"),
                )
        for sentences in ([")"], ["!"], ["   "], ["Missing emoji.", ")"]):
            with (
                self.subTest(sentences=sentences),
                patch(
                    "lighteval.tasks.tasks.ifbench.instructions.instructions_util.split_into_sentences",
                    return_value=sentences,
                ),
            ):
                self.assertFalse(RWKVEmojiSentenceChecker("emoji").check_following("response"))
                with self.assertRaises(IndexError):
                    EmojiSentenceChecker("emoji").check_following("response")

    def test_rwkv_ifbench_strict_loose_and_registry_are_private(self):
        from lighteval.tasks.tasks.ifbench.instructions import EmojiSentenceChecker, RWKVEmojiSentenceChecker
        from lighteval.tasks.tasks.ifbench.instructions_registry import INSTRUCTION_DICT
        from lighteval.tasks.tasks.ifbench.main import IFBench, RWKVIFBench
        from src.rwkv_eval.main import RWKVRegistry

        original = dict(INSTRUCTION_DICT)
        emoji_id = next(key for key, checker in original.items() if checker is EmojiSentenceChecker)
        native_before = Registry(tasks="ifbench_test,ifbench_multiturn").load_tasks()
        private = RWKVRegistry(tasks="ifbench_test,ifbench_multiturn").load_tasks()
        native_after = Registry(tasks="ifbench_test,ifbench_multiturn").load_tasks()
        for name, task in private.items():
            checker = task.metrics[0].sample_level_fn
            self.assertIsInstance(checker, RWKVIFBench)
            self.assertIs(checker.instruction_dict[emoji_id], RWKVEmojiSentenceChecker)
            # Native task hashes stringify the checker mapping. Relocation must
            # preserve the serialized identity of retained response caches.
            expected_dict = str(checker.instruction_dict).replace(
                repr(RWKVEmojiSentenceChecker), "<class 'src.rwkv_eval.main.RWKVEmojiSentenceChecker'>"
            )
            self.assertEqual(str(checker), f"RWKVIFBench(instruction_dict={expected_dict})")
            self.assertEqual(native_before[name].config.version, task.config.version)
            for native in (native_before[name], native_after[name]):
                self.assertIs(type(native.metrics[0].sample_level_fn), IFBench)
                self.assertIsNone(native.metrics[0].sample_level_fn.instruction_dict)
                self.assertEqual(vars(native.metrics[0].sample_level_fn), {})
                self.assertEqual(str(native.metrics[0].sample_level_fn), "IFBench()")
            self.assertEqual(task.metrics[0].metric_name, native_before[name].metrics[0].metric_name)
            self.assertEqual(task.config.solver, native_before[name].config.solver)
        self.assertEqual(INSTRUCTION_DICT, original)
        self.assertTrue(all(checker.instruction_dict[key] is cls for key, cls in original.items() if key != emoji_id))
        doc = Doc(
            query="Use an emoji after every sentence.",
            choices=[""],
            gold_index=0,
            specific={"instruction_id_list": [emoji_id], "kwargs": [{}]},
        )
        with patch(
            "lighteval.tasks.tasks.ifbench.instructions.instructions_util.split_into_sentences",
            side_effect=lambda value: value.splitlines(),
        ):
            bad = ModelResponse(text=["Missing emoji.\n)"], text_post_processed=["Missing emoji.\n)"])
            self.assertEqual(
                RWKVIFBench().compute(copy.deepcopy(doc), bad),
                {
                    "prompt_level_strict_acc": 0,
                    "inst_level_strict_acc": [False],
                    "prompt_level_loose_acc": 0,
                    "inst_level_loose_acc": [False],
                },
            )
            with self.assertRaises(IndexError):
                IFBench().compute(copy.deepcopy(doc), bad)
            loose = ModelResponse(text=["Done. 😊\n)"], text_post_processed=["Done. 😊\n)"])
            self.assertEqual(
                RWKVIFBench().compute(copy.deepcopy(doc), loose),
                {
                    "prompt_level_strict_acc": 0,
                    "inst_level_strict_acc": [False],
                    "prompt_level_loose_acc": 1,
                    "inst_level_loose_acc": [True],
                },
            )
            for value in ("Done. 😊", "Missing emoji.", ""):
                output = ModelResponse(text=[value], text_post_processed=[value])
                self.assertEqual(
                    RWKVIFBench().compute(copy.deepcopy(doc), output), IFBench().compute(copy.deepcopy(doc), output)
                )
        self.assertEqual(INSTRUCTION_DICT, original)


if __name__ == "__main__":
    unittest.main()
