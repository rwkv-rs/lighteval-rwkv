"""
name:
Math

dataset:
DigitalLearningGmbH/MATH-lighteval

abstract:

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2305.20050
"""

from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import math_normalizer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.aime import MATH_PROMPT_TEMPLATE
from src.rwkv_eval.answer_extract.free_response import extract_math_answer
from src.rwkv_eval.configs import RWKVExactMatches, RWKVLightevalTaskConfig


def math_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['problem']}\nAnswer:",
        choices=[f" {line['solution']}"],
        gold_index=0,
    )


def rwkv_math_prompt(line, task_name: str = None):
    doc = math_prompt(line, task_name)
    doc.query = MATH_PROMPT_TEMPLATE.format(prompt=line["problem"])
    doc.choices = [extract_math_answer(str(gold)) for gold in doc.choices]
    if any(gold is None for gold in doc.get_golds()):
        raise RuntimeError("RWKV task has an unextractable gold answer")
    return doc


math_algebra = LightevalTaskConfig(
    name="math:algebra",
    prompt_function=math_prompt,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="algebra",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[
        Metrics.maj_at_n(
            sample_params={
                "n": 4,
                "strip_strings": True,
                "normalize_pred": math_normalizer,
                "normalize_gold": math_normalizer,
            }
        ),
    ],
    stop_sequence=["\n"],
    version=1,
)

math_counting_and_probability = LightevalTaskConfig(
    name="math:counting_and_probability",
    prompt_function=math_prompt,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="counting_and_probability",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[
        Metrics.maj_at_n(
            sample_params={
                "n": 4,
                "strip_strings": True,
                "normalize_pred": math_normalizer,
                "normalize_gold": math_normalizer,
            }
        ),
    ],
    stop_sequence=["\n"],
    version=1,
)

math_geometry = LightevalTaskConfig(
    name="math:geometry",
    prompt_function=math_prompt,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="geometry",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[
        Metrics.maj_at_n(
            sample_params={
                "n": 4,
                "strip_strings": True,
                "normalize_pred": math_normalizer,
                "normalize_gold": math_normalizer,
            }
        ),
    ],
    stop_sequence=["\n"],
    version=1,
)

math_intermediate_algebra = LightevalTaskConfig(
    name="math:intermediate_algebra",
    prompt_function=math_prompt,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="intermediate_algebra",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[
        Metrics.maj_at_n(
            sample_params={
                "n": 4,
                "strip_strings": True,
                "normalize_pred": math_normalizer,
                "normalize_gold": math_normalizer,
            }
        ),
    ],
    stop_sequence=["\n"],
    version=1,
)

math_number_theory = LightevalTaskConfig(
    name="math:number_theory",
    prompt_function=math_prompt,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="number_theory",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[
        Metrics.maj_at_n(
            sample_params={
                "n": 4,
                "strip_strings": True,
                "normalize_pred": math_normalizer,
                "normalize_gold": math_normalizer,
            }
        ),
    ],
    stop_sequence=["\n"],
    version=1,
)

math_prealgebra = LightevalTaskConfig(
    name="math:prealgebra",
    prompt_function=math_prompt,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="prealgebra",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[
        Metrics.maj_at_n(
            sample_params={
                "n": 4,
                "strip_strings": True,
                "normalize_pred": math_normalizer,
                "normalize_gold": math_normalizer,
            }
        ),
    ],
    stop_sequence=["\n"],
    version=1,
)

math_precalculus = LightevalTaskConfig(
    name="math:precalculus",
    prompt_function=math_prompt,
    hf_repo="DigitalLearningGmbH/MATH-lighteval",
    hf_subset="precalculus",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[
        Metrics.maj_at_n(
            sample_params={
                "n": 4,
                "strip_strings": True,
                "normalize_pred": math_normalizer,
                "normalize_gold": math_normalizer,
            }
        ),
    ],
    stop_sequence=["\n"],
    version=1,
)

TASKS_TABLE = [
    math_algebra,
    math_counting_and_probability,
    math_geometry,
    math_intermediate_algebra,
    math_number_theory,
    math_prealgebra,
    math_precalculus,
]


# Private RWKV overrides; native models retain TASKS_TABLE.
RWKV_TASKS_TABLE = [
    RWKVLightevalTaskConfig(
        task,
        sample_scoring_function=RWKVExactMatches(),
        prompt_function=rwkv_math_prompt,
        version=task.version + 1,
        solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
        generation_size=24576,
        prompt_template="bot",
    )
    for task in TASKS_TABLE
]
