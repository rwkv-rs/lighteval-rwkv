"""
name:
AIMO Progress Prize 1

dataset:
lighteval/aimo_progress_prize_1

abstract:
Task to evaluate LLMs on the training set of the Kaggle AIMO competition:

languages:
english

tags:
math, reasoning

paper:
"""

from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics, math_scorer
from lighteval.metrics.normalizations import math_normalizer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.aime import MATH_PROMPT_TEMPLATE as RWKV_MATH_PROMPT_TEMPLATE
from src.rwkv_eval.answer_extract.free_response import extract_math_answer
from src.rwkv_eval.configs import RWKVExactMatches, RWKVLightevalTaskConfig


def aimo_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        choices=[str(line["answer"])],
        gold_index=0,
        query=line["problem"],
    )


def record_to_sample(record):
    return Sample(input=record["problem"], target=str(record["answer"]))


def rwkv_aimo_prompt(line, task_name: str = None):
    doc = aimo_prompt(line, task_name)
    doc.query = RWKV_MATH_PROMPT_TEMPLATE.format(prompt=line["problem"])
    doc.choices = [extract_math_answer(str(gold)) for gold in doc.choices]
    if any(gold is None for gold in doc.get_golds()):
        raise RuntimeError("RWKV task has an unextractable gold answer")
    return doc


task = LightevalTaskConfig(
    name="aimo_progress_prize_1",
    prompt_function=aimo_prompt,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=math_scorer(),
    hf_subset="",
    hf_repo="lighteval/aimo_progress_prize_1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="sequential",
    metrics=[
        Metrics.exact_match(sample_params={"normalize_gold": math_normalizer, "normalize_pred": math_normalizer})
    ],
    generation_size=2048,
    stop_sequence=None,
)

# STORE YOUR EVALS
TASKS_TABLE = [task]


# Private RWKV overrides; native models retain TASKS_TABLE.
RWKV_TASKS_TABLE = [
    RWKVLightevalTaskConfig(
        task,
        sample_scoring_function=RWKVExactMatches(),
        prompt_function=rwkv_aimo_prompt,
        version=task.version + 1,
        solver=[prompt_template(RWKV_MATH_PROMPT_TEMPLATE), generate(cache=True)],
        generation_size=24576,
        metrics=[Metrics.avg_at_n_math(sample_params={"n": 1})],
    )
    for task in TASKS_TABLE
]
