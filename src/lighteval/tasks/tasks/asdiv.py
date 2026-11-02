"""
name:
Asdiv

dataset:
EleutherAI/asdiv

abstract:
ASDiv is a dataset for arithmetic reasoning that contains 2,000+ questions
covering addition, subtraction, multiplication, and division.

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2410.12853
"""

from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from lighteval.metrics.metrics import Metrics, math_scorer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from src.rwkv_eval.configs import RWKVExactMatches, RWKVLightevalTaskConfig


def asdiv_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['body']}\nQuestion:{line['question']}\nAnswer:",
        choices=line["answer"].split(" (")[0],
        gold_index=[0],
    )


def rwkv_asdiv_prompt(line, task_name: str = None):
    """Prototype: asdiv_prompt; preserve native target components."""
    doc = asdiv_prompt(line, task_name)
    doc.choices = ["; ".join(part.split(" (")[0].strip() for part in line["answer"].split(";"))]
    return doc


def record_to_sample(record):
    query = f"{record['body']}\n{record['question']}"
    target = record["answer"].split(" (")[0]
    return Sample(input=query, target=target)


asdiv = LightevalTaskConfig(
    name="asdiv",
    prompt_function=asdiv_prompt,
    hf_repo="EleutherAI/asdiv",
    hf_subset="asdiv",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=math_scorer(),
)

TASKS_TABLE = [asdiv]


# Private RWKV overrides; native models retain TASKS_TABLE.
RWKV_TASKS_TABLE = [
    RWKVLightevalTaskConfig(
        task,
        sample_scoring_function=RWKVExactMatches(allow_text=True),
        prompt_function=rwkv_asdiv_prompt,
        cot_mode="FakeCoT",
    )
    for task in TASKS_TABLE
]
