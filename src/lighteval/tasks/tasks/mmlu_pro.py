"""
name:
MMLU Pro

dataset:
TIGER-Lab/MMLU-Pro

abstract:
MMLU-Pro dataset is a more robust and challenging massive multi-task
understanding dataset tailored to more rigorously benchmark large language
models' capabilities. This dataset contains 12K complex questions across various
disciplines.

languages:
english

tags:
general-knowledge, knowledge, multiple-choice

paper:
https://arxiv.org/abs/2406.01574

starred:
true
"""

from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def mmlu_pro_prompt_function(line, task_name: str = None):
    choices = "\n".join([f"{letter}: {choice}" for letter, choice in zip(ascii_uppercase, line["options"])])

    query = f"{line['question']}\n\n{choices}"

    return Doc(
        task_name=task_name,
        query=query,
        choices=list(ascii_uppercase[: len(line["options"])]),
        gold_index=line["answer_index"],
        instruction="",
    )


def record_to_sample(record):
    return Sample(input=record["question"], target=record["answer"], choices=record["options"])


mmlu_pro = LightevalTaskConfig(
    name="mmlu_pro",
    prompt_function=mmlu_pro_prompt_function,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
    evaluation_splits=("test",),
    few_shots_split="validation",
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
)

TASKS_TABLE = [mmlu_pro]
