"""
name:
Mathqa

dataset:
allenai/math_qa

abstract:
large-scale dataset of math word problems.  Our dataset is gathered by using a
new representation language to annotate over the AQuA-RAT dataset with
fully-specified operational programs.  AQuA-RAT has provided the questions,
options, rationale, and the correct options.

languages:
english

tags:
math, qa, reasoning

paper:
https://arxiv.org/abs/1905.13319
"""

import ast
import re

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def mathqa_prompt(line, task_name: str = None):
    serialized_options = line["options"]
    labeled_options = (
        ast.literal_eval(serialized_options)
        if serialized_options.startswith("[")
        else re.split(r",\s*(?=[a-e]\s*\))", serialized_options)
    )
    option_matches = [
        re.fullmatch(rf"{label}\s*\)\s*(.*)", option) for label, option in zip("abcde", labeled_options, strict=True)
    ]
    if len(option_matches) != 5 or any(match is None for match in option_matches):
        raise ValueError("MathQA options must contain labels a through e in order")
    choices = [match.group(1).strip() for match in option_matches if match is not None]

    query = f"Problem: {line['Problem']}\n"
    query += "Options:\n"
    query += "".join(f"{key}) {choice}\n" for key, choice in zip("abcde", choices, strict=True))
    query += "Answer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {choice}" for choice in choices],
        gold_index="abcde".index(line["correct"]),
    )


mathqa = LightevalTaskConfig(
    name="mathqa",
    prompt_function=mathqa_prompt,
    hf_repo="allenai/math_qa",
    hf_subset="default",
    hf_revision="fafb9f7ee5b9ec4da9499f9c4177a4c91389f2d6",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=1,
)

TASKS_TABLE = [
    mathqa,
]
