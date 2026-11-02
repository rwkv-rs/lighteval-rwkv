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

from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def mathqa_prompt(line, task_name: str = None):
    if "options" in line:
        options = line["options"]
        if isinstance(options, str):
            options = (
                ast.literal_eval(options)
                if options.lstrip().startswith("[")
                else re.split(r",\s*(?=[a-e]\s*\))", options.strip(), flags=re.IGNORECASE)
            )
        options = [re.sub(r"^\s*[a-e]\s*\)\s*", "", option, flags=re.IGNORECASE).strip() for option in options]
    else:
        options = [str(line[f"option_{label}"]).strip() for label in "abcde"]
    query = f"Problem: {line['Problem']}\n"
    query += "Options:\n"
    query += "".join(f"{key}) {choice}\n" for key, choice in zip("abcde", options))
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {choice}" for choice in options],
        gold_index=["a", "b", "c", "d", "e"].index(line["correct"]),
    )


mathqa = LightevalTaskConfig(
    name="mathqa",
    prompt_function=mathqa_prompt,
    hf_repo="allenai/math_qa",
    hf_subset="default",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    mathqa,
]
