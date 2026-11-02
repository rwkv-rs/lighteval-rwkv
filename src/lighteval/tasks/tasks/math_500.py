"""
name:
Math 500

dataset:
HuggingFaceH4/MATH-500

abstract:
This dataset contains a subset of 500 problems from the MATH benchmark that
OpenAI created in their Let's Verify Step by Step paper.

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2305.20050

starred:
true
"""

from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.aime import MATH_PROMPT_TEMPLATE as RWKV_MATH_PROMPT_TEMPLATE
from src.rwkv_eval.answer_extract.free_response import extract_math_answer
from src.rwkv_eval.configs import RWKVExactMatches, RWKVLightevalTaskConfig


MATH_QUERY_TEMPLATE = """
Solve the following problem. The final line of your response MUST be of the following format:
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the final answer. Think step by step before answering.

{prompt}
""".strip()


def math_500_prompt(line, task_name: str = None):
    query = MATH_QUERY_TEMPLATE.format(prompt=line["problem"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f"ANSWER: {line['solution']}"],
        gold_index=0,
    )


def record_to_sample(record):
    query = record["problem"]
    target = record["answer"]
    return Sample(input=query, target=target)


def rwkv_math_500_prompt(line, task_name: str = None):
    doc = math_500_prompt(line, task_name)
    doc.query = RWKV_MATH_PROMPT_TEMPLATE.format(prompt=line["problem"])
    doc.choices = [extract_math_answer(str(gold)) for gold in doc.choices]
    if any(gold is None for gold in doc.get_golds()):
        raise RuntimeError("RWKV task has an unextractable gold answer")
    return doc


math_500 = LightevalTaskConfig(
    name="math_500",
    prompt_function=math_500_prompt,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
    ],
    version=2,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=model_graded_fact(),
)

TASKS_TABLE = [
    math_500,
]


# Private RWKV overrides; native models retain TASKS_TABLE.
RWKV_TASKS_TABLE = [
    RWKVLightevalTaskConfig(
        task,
        sample_scoring_function=RWKVExactMatches(),
        prompt_function=rwkv_math_500_prompt,
        version=task.version + 1,
        solver=[prompt_template(RWKV_MATH_PROMPT_TEMPLATE), generate(cache=True)],
        generation_size=24576,
        prompt_template="bot",
        metrics=[Metrics.avg_at_n_math(sample_params={"n": 1})],
    )
    for task in TASKS_TABLE
]
