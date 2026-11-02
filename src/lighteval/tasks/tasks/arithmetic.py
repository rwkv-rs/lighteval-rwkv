"""
name:
Arithmetic

dataset:
EleutherAI/arithmetic

abstract:
A small battery of 10 tests that involve asking language models a simple
arithmetic problem in natural language.

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2005.14165
"""

from pathlib import Path

from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from lighteval.metrics.metrics import Metrics, math_scorer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from src.rwkv_eval.answer_extract.free_response import extract_math_answer
from src.rwkv_eval.configs import RWKVExactMatches, RWKVLightevalTaskConfig


# The Hub's original Python builder is no longer exposed by the current
# datasets service.  Keep the benchmark data identical by loading the
# published JSONL files directly instead of relying on the removed builder.
_ARITHMETIC_DATA_URL = "https://huggingface.co/datasets/EleutherAI/arithmetic/resolve/main/data"


def arithmetic_prompt(line, task_name: str = None):
    sample = record_to_sample(line)
    return Doc(task_name=task_name, query=sample.input, choices=[sample.target], gold_index=[0])


def record_to_sample(record):
    context = record["context"].strip().replace("\n\n", "\n").replace("Q:", "Question:").replace("A:", "Answer:")
    return Sample(input=context, target=record["completion"])


def rwkv_arithmetic_prompt(line, task_name: str = None):
    doc = arithmetic_prompt(line, task_name)
    doc.choices = [extract_math_answer(str(gold)) for gold in doc.choices]
    if any(gold is None for gold in doc.get_golds()):
        raise RuntimeError("RWKV task has an unextractable gold answer")
    return doc


arithmetic_1dc = LightevalTaskConfig(
    name="arithmetic:1dc",
    prompt_function=arithmetic_prompt,
    hf_repo="json",
    hf_subset="default",
    hf_data_files={"validation": f"{_ARITHMETIC_DATA_URL}/single_digit_three_ops.jsonl"},
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=math_scorer(),
)

arithmetic_2da = LightevalTaskConfig(
    name="arithmetic:2da",
    prompt_function=arithmetic_prompt,
    hf_repo="json",
    hf_subset="default",
    hf_data_files={"validation": f"{_ARITHMETIC_DATA_URL}/two_digit_addition.jsonl"},
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=math_scorer(),
)

arithmetic_2dm = LightevalTaskConfig(
    name="arithmetic:2dm",
    prompt_function=arithmetic_prompt,
    hf_repo="json",
    hf_subset="default",
    hf_data_files={"validation": f"{_ARITHMETIC_DATA_URL}/two_digit_multiplication.jsonl"},
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=math_scorer(),
)

arithmetic_2ds = LightevalTaskConfig(
    name="arithmetic:2ds",
    prompt_function=arithmetic_prompt,
    hf_repo="json",
    hf_subset="default",
    hf_data_files={"validation": f"{_ARITHMETIC_DATA_URL}/two_digit_subtraction.jsonl"},
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=math_scorer(),
)

arithmetic_3da = LightevalTaskConfig(
    name="arithmetic:3da",
    prompt_function=arithmetic_prompt,
    hf_repo="json",
    hf_subset="default",
    hf_data_files={"validation": f"{_ARITHMETIC_DATA_URL}/three_digit_addition.jsonl"},
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=math_scorer(),
)

arithmetic_3ds = LightevalTaskConfig(
    name="arithmetic:3ds",
    prompt_function=arithmetic_prompt,
    hf_repo="json",
    hf_subset="default",
    hf_data_files={"validation": f"{_ARITHMETIC_DATA_URL}/three_digit_subtraction.jsonl"},
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=math_scorer(),
)

arithmetic_4da = LightevalTaskConfig(
    name="arithmetic:4da",
    prompt_function=arithmetic_prompt,
    hf_repo="json",
    hf_subset="default",
    hf_data_files={"validation": f"{_ARITHMETIC_DATA_URL}/four_digit_addition.jsonl"},
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=math_scorer(),
)

arithmetic_4ds = LightevalTaskConfig(
    name="arithmetic:4ds",
    prompt_function=arithmetic_prompt,
    hf_repo="json",
    hf_subset="default",
    hf_data_files={"validation": f"{_ARITHMETIC_DATA_URL}/four_digit_subtraction.jsonl"},
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=math_scorer(),
)

arithmetic_5da = LightevalTaskConfig(
    name="arithmetic:5da",
    prompt_function=arithmetic_prompt,
    hf_repo="json",
    hf_subset="default",
    hf_data_files={"validation": f"{_ARITHMETIC_DATA_URL}/five_digit_addition.jsonl"},
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=math_scorer(),
)

arithmetic_5ds = LightevalTaskConfig(
    name="arithmetic:5ds",
    prompt_function=arithmetic_prompt,
    hf_repo="json",
    hf_subset="default",
    hf_data_files={"validation": f"{_ARITHMETIC_DATA_URL}/five_digit_subtraction.jsonl"},
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=math_scorer(),
)

TASKS_TABLE = [
    arithmetic_1dc,
    arithmetic_2da,
    arithmetic_2dm,
    arithmetic_2ds,
    arithmetic_3da,
    arithmetic_3ds,
    arithmetic_4da,
    arithmetic_4ds,
    arithmetic_5da,
    arithmetic_5ds,
]


class RWKVArithmeticTaskConfig(RWKVLightevalTaskConfig):
    """Resolve cached JSONL files when a selected offline task reads its data files."""

    @property
    def hf_data_files(self):
        from huggingface_hub import constants, hf_hub_download

        if not constants.HF_HUB_OFFLINE:
            return self._hf_data_files
        return {
            split: hf_hub_download(
                repo_id="EleutherAI/arithmetic",
                filename=f"data/{Path(url).name}",
                repo_type="dataset",
                local_files_only=True,
            )
            for split, url in self._hf_data_files.items()
        }

    @hf_data_files.setter
    def hf_data_files(self, data_files):
        self._hf_data_files = data_files


# Private RWKV overrides; native models retain TASKS_TABLE.
RWKV_TASKS_TABLE = [
    RWKVArithmeticTaskConfig(
        task, sample_scoring_function=RWKVExactMatches(), prompt_function=rwkv_arithmetic_prompt, cot_mode="FakeCoT"
    )
    for task in TASKS_TABLE
]
