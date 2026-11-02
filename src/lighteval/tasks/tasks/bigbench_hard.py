"""
name:
Bigbench Hard

dataset:
lighteval/bbh

abstract:

languages:

tags:
reasoning

paper:
"""

import re
from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from src.rwkv_eval.configs import RWKVLightevalTaskConfig


def bbh_prompt(line, task_name: str = None):
    line = {k: v for k, v in line.items() if v is not None}

    query = line.get("task_prefix", "")
    query += line.get("example_input_prefix", "\nQuestion: ")
    query += line["input"]
    query += line.get("choice_prefix", "\n  Choices: ")
    query += "".join([f"\n{key}. {choice}" for key, choice in zip(ascii_uppercase, line["choices"])])
    query += line.get("example_output_prefix", "\nAnswer: ")

    return Doc(
        task_name=task_name,
        query=query,
        choices=list(ascii_uppercase[: len(line["choices"])]),
        gold_index=line["target_idx"],
        instruction=line.get("task_prefix", None),
    )


def rwkv_bbh_prompt(line, task_name: str = None):
    """Prototype: bbh_prompt; correct tracking data and balance answer positions.

    Each copied RWKV task cycles its gold position by the dataset row index.
    The gold option's text is retained; only its displayed position changes.
    """
    line = dict(line)
    if "target" in line:
        # The canonical BBH mirror stores the question and labelled options
        # together. Adapt that record to the native formatter's fields.
        question, options = line["input"].rsplit("\nOptions:\n", 1)
        line["input"] = question
        line["choices"] = re.findall(r"(?m)^\([A-Z]\) (.+)$", options)
        line["target_idx"] = ascii_uppercase.index(line["target"].strip("()"))
    choices = line["choices"]
    target = line["__index"] % len(choices)
    offset = (line["target_idx"] - target) % len(choices)
    line["choices"] = choices[offset:] + choices[:offset]
    line["target_idx"] = target
    return bbh_prompt(line, task_name)


def record_to_sample(record):
    query = f"{record.get('task_prefix', '')}\n{record['input']}"
    target = ascii_uppercase[record["target_idx"]]
    choices = record["choices"]

    return Sample(input=query, target=target, choices=choices)


causal_judgment = LightevalTaskConfig(
    name="bigbench_hard:causal_judgment",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="causal_judgement",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

date_understanding = LightevalTaskConfig(
    name="bigbench_hard:date_understanding",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="date_understanding",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

disambiguation_qa = LightevalTaskConfig(
    name="bigbench_hard:disambiguation_qa",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="disambiguation_qa",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

geometric_shapes = LightevalTaskConfig(
    name="bigbench_hard:geometric_shapes",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="geometric_shapes",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

logical_deduction_five_objects = LightevalTaskConfig(
    name="bigbench_hard:logical_deduction_five_objects",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="logical_deduction_five_objects",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

logical_deduction_seven_objects = LightevalTaskConfig(
    name="bigbench_hard:logical_deduction_seven_objects",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="logical_deduction_seven_objects",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

logical_deduction_three_objects = LightevalTaskConfig(
    name="bigbench_hard:logical_deduction_three_objects",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="logical_deduction_three_objects",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

movie_recommendation = LightevalTaskConfig(
    name="bigbench_hard:movie_recommendation",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="movie_recommendation",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

navigate = LightevalTaskConfig(
    name="bigbench_hard:navigate",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="navigate",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

reasoning_about_colored_objects = LightevalTaskConfig(
    name="bigbench_hard:reasoning_about_colored_objects",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="reasoning_about_colored_objects",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

ruin_names = LightevalTaskConfig(
    name="bigbench_hard:ruin_names",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="ruin_names",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

salient_translation_error_detection = LightevalTaskConfig(
    name="bigbench_hard:salient_translation_error_detection",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="salient_translation_error_detection",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

snarks = LightevalTaskConfig(
    name="bigbench_hard:snarks",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="snarks",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

sports_understanding = LightevalTaskConfig(
    name="bigbench_hard:sports_understanding",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="sports_understanding",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

temporal_sequences = LightevalTaskConfig(
    name="bigbench_hard:temporal_sequences",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="temporal_sequences",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

tracking_shuffled_objects_five_objects = LightevalTaskConfig(
    name="bigbench_hard:tracking_shuffled_objects_five_objects",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="tracking_shuffled_objects_five_objects",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

tracking_shuffled_objects_seven_objects = LightevalTaskConfig(
    name="bigbench_hard:tracking_shuffled_objects_seven_objects",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="tracking_shuffled_objects_seven_objects",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

tracking_shuffled_objects_three_objects = LightevalTaskConfig(
    name="bigbench_hard:tracking_shuffled_objects_three_objects",
    prompt_function=bbh_prompt,
    hf_repo="lighteval/bbh",
    hf_subset="tracking_shuffled_objects_three_objects",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

TASKS_TABLE = [
    causal_judgment,
    date_understanding,
    disambiguation_qa,
    geometric_shapes,
    logical_deduction_five_objects,
    logical_deduction_seven_objects,
    logical_deduction_three_objects,
    movie_recommendation,
    navigate,
    reasoning_about_colored_objects,
    ruin_names,
    salient_translation_error_detection,
    snarks,
    sports_understanding,
    temporal_sequences,
    tracking_shuffled_objects_five_objects,
    tracking_shuffled_objects_seven_objects,
    tracking_shuffled_objects_three_objects,
]


RWKV_TASKS_TABLE = [
    RWKVLightevalTaskConfig(
        task,
        prompt_function=rwkv_bbh_prompt,
        version=task.version + 1,
        **(
            {
                "hf_repo": "lukaemon/bbh",
                "hf_revision": "982bb89fd79532a8ac676a61fc42eb1aeec63f99",
                "hf_subset": "tracking_shuffled_objects_three_objects",
                "hf_avail_splits": ["test"],
                "evaluation_splits": ["test"],
            }
            if task.name == "bigbench_hard:tracking_shuffled_objects_three_objects"
            else {}
        ),
    )
    for task in TASKS_TABLE
]
