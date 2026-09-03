"""
name:
Med

dataset:
lighteval/med_mcqa, lighteval/med_paragraph_simplification, bigbio/med_qa

abstract:
A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering

languages:
english

tags:
health, medical, field:medical

paper:
https://medmcqa.github.io/
"""

from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def med_mcqa_prompt(line, task_name: str = None):
    query = f"Give a letter answer among A, B, C or D.\nQuestion: {line['question']}\n"
    query += "".join(
        [
            f"{key}. {choice}\n"
            for key, choice in zip(ascii_uppercase, [line["opa"], line["opb"], line["opc"], line["opd"]])
        ]
    )
    query += "Answer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=list(ascii_uppercase)[:4],
        gold_index=line["cop"] - 1,
        instruction="Give a letter answer among A, B, C or D.\n",
    )


def med_paragraph_simplification_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"###\nArticle:{line['query']}\n\nSummarize the above article in 10 sentences.\n",
        gold_index=0,
        choices=[line["answer"]],
    )


def med_qa_prompt(line, task_name: str = None):
    choices = [option["key"] for option in line["options"]]
    instruction = f"Give a letter answer among {', '.join(choices[:-1])} or {choices[-1]}.\n"
    query = f"{instruction}Question: {line['question']}\n"
    query += "".join([f"{option['key']}. {option['value']}\n" for option in line["options"]])
    query += "Answer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=list(ascii_uppercase).index(line["answer_idx"]),
        instruction=instruction,
    )


med_mcqa = LightevalTaskConfig(
    name="med_mcqa",
    prompt_function=med_mcqa_prompt,
    hf_repo="lighteval/med_mcqa",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)


med_paragraph_simplification = LightevalTaskConfig(
    name="med_paragraph_simplification",
    prompt_function=med_paragraph_simplification_prompt,
    hf_repo="lighteval/med_paragraph_simplification",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=512,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)


med_qa = LightevalTaskConfig(
    name="med_qa",
    prompt_function=med_qa_prompt,
    hf_repo="bigbio/med_qa",
    hf_subset="default",
    hf_data_files={
        split: (
            "https://huggingface.co/datasets/bigbio/med_qa/resolve/"
            f"e04abdc0672c54547fa1dbe36cfefc000e4f2657/med_qa_en_source/{split}/0000.parquet"
        )
        for split in ("train", "validation", "test")
    },
    hf_revision="e04abdc0672c54547fa1dbe36cfefc000e4f2657",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    med_mcqa,
    med_paragraph_simplification,
    med_qa,
]
