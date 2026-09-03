import random
from unittest.mock import Mock

import pytest

from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.rwkv_answer_extractor import (
    convert_rwkv_choice,
    extract_rwkv_choice_answer,
    is_rwkv_choice,
)
from lighteval.tasks.tasks.aimo import task as aimo_progress_prize_1
from lighteval.tasks.tasks.arithmetic import TASKS_TABLE as ARITHMETIC_TASKS
from lighteval.tasks.tasks.asdiv import asdiv_prompt
from lighteval.tasks.tasks.gpqa import gpqa_instruct_prompt
from lighteval.tasks.tasks.mathqa import mathqa, mathqa_prompt
from lighteval.tasks.tasks.med import med_qa, med_qa_prompt
from lighteval.tasks.tasks.mmlu_pro import mmlu_pro_prompt_function
from lighteval.tasks.tasks.olympiade_bench import olympiad_bench_prompt


def test_med_qa_uses_official_parquet_schema_and_does_not_repeat_letter_options():
    doc = med_qa_prompt(
        {
            "question": "Question?",
            "options": [
                {"key": "A", "value": "one"},
                {"key": "B", "value": "two"},
                {"key": "C", "value": "three"},
                {"key": "D", "value": "four"},
                {"key": "E", "value": "five"},
            ],
            "answer_idx": "E",
        },
        "med_qa|0",
    )
    assert med_qa.hf_subset == "default"
    assert med_qa.hf_revision == "e04abdc0672c54547fa1dbe36cfefc000e4f2657"
    assert set(med_qa.hf_data_files) == {"train", "validation", "test"}
    assert doc.query.count("A. one") == 1
    assert "Give a letter answer among A, B, C, D or E." in doc.query


def test_new_math_tasks_use_published_parquet_layouts():
    for task in ARITHMETIC_TASKS:
        subset = task.name.replace(":", "_")
        assert task.hf_subset == "default"
        assert task.hf_data_files == {"validation": f"{subset}/validation/0000.parquet"}
        assert task.hf_revision == "14413db3567723ff76bc468508333b5c7a9dcf5d"

    assert aimo_progress_prize_1.hf_subset == "default"
    assert aimo_progress_prize_1.hf_revision == "6e33ae2d1995bcbac59b18536b561669b15ff0b1"
    assert mathqa.hf_revision == "fafb9f7ee5b9ec4da9499f9c4177a4c91389f2d6"


def test_asdiv_prompt_exposes_one_gold_choice():
    doc = asdiv_prompt(
        {"body": "There are 20 apples.", "question": "How many?", "answer": "20 (apples)"},
        "asdiv",
    )

    assert doc.choices == ["20"]


@pytest.mark.parametrize(
    "serialized_options",
    [
        "a ) one , b ) two , c ) three , d ) four , e ) five",
        "['a ) one', 'b ) two', 'c ) three', 'd ) four', 'e ) five']",
    ],
)
def test_mathqa_prompt_reads_both_published_option_encodings(serialized_options):
    doc = mathqa_prompt(
        {"Problem": "Choose three.", "options": serialized_options, "correct": "c"},
        "mathqa",
    )

    assert doc.choices == [" one", " two", " three", " four", " five"]
    assert doc.gold_index == 2


def test_mmlu_pro_exposes_exact_native_letter_choices_for_rwkv_conversion():
    doc = mmlu_pro_prompt_function(
        {
            "question": "Question?",
            "options": [f"option {letter}" for letter in "ABCDEFGHIJ"],
            "answer_index": 9,
        },
        "mmlu_pro|0",
    )

    assert doc.choices == list("ABCDEFGHIJ")
    assert "where LETTER is one of ABCDEFGHIJ" in doc.query
    assert "J: option J" in doc.query


def test_gpqa_choice_order_is_stable_per_question(monkeypatch):
    line = {
        "Question": "Question?",
        "Correct Answer": "correct",
        "Incorrect Answer 1": "wrong 1",
        "Incorrect Answer 2": "wrong 2",
        "Incorrect Answer 3": "wrong 3",
    }
    other = dict(line, Question="Other question?")

    monkeypatch.setattr(random, "randint", Mock(side_effect=AssertionError("global RNG must not be used")))
    random.seed(1)
    first = gpqa_instruct_prompt(line, "gpqa:diamond|0")
    gpqa_instruct_prompt(other, "gpqa:diamond|0")
    for _ in range(20):
        random.random()
    second = gpqa_instruct_prompt(line, "gpqa:diamond|0")
    gpqa_instruct_prompt(other, "gpqa:diamond|0")
    third = gpqa_instruct_prompt(line, "gpqa:diamond|0")

    assert second.query == first.query
    assert second.gold_index == first.gold_index
    assert third.query == first.query
    assert third.gold_index == first.gold_index


def test_olympiad_bench_does_not_emit_an_empty_specific_struct():
    doc = olympiad_bench_prompt(
        {
            "subject": "Math",
            "language": "English",
            "unit": None,
            "is_multiple_answer": False,
            "answer_type": "Numerical",
            "final_answer": "1",
            "question": "Compute 1.",
        },
        "olympiad_bench:OE_TO_maths_en_COMP|0",
    )

    assert doc.specific is None


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("Answer: B", "two"),
        ("<think>x</think>Answer: <B>", "two"),
        ("<think>x</think>The correct answer is B. two.", "two"),
        ("<think>x</think>正确答案是 **B**。", "two"),
        ("<think>x</think><answer>B</answer>", "two"),
        ("<think>x</think>Answer: <choice B>", "two"),
        ("<think>x</think>Answer: <final>B</final>", "two"),
        ('<think>x</think>Answer: <span class="choice">B</span>', "two"),
        ("<think>x</think>Answer: <letter>B</letter>", "two"),
        ("<think>x</think><letter>C</letter>", "three"),
        ('<think>x</think>Answer: {"answer": "B"}', "two"),
        ("<think>x</think>Answer: <function=finish>\n<parameter=message>\nB\n</parameter>\n</function>", "two"),
        ("<think>x</think>C. three", "three"),
        ("<think>x</think><<B>>", "two"),
        ("<think>x</think><<C> three>", "three"),
        ("<think>x</think>因此，不会发生的反应是选项(D)。", "four"),
        (">reasoning</think>答案：D\n</think>", "four"),
        ("<think>x</think>选项A直接给出了定义，其他选项过于宽泛（C）或不够具体（D）。", "one"),
        ("<think>x</think>This aligns with option B.", "two"),
        ("<think>x</think>This makes (B) the correct effect.", "two"),
        ("<think>x</think>Answer: A\nAnswer: B", "two"),
        ("<think>x</think>Answer: A\nI choose B", "two"),
        ("<think>x</think>C.", "three"),
        ("<think>x</think>The correct answer is A, as the evidence shows.", "one"),
        ("<think>x</think>因此，正确选项为B。\nAnswer: <", "two"),
        ("<think>x</think>Answer: C, A", '["one","three"]'),
        ("<think>x</think>Answer: <letter>", ""),
        ("<think>x</think>Answer: E", ""),
        ("<think>x</think>I am unsure", ""),
    ],
)
def test_choice_extraction_handles_dashboard_formats(completion, expected):
    assert extract_rwkv_choice_answer(completion, ["one", "two", "three", "four"]) == expected


def test_choice_extraction_uses_unique_full_choice_text_as_last_fallback():
    choices = ["store bile", "produce digestive enzymes", "filter blood"]

    assert (
        extract_rwkv_choice_answer("<think>x</think>The gallbladder's function is to store bile.", choices)
        == "store bile"
    )
    assert extract_rwkv_choice_answer("<think>x</think>Both store bile and filter blood are discussed.", choices) == ""


def test_choice_extraction_rejects_conflicting_direct_option_references():
    assert extract_rwkv_choice_answer("<think>x</think>选项A正确，选项B也正确。", ["one", "two", "three"]) == ""


def test_choice_extraction_uses_query_options_for_native_letter_choices():
    query = "Question?\nA。first option\nB。collective defense\nC。third option\nAnswer:"

    assert extract_rwkv_choice_answer("<think>x</think>Collective defense", ["A", "B", "C"], query) == "B"
    assert (
        extract_rwkv_choice_answer(
            "<think>x</think>The correct answer is <collective defense>.", ["A", "B", "C"], query
        )
        == "B"
    )


def test_choice_extraction_uses_consistent_unique_option_text_across_lines():
    query = "Question?\nA。chlorine\nB。sodium bicarbonate\nC。silicon dioxide\nD。magnesium\nAnswer:"
    completion = (
        "<think>x</think>Silicon dioxide does not react with sodium hydroxide.\n"
        "Silicon dioxide is inert, while chlorine, sodium bicarbonate, and magnesium react."
    )

    assert extract_rwkv_choice_answer(completion, ["A", "B", "C", "D"], query) == "C"


def test_choice_extraction_normalizes_numeric_range_option_text():
    query = "Question?\nA. 14-28 weeks\nB. 3-9 weeks\nC. 28-37weeks\nD. 8-14weeks\nAnswer:"

    assert (
        extract_rwkv_choice_answer(
            "<think>x</think>The dangerous period is specifically between 8 and 14 weeks of gestation.",
            ["A", "B", "C", "D"],
            query,
        )
        == "D"
    )


def test_choice_extraction_prefers_the_longest_unique_option_text():
    query = "Question?\nA. /\nB. //\nC. %\nAnswer:"

    assert extract_rwkv_choice_answer("<think>x</think>Python floor division uses //.", ["A", "B", "C"], query) == "B"


@pytest.mark.parametrize(
    ("benchmark", "labels", "completion", "expected"),
    [
        ("truthfulqa", list("AB"), "Answer: B", "B"),
        ("winogrande", ["option1", "option2"], "Answer: A", "option1"),
        ("gpqa", list("ABCD"), "Answer: D", "D"),
        ("medqa", list("ABCDE"), "Answer: E", "E"),
        ("mmlu_pro", list("ABCDEFGHIJ"), "Answer: J", "J"),
        ("ceval", list("ABCD"), "答案：C", "C"),
        ("agieval", list("ABCD"), "选项(D)", "D"),
    ],
)
def test_rwkv_choice_extractor_covers_supported_benchmark_shapes(benchmark, labels, completion, expected):
    assert benchmark
    assert extract_rwkv_choice_answer(completion, labels) == expected


def test_rwkv_choice_conversion_supports_canonical_a_through_j():
    doc = Doc(
        query="Question?",
        choices=[f"choice {label}" for label in "ABCDEFGHIJ"],
        gold_index=9,
        sampling_methods=[SamplingMethod.LOGPROBS],
        generation_size=1,
        stop_sequences=["\\n"],
    )

    assert is_rwkv_choice(doc)

    convert_rwkv_choice(doc)

    assert doc.sampling_methods == [SamplingMethod.GENERATIVE]
    assert "J. choice J" in doc.query
    assert "Answer: <letter>" in doc.query
    assert doc.generation_size == 8192
    assert doc.stop_sequences == []


def test_rwkv_choice_extractor_rejects_cross_line_conflicts():
    completion = "<think>x</think>Answer: A\\nAnswer: B"

    assert extract_rwkv_choice_answer(completion, ["one", "two"]) == "two"
    assert extract_rwkv_choice_answer("<think>x</think>one\\ntwo", ["one", "two"]) == ""


def test_rwkv_choice_extractor_uses_first_think_boundary_and_removes_repeated_closers():
    assert extract_rwkv_choice_answer(">reasoning</think>答案：D\\n</think>", list("ABCD")) == "D"


def test_rwkv_choice_extractor_returns_empty_for_truncated_or_unextractable_output():
    assert extract_rwkv_choice_answer("", list("ABCD")) == ""
    assert extract_rwkv_choice_answer("<think>unfinished reasoning", list("ABCD")) == ""
