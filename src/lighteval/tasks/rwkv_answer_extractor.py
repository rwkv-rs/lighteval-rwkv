# MIT License

from __future__ import annotations

import json
import re

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.requests import SamplingMethod


RWKV_CHOICE_GENERATION_SIZE = 8192
_CHOICE_MARKUP = re.compile(r"\*\*|__|`+")
_CHOICE_SINGLE_LABEL = r"[A-Z](?![A-Z])"
_CHOICE_LABELS = rf"{_CHOICE_SINGLE_LABEL}(?:\s*(?:,|/|&|\+|\band\b)\s*{_CHOICE_SINGLE_LABEL})*"
_CHOICE_PATTERNS = (
    re.compile(
        rf"\\boxed\s*\{{\s*(?:\\(?:text|mathrm)\s*\{{\s*)?({_CHOICE_LABELS})\s*\}}?\s*\}}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?i:(?:final\s+answer|correct\s+answer|answer|choice|option|最终答案|正确答案|答案|选项)\s*"
        rf"(?:(?:choice|option)\s*)?(?:is\s*|would\s+be\s*|[是为]\s*|[:：=]\s*)"
        rf"(?:<letter>\s*)?(?:<\s*(?:answer|choice|b)\s*>\s*)?[\"'\[(]*\s*)"
        rf"({_CHOICE_LABELS})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?i:(?:final\s+answer|correct\s+answer|answer|choice|option|最终答案|正确答案|答案|选项)\s*"
        rf"(?:(?:choice|option)\s*)?(?:is\s*|would\s+be\s*|[是为]\s*|[:：=]\s*)<\s*)"
        rf"({_CHOICE_LABELS})\s*>",
        re.IGNORECASE,
    ),
    re.compile(
        rf"<\s*(?:answer|choice|b|final|final_answer|letter|span)(?:\s+[^>]*)?\s*>\s*"
        rf"({_CHOICE_LABELS})(?:\s*[.:]\s*[^<]*)?\s*"
        rf"</\s*(?:answer|choice|b|final|final_answer|letter|span)\s*>",
        re.IGNORECASE,
    ),
    re.compile(rf"<\s*({_CHOICE_LABELS})\s*>[^<]+</\s*[A-Z]\s*>", re.IGNORECASE),
    re.compile(rf"<\s*(?:answer|choice)\s*[:=]?\s*({_CHOICE_LABELS})\s*>", re.IGNORECASE),
    re.compile(rf"<<\s*({_CHOICE_LABELS})\s*>\s*[^<>]+\s*>", re.IGNORECASE),
    re.compile(rf"\\?[\"']answer\\?[\"']\s*:\s*\\?[\"']({_CHOICE_LABELS})\\?[\"']", re.IGNORECASE),
    re.compile(
        rf"(?i:(?:(?:choice|option)\s*)?\(?\s*)({_CHOICE_LABELS})"
        rf"(?i:\s*\)?\s+is\s+(?:the\s+)?(?:final\s+|correct\s+)?answer)",
    ),
)
_CHOICE_FALLBACK_PATTERNS = (
    re.compile(
        rf"(?i:\b(?:choose|select|pick)\s+(?:(?:choice|option|answer)\s*)?[:=]?\s*\(?\s*)"
        rf"({_CHOICE_LABELS})(?i:\s*\)?\b)",
    ),
    re.compile(
        rf"(?i:\b(?:corresponds?|maps?)\s+to\s+(?:(?:choice|option|answer)\s*)?\(?\s*)"
        rf"({_CHOICE_LABELS})(?i:\s*\)?\b)",
    ),
    re.compile(
        rf"(?i:\b(?:aligns?|matches?)\s+with\s+(?:(?:choice|option|answer)\s*)?\(?\s*)"
        rf"({_CHOICE_LABELS})(?i:\s*\)?\b)",
    ),
    re.compile(
        rf"(?i:\b(?:therefore|thus|hence|so|consequently)[,:]?\s+(?:the\s+)?"
        rf"(?:(?:correct|final)\s+)?(?:answer|choice|option)\s+(?:is|would\s+be)\s+\(?\s*)"
        rf"({_CHOICE_LABELS})(?i:\s*\)?\b)",
    ),
    re.compile(rf"(?i:\b\(?\s*)({_CHOICE_LABELS})(?i:\s*\)?\s+(?:the\s+)?correct\b)"),
)
_CHOICE_BARE = re.compile(
    rf"\s*(?:final\s+answer\s*[:=]?\s*)?[\[(<]*({_CHOICE_LABELS})[\])>]*"
    r"(?:\s*[.:：](?:\s*\S.*)?)?\s*",
    re.IGNORECASE,
)
_CHOICE_TEXT_EXPLICIT = re.compile(
    r"(?:final\s+answer|correct\s+answer|answer|最终答案|正确答案|答案)\s*"
    r"(?:is\s*|would\s+be\s*|是\s*|[:：=]\s*)(.+?)(?:[.。]\s*$|$)",
    re.IGNORECASE | re.MULTILINE,
)


def rwkv_choice_gold_indices(doc) -> tuple[int, ...] | None:
    gold = doc.gold_index
    indices = tuple(gold) if isinstance(gold, (list, tuple)) else (gold,)
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(
            not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(doc.choices)
            for index in indices
        )
    ):
        return None
    return tuple(sorted(indices))


def is_rwkv_choice(doc) -> bool:
    if not (
        isinstance(doc.query, str)
        and isinstance(doc.choices, list)
        and 2 <= len(doc.choices) <= 26
        and all(isinstance(choice, str) and choice.strip() for choice in doc.choices)
        and rwkv_choice_gold_indices(doc) is not None
    ):
        return False
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(doc.choices)]
    native_letter_choice = SamplingMethod.GENERATIVE in doc.sampling_methods and [
        choice.strip().upper() for choice in doc.choices
    ] == list(labels)
    return SamplingMethod.LOGPROBS in doc.sampling_methods or native_letter_choice


def convert_rwkv_choice(doc) -> None:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(doc.choices)]
    gold_indices = rwkv_choice_gold_indices(doc)
    assert gold_indices is not None
    answer_format = "<letter>" if len(gold_indices) == 1 else "<letters separated by commas>"
    answer_instruction = f'After reasoning, end with "Answer: {answer_format}".'
    if answer_instruction not in doc.query:
        if [choice.strip().upper() for choice in doc.choices] == list(labels):
            doc.query = f"{doc.query.rstrip()}\n\n{answer_instruction}"
        else:
            options = "\n".join(
                f"{label}. {choice.strip()}" for label, choice in zip(labels, doc.choices, strict=True)
            )
            doc.query = f"{doc.query.rstrip()}\n\n{options}\n\n{answer_instruction}"
    doc.sampling_methods = list(
        dict.fromkeys(
            SamplingMethod.GENERATIVE if method == SamplingMethod.LOGPROBS else method
            for method in doc.sampling_methods
        )
    )
    doc.generation_size = RWKV_CHOICE_GENERATION_SIZE
    doc.stop_sequences = []
    doc.specific = dict(doc.specific or {}, rwkv_choice=True)


class RWKVChoiceExactMatches(SampleLevelComputation):
    def compute(self, doc, model_response, **_kwargs) -> float:
        gold_indices = rwkv_choice_gold_indices(doc)
        if gold_indices is None:
            return 0.0
        expected = _canonical_choice_answer(gold_indices, doc.choices)
        return float(any(prediction == expected for prediction in model_response.final_text))

    @staticmethod
    def extract_answer(_doc, model_response) -> str:
        return model_response.final_text[0] if model_response.final_text else ""


def rwkv_choice_metrics(metric):
    names = (metric.metric_name,) if isinstance(metric.metric_name, str) else tuple(metric.metric_name)
    grouped = not isinstance(metric.metric_name, str)
    return tuple(
        SampleLevelMetric(
            metric_name=name,
            sample_level_fn=RWKVChoiceExactMatches(),
            category=SamplingMethod.GENERATIVE,
            corpus_level_fn=metric.corpus_level_fn[name] if grouped else metric.corpus_level_fn,
            higher_is_better=metric.higher_is_better[name] if grouped else metric.higher_is_better,
        )
        for name in names
    )


def _parse_choice_labels(value: str, labels: str) -> tuple[int, ...] | None:
    normalized = re.sub(r"\band\b", ",", value.upper())
    parts = [part.strip() for part in re.split(r"[,/&+]", normalized)]
    if not parts or any(not part for part in parts):
        return None
    if len(parts) == 1 and " " in parts[0]:
        parts = parts[0].split()
    if any(len(part) != 1 or part not in labels for part in parts) or len(set(parts)) != len(parts):
        return None
    return tuple(sorted(labels.index(part) for part in parts))


def _canonical_choice_answer(indices: tuple[int, ...], choices: list[str]) -> str:
    selected = [choices[index] for index in indices]
    if len(selected) == 1:
        return selected[0]
    return json.dumps(selected, ensure_ascii=False, separators=(",", ":"))


def _normalized_choice_text(value: str) -> str:
    normalized = " ".join(_CHOICE_MARKUP.sub("", value).replace("<", "").replace(">", "").casefold().split())
    normalized = re.sub(r"(?<=\d)\s*(?:[-–—]|\band\b)\s*(?=\d)", "-", normalized)
    normalized = re.sub(r"(?<=\d)\s+(?=[a-z])", "", normalized)
    normalized = re.sub(r"\s+([%/.,，。])", r"\1", normalized)
    return normalized.strip(" \t\r\n.。,:：;；!?！？()[]{}\"'")


def _choice_texts(query: str | None, choices: list[str], labels: str) -> list[str]:
    if [choice.strip().upper() for choice in choices] != list(labels) or not isinstance(query, str):
        return choices
    parsed = {}
    for match in re.finditer(r"(?m)^\s*([A-Z])\s*[.)。、]\s*(.+?)\s*$", query, re.IGNORECASE):
        label = match.group(1).upper()
        if label in labels:
            parsed[label] = match.group(2)
    return [parsed.get(label, choice) for label, choice in zip(labels, choices, strict=True)]


def _choice_text_answer(text: str, choices: list[str], labels: str, query: str | None) -> tuple[int, ...] | None:
    choice_texts = _choice_texts(query, choices, labels)
    segments = [line for line in text.splitlines() if line.strip()]
    if len(segments) > 1:
        segments.append(text)
    resolved = []
    for segment in segments:
        normalized_text = _normalized_choice_text(segment)
        matched = []
        for index, (label, choice) in enumerate(zip(labels, choice_texts, strict=True)):
            normalized_choice = _normalized_choice_text(choice)
            without_label = re.sub(rf"^\s*(?:\({label}\)|{label}[.)、])\s*", "", normalized_choice, flags=re.I)
            variants = {variant for variant in (normalized_choice, without_label) if variant}
            matching_variants = [variant for variant in variants if variant in normalized_text]
            if matching_variants:
                matched.append((index, max(matching_variants, key=len)))
        maximal = [
            index
            for index, variant in matched
            if not any(variant != other and variant in other for _, other in matched)
        ]
        if len(maximal) == 1:
            resolved.append(maximal[0])
    return (resolved[0],) if resolved and len(set(resolved)) == 1 else None


def _choice_payload_answer(
    text: str,
    choices: list[str],
    labels: str,
    query: str | None,
) -> tuple[int, tuple[int, ...]] | None:
    choice_texts = _choice_texts(query, choices, labels)
    matches = []
    for match in _CHOICE_TEXT_EXPLICIT.finditer(text):
        payload = _normalized_choice_text(match.group(1))
        if not payload or payload in {"letter", "answer_letter"}:
            continue
        candidates = []
        for index, choice in enumerate(choice_texts):
            normalized_choice = _normalized_choice_text(choice)
            without_label = re.sub(
                rf"^\s*(?:\({labels[index]}\)|{labels[index]}[.)、])\s*",
                "",
                normalized_choice,
                flags=re.I,
            )
            if payload in {normalized_choice, without_label} or (
                len(payload) > 1 and (payload in without_label or without_label in payload)
            ):
                candidates.append(index)
        if len(candidates) == 1:
            matches.append((match.start(), (candidates[0],)))
    return max(matches, key=lambda item: item[0]) if matches else None


def extract_rwkv_choice_answer(raw: str, choices: list[str], query: str | None = None) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""
    _, think_closed, suffix = raw.partition("</think>")
    answer_text = _CHOICE_MARKUP.sub("", suffix.replace("</think>", "") if think_closed else raw)
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)]
    matches = [
        (match.start(), parsed)
        for pattern in (*_CHOICE_PATTERNS, *_CHOICE_FALLBACK_PATTERNS)
        for match in pattern.finditer(answer_text)
        if (parsed := _parse_choice_labels(match.group(1), labels)) is not None
    ]
    direct_options = [
        (match.start(), parsed)
        for match in re.finditer(
            rf"(?i:(?:option|选项)\s*[\(（<]*)({_CHOICE_SINGLE_LABEL})[\)）>]?",
            answer_text,
        )
        if (parsed := _parse_choice_labels(match.group(1), labels)) is not None
    ]
    if len({parsed for _, parsed in direct_options}) == 1:
        matches.append(max(direct_options, key=lambda item: item[0]))
    if payload := _choice_payload_answer(answer_text, choices, labels, query):
        matches.append(payload)
    for line_match in re.finditer(r"(?m)^.*$", answer_text):
        if match := _CHOICE_BARE.fullmatch(line_match.group()):
            if parsed := _parse_choice_labels(match.group(1), labels):
                matches.append((line_match.start(), parsed))
    if matches:
        return _canonical_choice_answer(max(matches, key=lambda item: item[0])[1], choices)
    if parsed := _choice_text_answer(answer_text, choices, labels, query):
        return _canonical_choice_answer(parsed, choices)
    return ""
