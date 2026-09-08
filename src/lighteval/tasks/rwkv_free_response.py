# MIT License

from __future__ import annotations

from math_verify import parse, verify

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.timeout import timeout


_FREE_RESPONSE_TASK_PREFIXES = ("arithmetic:", "math:")
_FREE_RESPONSE_TASK_NAMES = ("asdiv", "gsm_plus", "aimo_progress_prize_1")


def is_rwkv_free_response(task_full_name: str) -> bool:
    leaf = task_full_name.rsplit("|", 1)[0]
    return leaf.startswith(_FREE_RESPONSE_TASK_PREFIXES) or leaf in _FREE_RESPONSE_TASK_NAMES


class RWKVFreeResponseMatch(SampleLevelComputation):
    """Short-answer/math judge mirroring Albatross's eval_math500.py verify_one.

    RWKV completions often show arithmetic as a trailing equation line (e.g.
    "230 - 601 = -371") instead of Albatross's boxed-answer convention, so a
    bare math_verify.parse() picks up the first operand instead of the result.
    _parsed_candidates keeps verify_one's algorithm as the primary path and
    only falls back to the last equation line when the text has neither a
    boxed answer nor an explicit "answer" callout.
    """

    @staticmethod
    @timeout(5)
    def _parse(text: str):
        return parse(text)

    @staticmethod
    @timeout(5)
    def _verify(gold, prediction) -> bool:
        return bool(verify(gold, prediction, strict=False))

    @classmethod
    def _parse_or_empty(cls, text: str):
        try:
            return cls._parse(text)
        except Exception:
            return []

    @classmethod
    def _parsed_candidates(cls, text: str):
        normalized = text.replace("−", "-").replace("＝", "=")
        equality_lines = [line for line in normalized.splitlines() if "=" in line]
        candidates = [normalized]
        if equality_lines and "\\boxed" not in normalized and "answer" not in normalized.lower():
            candidates.insert(0, equality_lines[-1])
        return [parsed for candidate in candidates if (parsed := cls._parse_or_empty(candidate))]

    def compute(self, doc: Doc, model_response, **_kwargs) -> float:
        gold = self._parse_or_empty(f"$\\boxed{{{doc.get_golds()[0]}}}$")
        candidates = self._parsed_candidates(model_response.final_text[0] if model_response.final_text else "")
        doc.specific = dict(doc.specific or {}, extracted_predictions=[str(candidates[0][0])] if candidates else [])
        for prediction in candidates:
            try:
                if gold and self._verify(gold, prediction):
                    doc.specific["extracted_predictions"] = [str(prediction[0])]
                    return 1.0
            except Exception:
                continue
        return 0.0

    @classmethod
    def extract_answer(cls, doc: Doc, model_response) -> str:
        extracted = (doc.specific or {}).get("extracted_predictions")
        if extracted is not None:
            return str(extracted[0]) if extracted else ""
        candidates = cls._parsed_candidates(model_response.final_text[0] if model_response.final_text else "")
        return str(candidates[0][0]) if candidates else ""


def rwkv_free_response_metrics(metric) -> tuple[SampleLevelMetric, ...]:
    names = (metric.metric_name,) if isinstance(metric.metric_name, str) else tuple(metric.metric_name)
    grouped = not isinstance(metric.metric_name, str)
    return tuple(
        SampleLevelMetric(
            metric_name=name,
            sample_level_fn=RWKVFreeResponseMatch(),
            category=SamplingMethod.GENERATIVE,
            corpus_level_fn=metric.corpus_level_fn[name] if grouped else metric.corpus_level_fn,
            higher_is_better=metric.higher_is_better[name] if grouped else metric.higher_is_better,
        )
        for name in names
    )
