from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, fields
from dataclasses import field as dataclass_field
from functools import partial
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from lighteval.metrics.metrics_sample import AvgAtN, ExactMatches, PassAtK, SamplingMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import SamplingMethod
from src.rwkv_eval.answer_extract.free_response import verify_math_answer


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def rwkv_prompt(line, task_name=None, *, prompt_function, sampling_method=SamplingMethod.GENERATIVE):
    """Apply the shared RWKV answer boundary to a benchmark's formatted Doc."""
    doc = prompt_function(line, task_name)
    if doc is None:
        return None
    if sampling_method == SamplingMethod.LOGPROBS:
        if not doc.query.rstrip().endswith("\nAnswer:"):
            doc.query = doc.query.rstrip().removesuffix("Answer:").rstrip() + "\nAnswer: "
    elif doc.query.rstrip().endswith("Answer:"):
        doc.query = doc.query.rstrip()[: -len("Answer:")].rstrip() + "\n"
    return doc


class RWKVExactMatches(ExactMatches):
    """RWKV comparison, following ExactMatches.compute_one_item."""

    def __init__(self, *, allow_text: bool = False, choice: bool = False):
        super().__init__()
        self.allow_text = allow_text
        self.choice = choice

    def compute(self, doc, model_response, **kwargs) -> float:
        score = super().compute(doc, model_response, **kwargs)
        if self.choice and any(not pred for pred in model_response.final_text):
            return max(score, 1 / len(doc.choices))
        return score

    def compute_one_item(self, gold: str, pred: str) -> float:
        if self.choice:
            return super().compute_one_item(gold, pred)
        return int(verify_math_answer(gold, pred, allow_text=self.allow_text))


class RWKVAvgAtN(AvgAtN):
    """Prototype: AvgAtN; retain individual scores for publication details."""

    def __init__(self, metric, n):
        super().__init__(n=n)
        self.metric_name = metric.metric_name
        self.sample_level_fn = copy.deepcopy(metric.sample_level_fn)
        sample_fn = self.sample_level_fn
        if isinstance(sample_fn, SamplingMetric):
            sample_fn.n = 1
            if isinstance(sample_fn, PassAtK):
                sample_fn.k = 1

    def compute(self, doc, model_response, **kwargs):
        scores = []

        def compute_score(doc, model_response):
            sample_fn = self.sample_level_fn
            if isinstance(sample_fn, AvgAtN):
                score = sample_fn.compute_score(doc, model_response)
            else:
                score = sample_fn.compute(doc=doc, model_response=model_response)
                if isinstance(self.metric_name, list):
                    score = score[self.metric_name[0]]
            scores.append(float(score))
            return score

        result = AvgAtN(n=self.n, sample_scoring_function=compute_score).compute(doc, model_response, **kwargs)
        doc.specific = {**(doc.specific or {}), "rwkv_rollout_scores": scores}
        return result


class RWKVLightevalTaskConfig(LightevalTaskConfig):
    """Prototype: LightevalTaskConfig.__str__; stable RWKV cache identity."""

    def __init__(
        self,
        config: LightevalTaskConfig,
        *,
        cot_mode="CoT",
        prompt_template="assistant",
        sample_scoring_function=None,
        **kwargs,
    ):
        super().__init__(
            **{
                **{field.name: getattr(config, field.name) for field in fields(config) if field.init},
                "generation_size": 4096,
                **kwargs,
            }
        )
        if sample_scoring_function is not None:
            self.metrics = copy.deepcopy(self.metrics)
            for metric in self.metrics:
                if metric.category != SamplingMethod.GENERATIVE:
                    continue
                if isinstance(metric.sample_level_fn, SamplingMetric):
                    metric.sample_level_fn.compute_score = sample_scoring_function.compute
                else:
                    metric.sample_level_fn = sample_scoring_function
        self.cot_mode = cot_mode
        self.prompt_template = prompt_template
        methods = {metric.category for metric in self.metrics}
        if len(methods) != 1:
            raise ValueError(f"{self.name} has multiple request types")
        method = next(iter(methods))
        self.prompt_function = partial(rwkv_prompt, prompt_function=self.prompt_function, sampling_method=method)
        if method == SamplingMethod.LOGPROBS:
            from inspect_ai.scorer import choice
            from inspect_ai.solver import multiple_choice

            self.version += 1
            self.solver = [multiple_choice(cache=True)]
            self.scorer = choice()

    def __str__(self, lite: bool = False):
        if lite:
            config = copy.copy(self)
            # Native formatting names individual callables. Solver lists and
            # grouped corpus-function dictionaries must also be process-stable.
            config.solver = [solver.__qualname__ for solver in self.solver] if self.solver else self.solver
            return re.sub(r" at 0x[0-9a-fA-F]+(?=>)", "", LightevalTaskConfig.__str__(config, lite=True))
        return super().__str__(lite=lite)


@dataclass
class RwkvModel:
    arch_version: Literal["rwkv7", "rwkv7a", "rwkv7b"]
    data_version: str
    param_size: Literal["1.5b", "2.9b", "7.2b", "13.3b"]
    ctx_len: int
    is_preview: bool

    @classmethod
    def from_model_name(cls, name: str, ctx_len: int, is_preview: bool):
        parts = Path(name).name.removesuffix(".pth").lower().split("-")
        if len(parts) < 3 or parts[0] not in {"rwkv7", "rwkv7a", "rwkv7b"}:
            raise ValueError(f"cannot parse RWKV model name: {name}")
        if not re.fullmatch(r"g1[a-z0-9]*", parts[1]) or parts[2] not in {"1.5b", "2.9b", "7.2b", "13.3b"}:
            raise ValueError(f"invalid RWKV model name: {name}")
        return cls(parts[0], parts[1].capitalize(), parts[2], ctx_len, is_preview)

    @property
    def key(self):
        return (
            self.arch_version.casefold(),
            self.data_version.casefold(),
            self.param_size.casefold(),
            int(self.ctx_len),
            self.is_preview,
        )


@dataclass
class SamplingConfig:
    max_generated_tokens: int
    temp: float
    top_k: int
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    penalty_decay: float
    seed: int


@dataclass(frozen=True, slots=True)
class ModelEndpoint:
    url: str
    api_key: str
    model_name: str
    max_num_seqs: int
    ctx_len: int = 4096
    is_preview: bool = dataclass_field(kw_only=True)


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    field: Literal[
        "knowledge",
        "reasoning",
        "maths",
        "coding",
        "instruction_following",
        "agentic",
        "vision",
    ]
    selector: str


def _read_toml(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("rb") as stream:
            value = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"invalid TOML {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be a TOML table")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _api_key(value: Any, *, resolve: bool = True) -> str:
    value = _string(value, "api_key")
    match = re.fullmatch(r"\$\{([A-Z][A-Z0-9_]*)\}", value)
    if match and resolve:
        value = os.environ.get(match.group(1), "")
    if not value:
        raise ValueError(f"missing environment variable {match.group(1)} for api_key" if match else "api_key is empty")
    return value


def _url(value: Any) -> str:
    value = _string(value, "url")
    if "://" not in value:
        value = f"https://{value}"
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"invalid HTTP endpoint URL: {value}")
    return value.rstrip("/")


def read_models(path: str | Path, *, resolve_api_keys: bool = True) -> tuple[ModelEndpoint, ...]:
    raw = _read_toml(path)
    entries = raw.get("model") or raw.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected one or more [[model]] entries")
    models = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: model entry {index} must be a table")
        try:
            models.append(
                ModelEndpoint(
                    _url(entry["url"]),
                    _api_key(entry["api_key"], resolve=resolve_api_keys),
                    _string(entry["model_name"], "model_name"),
                    _positive_int(entry["max_num_seqs"], "max_num_seqs"),
                    _positive_int(entry.get("ctx_len", 4096), "ctx_len"),
                    is_preview=_boolean(entry["is_preview"], "is_preview"),
                )
            )
        except KeyError as error:
            raise ValueError(f"{path}: model entry {index} is missing {error.args[0]}") from error
    return tuple(models)


def read_benchmarks(path: str | Path) -> tuple[BenchmarkSpec, ...]:
    allowed = {"knowledge", "reasoning", "maths", "coding", "instruction_following", "agentic", "vision"}
    specs, seen = [], set()
    for field, selectors in _read_toml(path).items():
        if field not in allowed or not isinstance(selectors, list):
            raise ValueError(f"{path}: invalid benchmark field {field}")
        for selector in selectors:
            selector = _string(selector, "benchmark selector")
            if selector in seen:
                raise ValueError(f"{path}: duplicate benchmark selector {selector}")
            seen.add(selector)
            specs.append(BenchmarkSpec(field, selector))
    if not specs:
        raise ValueError(f"{path}: no benchmarks configured")
    return tuple(specs)
