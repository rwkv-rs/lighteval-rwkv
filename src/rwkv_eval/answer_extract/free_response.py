"""Answer extraction for RWKV reasoning and short-answer evaluations."""

from __future__ import annotations

import re


_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_BOXED = re.compile(r"\\boxed\s*\{")
_FINAL = re.compile(
    r"(?im)^\s*(?:final\s+answer|the\s+answer|answer|final|therefore|thus|so)"
    r"\s*(?:is\s*)?(?::|=)?\s*(.+?)\s*$"
)
_LATEX_EXPRESSION = re.compile(r"\\(?:sqrt|frac|dfrac|tfrac)\s*(?:\{[^{}]*\}){1,2}")
_MATH_TOKEN = re.compile(
    r"\\(?:frac|dfrac|tfrac)\s*\{[^{}]*\}\s*\{[^{}]*\}"
    r"|[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*%|\s*/\s*\d+)?"
)


def extract_free_response(text: str, *, require_think_close: bool = False) -> str:
    """Remove RWKV reasoning tags and assistant-message spillover."""
    if require_think_close and "</think>" not in text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think" in text:
        # An unterminated reasoning trace reached the output limit.  It is a
        # failed answer, not a valid fallback response.
        return ""
    if "\nUser:" in text:
        text = text.split("\nUser:", 1)[0]
    if "\nAssistant:" in text:
        text = text.rsplit("\nAssistant:", 1)[1]
    text = text.strip()
    if text.startswith("Assistant:"):
        text = text[len("Assistant:") :].lstrip()
    if text.startswith(">"):
        text = text[1:].lstrip()
    return text


def _boxed_content(text: str) -> str | None:
    """Extract the final balanced ``\\boxed{...}`` expression."""
    valid: list[tuple[int, int, str]] = []
    for match in _BOXED.finditer(text):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    valid.append((index, match.start(), text[start:index].strip()))
                    break
    if not valid:
        return None
    last_end = max(item[0] for item in valid)
    return min((item for item in valid if item[0] == last_end), key=lambda item: item[1])[2]


def extract_math_answer(text: str) -> str:
    """Extract the concise final answer used by MATH/MATH-500 graders."""
    text = extract_free_response(text)
    boxed = _boxed_content(text)
    if boxed is not None:
        return boxed
    marked = _FINAL.findall(text)
    if marked:
        return marked[-1].strip().strip("`$ ")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    final_line = lines[-1] if lines else text.strip()
    latex = _LATEX_EXPRESSION.search(final_line)
    if latex:
        return latex.group(0).strip()
    if re.fullmatch(r"[A-Za-z0-9\s^_{}()+=*/.,%-]+", final_line) and re.search(r"[=^+*/]", final_line):
        return final_line.strip("`$ .")
    candidates = _MATH_TOKEN.findall(final_line or text)
    return candidates[-1].strip() if candidates else final_line


def extract_code_answer(text: str) -> str:
    """Extract the final fenced program used by LiveCodeBench."""
    text = extract_free_response(text)
    blocks = _CODE_BLOCK.findall(text)
    return blocks[-1].strip() if blocks else ""
