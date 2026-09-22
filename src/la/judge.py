"""Scorers, including the model-graded kind whose noise this system measures.

Two families:

  ExactMatch / Contains   deterministic. Zero variance by construction, which
                          makes them the control: run the audit with one of
                          these and any residual variance you see is the model
                          under test, not the grader.

  ModelJudge              a model grades the answer. This is what most agent
                          evaluations actually use, and it is the part nobody
                          measures the variance of.

A judge is configured with a temperature and an explicit sample index, so the
same answer can be graded repeatedly and the disagreement recorded. Grading
once at temperature zero hides the problem rather than removing it: a
zero-temperature judge is still sensitive to prompt order, formatting, and its
own version, and a panel of judges disagrees regardless of temperature.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from .providers import Provider

__all__ = ["Score", "Scorer", "ExactMatch", "Contains", "NumericMatch",
           "ModelJudge", "JudgePanel", "DEFAULT_RUBRIC", "build_scorer"]


class Score:
    """One grade, plus where it came from."""

    __slots__ = ("value", "grader", "raw", "parsed", "cost_usd")

    def __init__(self, value: float, grader: str, raw: str = "",
                 parsed: bool = True, cost_usd: float = 0.0):
        self.value = float(value)
        self.grader = grader
        self.raw = raw
        self.parsed = bool(parsed)
        self.cost_usd = float(cost_usd)

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "grader": self.grader,
                "parsed": self.parsed, "cost_usd": self.cost_usd}

    def __repr__(self) -> str:
        return "Score(%.3f by %s%s)" % (
            self.value, self.grader, "" if self.parsed else ", UNPARSED")


class Scorer:
    name = "scorer"

    def score(self, prompt: str, answer: str, reference: Optional[str],
              sample: int = 0) -> Score:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# deterministic scorers: the control condition
# ---------------------------------------------------------------------------

class ExactMatch(Scorer):
    """Normalised string equality. Deterministic, so it contributes no
    grading variance at all -- which is exactly what makes it useful as a
    baseline to compare a model judge against."""

    name = "exact_match"

    def __init__(self, case_sensitive: bool = False, strip_punctuation: bool = True):
        self.case_sensitive = case_sensitive
        self.strip_punctuation = strip_punctuation

    def _normalise(self, text: str) -> str:
        text = text.strip()
        if not self.case_sensitive:
            text = text.lower()
        if self.strip_punctuation:
            text = re.sub(r"[^\w\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def score(self, prompt, answer, reference, sample=0) -> Score:
        if reference is None:
            raise ValueError("ExactMatch needs a reference answer")
        value = float(self._normalise(answer) == self._normalise(reference))
        return Score(value, self.name)


class Contains(Scorer):
    """True when the reference appears in the answer. Also deterministic."""

    name = "contains"

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    def score(self, prompt, answer, reference, sample=0) -> Score:
        if reference is None:
            raise ValueError("Contains needs a reference answer")
        haystack = answer if self.case_sensitive else answer.lower()
        needle = reference if self.case_sensitive else reference.lower()
        return Score(float(needle.strip() in haystack), self.name)


_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


class NumericMatch(Scorer):
    """Compare the last number in the answer to the reference number.

    The convention used for grade-school math benchmarks: a model that reasons
    aloud states its result last, so the final number is the answer. Commas,
    currency symbols and trailing full stops are stripped, and a tolerance
    handles the float formatting differences that would otherwise fail a
    correct answer.

    Deterministic, so it contributes no grading variance. That makes it the
    control: run the same items through this and through a model judge, and
    the difference between the two variance decompositions is what the judge
    is adding.
    """

    name = "numeric_match"

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = float(tolerance)

    @staticmethod
    def extract(text: str) -> Optional[float]:
        matches = _NUMBER.findall(text.replace("$", "").replace("%", ""))
        if not matches:
            return None
        for candidate in reversed(matches):
            cleaned = candidate.replace(",", "").rstrip(".")
            try:
                return float(cleaned)
            except ValueError:
                continue
        return None

    def score(self, prompt, answer, reference, sample=0) -> Score:
        if reference is None:
            raise ValueError("NumericMatch needs a reference answer")
        target = self.extract(reference)
        got = self.extract(answer)
        if target is None:
            raise ValueError("reference %r contains no number" % reference)
        if got is None:
            # No number at all is a wrong answer, and a flagged one: a model
            # that stops answering numerically is a finding, not a blank.
            return Score(0.0, self.name, raw=answer[:200], parsed=False)
        return Score(float(abs(got - target) <= self.tolerance), self.name)


# ---------------------------------------------------------------------------
# model-graded
# ---------------------------------------------------------------------------

DEFAULT_RUBRIC = """You are grading one answer to one question.

Question:
{prompt}

{reference_block}Answer to grade:
{answer}

Decide whether the answer is correct. Ignore style, length and formatting;
judge only whether it is right.

Reply with exactly one line, nothing else:
GRADE: CORRECT
or
GRADE: INCORRECT"""

REFERENCE_BLOCK = """Reference answer:
{reference}

"""

_GRADE = re.compile(r"GRADE\s*:\s*(CORRECT|INCORRECT)", re.IGNORECASE)
_FALLBACK = re.compile(r"\b(CORRECT|INCORRECT)\b", re.IGNORECASE)


class ModelJudge(Scorer):
    """A model grades the answer.

    `temperature` above zero and a varying `sample` are what let the same
    answer be graded more than once. The run records every grade, so
    disagreement between repeats becomes a measured quantity instead of an
    assumption.

    An unparseable reply is recorded as such and scored 0, never silently
    dropped: a judge that fails to follow its own output format on 8% of items
    is a finding, and dropping those items would erase it.
    """

    def __init__(
        self,
        provider: Provider,
        model: str,
        rubric: str = DEFAULT_RUBRIC,
        temperature: float = 0.0,
        max_tokens: int = 16,
        name: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.rubric = rubric
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.name = name or ("judge:%s@T%.1f" % (model, temperature))

    def build_prompt(self, prompt: str, answer: str,
                     reference: Optional[str]) -> str:
        block = REFERENCE_BLOCK.format(reference=reference) if reference else ""
        return self.rubric.format(prompt=prompt, answer=answer,
                                  reference_block=block, reference=reference or "")

    def score(self, prompt, answer, reference, sample=0) -> Score:
        text = self.build_prompt(prompt, answer, reference)
        completion = self.provider.complete(
            [{"role": "user", "content": text}],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            sample=sample,
        )
        reply = completion.text.strip()

        match = _GRADE.search(reply) or _FALLBACK.search(reply)
        if match is None:
            return Score(0.0, self.name, raw=reply, parsed=False,
                         cost_usd=completion.cost_usd)
        value = float(match.group(1).upper() == "CORRECT")
        return Score(value, self.name, raw=reply, parsed=True,
                     cost_usd=completion.cost_usd)


class JudgePanel(Scorer):
    """Several judges on the same answer; the mean is the score.

    A panel is how grader noise stops being invisible. One judge gives a number
    with no way to tell whether another judge would have said the same;
    a panel's spread is that answer, and the individual grades are kept so the
    spread can be analysed rather than just averaged away.
    """

    def __init__(self, judges: Sequence[Scorer], name: str = "panel"):
        if not judges:
            raise ValueError("a panel needs at least one judge")
        self.judges = list(judges)
        self.name = name
        self.last_scores: List[Score] = []

    def score(self, prompt, answer, reference, sample=0) -> Score:
        scores = [j.score(prompt, answer, reference, sample=sample)
                  for j in self.judges]
        self.last_scores = scores
        mean = sum(s.value for s in scores) / len(scores)
        cost = sum(s.cost_usd for s in scores)
        all_parsed = all(s.parsed for s in scores)
        return Score(mean, self.name, parsed=all_parsed, cost_usd=cost)

    def agreement(self) -> Optional[float]:
        """Share of judges agreeing with the panel majority on the last item."""
        if not self.last_scores:
            return None
        values = [s.value for s in self.last_scores]
        mean = sum(values) / len(values)
        majority = 1.0 if mean >= 0.5 else 0.0
        return sum(1 for v in values if v == majority) / len(values)


def build_scorer(spec: str, provider: Optional[Provider] = None,
                 temperature: float = 0.0) -> Scorer:
    """Build a scorer from a short spec string.

        exact                       -> ExactMatch
        contains                    -> Contains
        numeric                     -> NumericMatch
        judge:openai/gpt-4o-mini    -> ModelJudge on that model
    """
    if spec == "exact":
        return ExactMatch()
    if spec == "contains":
        return Contains()
    if spec == "numeric":
        return NumericMatch()
    if spec.startswith("judge:"):
        if provider is None:
            raise ValueError("a model judge needs a provider")
        return ModelJudge(provider, spec[len("judge:"):],
                          temperature=temperature)
    raise ValueError(
        "unknown scorer %r. Use 'exact', 'contains', 'numeric', or "
        "'judge:<model>'." % spec
    )
