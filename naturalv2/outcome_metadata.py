"""Validated metadata for continuous clinical outcomes."""

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, FiniteFloat, RootModel, model_validator


_NUMBER = r"[+\-\u2212]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)"
_SEPARATOR = (
    r"(?:-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2212|\bto\b|\bthrough\b|\band\b)"
)
_QUALIFIER = r"(?:\s*\([^)]{0,80}\))?"
_RANGE_PATTERNS = (
    re.compile(
        rf"\b(?:possible\s+)?(?:scores?|scale)\s+(?:can\s+)?ranges?"
        rf"(?:\s+(?:is|from|between|of))?\s*:?\s*"
        rf"(?P<minimum>{_NUMBER}){_QUALIFIER}\s*{_SEPARATOR}\s*"
        rf"(?P<maximum>{_NUMBER})",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:scores?|scale)\s+has\s+(?:a\s+)?range"
        rf"(?:\s+(?:of|from|between))?\s*:?\s*"
        rf"(?P<minimum>{_NUMBER}){_QUALIFIER}\s*{_SEPARATOR}\s*"
        rf"(?P<maximum>{_NUMBER})",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<minimum>{_NUMBER}){_QUALIFIER}\s*{_SEPARATOR}\s*"
        rf"(?P<maximum>{_NUMBER})\s*(?:point\s+)?(?:score|scale)\b",
        flags=re.IGNORECASE,
    ),
)
_ANY_NUMERIC_RANGE = re.compile(
    rf"(?P<minimum>{_NUMBER}){_QUALIFIER}\s*{_SEPARATOR}\s*"
    rf"(?P<maximum>{_NUMBER})",
    flags=re.IGNORECASE,
)


class ConfiguredOutcomeBounds(BaseModel):
    """Validated bounds accepted from user configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: FiniteFloat
    maximum: FiniteFloat

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        """Require a non-empty interval."""
        if self.minimum >= self.maximum:
            raise ValueError("minimum must be less than maximum")
        return self


class OutcomeBounds(ConfiguredOutcomeBounds):
    """Inclusive numeric bounds with their provenance."""

    source: Literal["configured", "description"]


class ConfiguredOutcomeBoundsMap(RootModel[dict[str, ConfiguredOutcomeBounds | None]]):
    """Configured bounds keyed by exact outcome name."""

    model_config = ConfigDict(frozen=True)


class OutcomeBoundsMap(RootModel[dict[str, OutcomeBounds | None]]):
    """Persisted outcome bounds keyed by exact outcome name."""

    model_config = ConfigDict(frozen=True)


def _parse_number(value: str) -> float:
    return float(value.replace(",", "").replace("\N{MINUS SIGN}", "-"))


def _find_ranges(
    texts: tuple[str | None, ...], patterns: tuple[re.Pattern[str], ...]
) -> set[tuple[float, float]]:
    """Return increasing numeric ranges matched by any supplied pattern."""
    ranges: set[tuple[float, float]] = set()
    for text in texts:
        if not text:
            continue
        for pattern in patterns:
            for match in pattern.finditer(text):
                minimum = _parse_number(match.group("minimum"))
                maximum = _parse_number(match.group("maximum"))
                if minimum < maximum:
                    ranges.add((minimum, maximum))
    return ranges


def infer_outcome_bounds(*texts: str | None) -> OutcomeBounds | None:
    """Infer one unambiguous, explicitly described score or scale range."""
    candidates = _find_ranges(texts, _RANGE_PATTERNS)
    numeric_ranges = _find_ranges(texts, (_ANY_NUMERIC_RANGE,))

    if len(candidates) != 1 or candidates != numeric_ranges:
        return None

    minimum, maximum = next(iter(candidates))
    return OutcomeBounds(minimum=minimum, maximum=maximum, source="description")
