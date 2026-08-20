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
# Endpoints reported as a difference between two timepoints rather than as a
# level. Matched against the outcome title, description and timeframe, since
# CT.gov states it in whichever of the three the sponsor chose.
_CHANGE_PATTERNS = (
    re.compile(r"\bchange\s+(?:from|in|of)\b", flags=re.IGNORECASE),
    re.compile(r"\bchange\s+from\s+baseline\b", flags=re.IGNORECASE),
    re.compile(
        r"\b(?:mean|median|absolute|percent(?:age)?)\s+change\b", flags=re.IGNORECASE
    ),
    re.compile(
        r"\b(?:difference|reduction|improvement)\s+from\s+baseline\b",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\bbaseline\s+to\s+(?:week|day|month|year|end)\b", flags=re.IGNORECASE),
    re.compile(r"\bdelta\b", flags=re.IGNORECASE),
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


def is_change_from_baseline(*texts: str | None) -> bool:
    """Whether the endpoint is scored as a change rather than as a level."""
    return any(
        pattern.search(text) for text in texts if text for pattern in _CHANGE_PATTERNS
    )


def infer_outcome_bounds(*texts: str | None) -> OutcomeBounds | None:
    """Infer one unambiguous, explicitly described score or scale range."""
    # A stated range describes the instrument's scale, i.e. the range of a
    # LEVEL. An endpoint scored as change-from-baseline spans
    # [-(max-min), +(max-min)] instead, so applying the scale range to it would
    # reject every improvement -- for a 1-49 instrument, the whole negative half.
    # Deriving the change span from the level range is only valid when both
    # measurements use the full scale, which the text does not tell us, so
    # refuse and let `outcome_bounds` configuration supply them explicitly.
    if is_change_from_baseline(*texts):
        return None

    candidates = _find_ranges(texts, _RANGE_PATTERNS)
    numeric_ranges = _find_ranges(texts, (_ANY_NUMERIC_RANGE,))

    if len(candidates) != 1 or candidates != numeric_ranges:
        return None

    minimum, maximum = next(iter(candidates))
    return OutcomeBounds(minimum=minimum, maximum=maximum, source="description")
