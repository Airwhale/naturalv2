"""Validated metadata for continuous clinical outcomes."""

from typing import Self

from pydantic import BaseModel, ConfigDict, FiniteFloat, RootModel, model_validator


class OutcomeBounds(BaseModel):
    """Inclusive numeric bounds for one continuous outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: FiniteFloat
    maximum: FiniteFloat

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        """Require a non-empty interval."""
        if self.minimum >= self.maximum:
            raise ValueError("minimum must be less than maximum")
        return self


class OutcomeBoundsMap(RootModel[dict[str, OutcomeBounds]]):
    """Validated bounds keyed by exact outcome name."""

    model_config = ConfigDict(frozen=True)
