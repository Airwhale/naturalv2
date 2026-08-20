"""Constants used throughout the pipeline."""

import logging
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field


TREATMENT_COL_NAME = "treatment_taken"
INCLUSION_COL_NAME = "meets_inclusion_criteria"
OUTCOME_COL_NAME = "outcome_category"

# Default share of records a validation gate may reject before the event stops
# being routine. One bad parse in a thousand is noise; a tenth of the sample
# means the bounds or the extraction are wrong.
HIGH_REJECTION_RATE = 0.10


class SampleValidationConfig(BaseModel):
    """Validated policy for rejecting sampled records before estimation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    high_rejection_rate: float = Field(default=HIGH_REJECTION_RATE, gt=0.0, le=1.0)
    allow_high_rejection_rate: bool = False


def rejection_log_level(
    logger: logging.Logger,
    n_rejected: int,
    rejection_rate: float,
    *,
    high_rejection_rate: float = HIGH_REJECTION_RATE,
) -> Callable[..., None]:
    """Pick a log method for a validation gate, by rate rather than by count.

    Returns ``logger.info`` when nothing was rejected, ``logger.warning`` below
    ``high_rejection_rate``, and ``logger.error`` at or above it.
    """
    if rejection_rate >= high_rejection_rate:
        return logger.error
    return logger.warning if n_rejected else logger.info
