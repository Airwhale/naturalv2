"""Constants used throughout the pipeline."""

from pydantic import BaseModel, ConfigDict, Field


TREATMENT_COL_NAME = "treatment_taken"
INCLUSION_COL_NAME = "meets_inclusion_criteria"
OUTCOME_COL_NAME = "outcome_category"


class SampleValidationConfig(BaseModel):
    """Validated policy for rejecting sampled records before estimation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    high_rejection_rate: float = Field(gt=0.0, le=1.0)
    allow_high_rejection_rate: bool = False
