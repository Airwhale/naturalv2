"""Constants used throughout the pipeline."""

TREATMENT_COL_NAME = "treatment_taken"
INCLUSION_COL_NAME = "meets_inclusion_criteria"
OUTCOME_COL_NAME = "outcome_category"

# Share of records a validation gate may reject before the event stops being
# routine. One bad parse in a thousand is noise; a tenth of the sample means the
# bounds or the extraction are wrong, and whatever estimate follows was computed
# on what survived.
HIGH_REJECTION_RATE = 0.10


def rejection_log_level(logger, n_rejected: int, rejection_rate: float):
    """Pick a log method for a validation gate, by rate rather than by count.

    Returns ``logger.info`` when nothing was rejected, ``logger.warning`` below
    ``HIGH_REJECTION_RATE``, and ``logger.error`` at or above it.
    """
    if rejection_rate >= HIGH_REJECTION_RATE:
        return logger.error
    return logger.warning if n_rejected else logger.info
