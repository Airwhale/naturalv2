"""Treatment-ownership gating for patient-authored social media reports."""

import logging
from enum import StrEnum
from typing import Literal

import pandas as pd
from omegaconf import DictConfig

from naturalv2.pipeline.constants import TREATMENT_COL_NAME
from naturalv2.pipeline.natural import PipelineContext
from naturalv2.pipeline.sample_extraction import (
    SampleExtractionStage,
    SampleTYStage,
    extract_covariates,
)
from naturalv2.sources.components.helpers import (
    build_treatment_automaton,
    extract_mentions,
    normalize_text_for_matching,
)
from naturalv2.utils import create_response_format


logger = logging.getLogger(__name__)

OWNERSHIP_DECISION_COL = "treatment_ownership"
OWNERSHIP_TREATMENT_COL = "ownership_treatment"
OWNERSHIP_REASON_COL = "ownership_reason"
OWNERSHIP_EVIDENCE_COL = "ownership_evidence"

FIRSTHAND = "Firsthand"
NOT_FIRSTHAND = "Not firsthand"
UNCLEAR = "Unclear"
NOT_APPLICABLE = "Not applicable"
UNKNOWN_TREATMENT = "Unknown"

_INPUT_ORDER_COL = "_ownership_input_order"
_MIN_USABLE_EVIDENCE = 10


class OwnershipExtractType(StrEnum):
    """Extraction artifact identifier for ownership decisions."""

    TREATMENT_OWNERSHIP = "treatment_ownership"


def _clean_text(value: object) -> str:
    """Return a trimmed string for an optional tabular value."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _build_gate_report(row: pd.Series) -> str:
    """Separate the target comment from context for the ownership classifier."""
    comment = _clean_text(row.get("report_text"))
    title = _clean_text(row.get("title"))
    context = _clean_text(row.get("initial_post"))
    return (
        "TARGET COMMENT (written by the target patient):\n"
        f"{comment}\n\n"
        "THREAD CONTEXT (written by another person):\n"
        f"Title: {title}\n"
        f"Post content: {context}"
    )


def _mentions_target_treatment(text: object, automaton) -> bool:
    """Return whether target-patient text contains a configured treatment alias."""
    normalized_text = normalize_text_for_matching(_clean_text(text))
    return bool(extract_mentions(normalized_text, automaton))


class TreatmentOwnershipGateStage(SampleExtractionStage):
    """Keep only Reddit comments whose authors personally used a target treatment.

    Submissions and non-Reddit sources pass through unchanged. Reddit comments first
    undergo a deterministic alias check against the comment text alone. The LLM then
    distinguishes firsthand use from questions, advice, plans, and third-person
    accounts. Accepted comments are reduced to commenter-authored text before any
    downstream patient extraction stage sees them.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        name: str | None = None,
        max_concurrent_workers: int | None = None,
    ) -> None:
        super().__init__(model_cfg, name, max_concurrent_workers)
        self.extract_type = OwnershipExtractType.TREATMENT_OWNERSHIP.value

    async def _classify_comments(
        self,
        eligible_comments: pd.DataFrame,
        treatment_options: list[str],
        context: PipelineContext,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Classify lexically eligible comments and return accepted decisions."""
        if eligible_comments.empty:
            empty = eligible_comments.copy()
            return empty, empty

        gate_input = eligible_comments.copy()
        gate_input["report"] = gate_input.apply(_build_gate_report, axis=1)
        response_treatments = list(
            dict.fromkeys([*treatment_options, UNKNOWN_TREATMENT])
        )
        response_format = create_response_format(
            "TreatmentOwnershipResponse",
            [
                OWNERSHIP_DECISION_COL,
                OWNERSHIP_TREATMENT_COL,
                OWNERSHIP_REASON_COL,
                OWNERSHIP_EVIDENCE_COL,
            ],
            types={
                OWNERSHIP_DECISION_COL: Literal[
                    "Firsthand", "Not firsthand", "Unclear"
                ],
                OWNERSHIP_TREATMENT_COL: Literal[*response_treatments],
                OWNERSHIP_REASON_COL: Literal[
                    "current_use",
                    "past_use",
                    "question",
                    "advice",
                    "future_plan",
                    "third_person",
                    "unclear",
                ],
                OWNERSHIP_EVIDENCE_COL: str,
            },
        )
        decisions = await extract_covariates(
            input_df=gate_input,
            pipeline_context=context,
            pipeline_stage_name=self.stage_name,
            extract_type=OwnershipExtractType.TREATMENT_OWNERSHIP,
            llm=self.llm,
            model_name=self._model_name,
            response_format=response_format,
            max_concurrent_requests=self.max_concurrent_workers,
        )
        if decisions.empty:
            return decisions, decisions.copy()

        decision_columns = {
            OWNERSHIP_DECISION_COL,
            OWNERSHIP_TREATMENT_COL,
            OWNERSHIP_REASON_COL,
            OWNERSHIP_EVIDENCE_COL,
        }
        missing_decision_columns = decision_columns - set(decisions.columns)
        if missing_decision_columns:
            missing = ", ".join(sorted(missing_decision_columns))
            raise RuntimeError(
                "Treatment ownership output is missing required fields: "
                f"{missing}. Use a fresh experiment name or clear the stale "
                "ownership artifact before resuming this pipeline."
            )

        accepted_mask = decisions[OWNERSHIP_DECISION_COL].eq(FIRSTHAND) & decisions[
            OWNERSHIP_TREATMENT_COL
        ].isin(treatment_options)
        accepted_comments = decisions.loc[accepted_mask].copy()
        accepted_comments["report"] = accepted_comments["report_text"].map(_clean_text)
        return decisions, accepted_comments

    async def process(
        self, data: pd.DataFrame, context: PipelineContext
    ) -> pd.DataFrame:
        """Filter and subject-bind Reddit comments before patient extraction."""
        if context.source_name.casefold() != "reddit":
            self.data = data.copy()
            return self.data

        required_columns = {"report", "report_type", "report_text"}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                "Treatment ownership gating requires Reddit curation columns: "
                f"{missing}."
            )

        working = data.copy()
        working[_INPUT_ORDER_COL] = range(len(working))
        comment_mask = working["report_type"].astype(str).str.casefold().eq("comment")
        comments = working.loc[comment_mask].copy()
        bypass = working.loc[~comment_mask].copy()

        treatment_options = list(context.experiment.options[TREATMENT_COL_NAME])
        aliases = context.experiment.get_all_treatment_names_for_source(
            context.source_name
        )
        if not aliases:
            raise ValueError("No treatment names or aliases are available for Reddit.")
        automaton = build_treatment_automaton(aliases)
        lexical_mask = comments["report_text"].map(
            lambda text: _mentions_target_treatment(text, automaton)
        )
        eligible_comments = comments.loc[lexical_mask].copy()
        lexical_rejections = int((~lexical_mask).sum())

        decisions, accepted_comments = await self._classify_comments(
            eligible_comments, treatment_options, context
        )

        bypass[OWNERSHIP_DECISION_COL] = NOT_APPLICABLE
        bypass[OWNERSHIP_TREATMENT_COL] = UNKNOWN_TREATMENT
        bypass[OWNERSHIP_REASON_COL] = "not_applicable"
        bypass[OWNERSHIP_EVIDENCE_COL] = ""

        self.data = (
            pd.concat([bypass, accepted_comments], axis=0, sort=False)
            .sort_values(_INPUT_ORDER_COL, kind="stable")
            .drop(columns=[_INPUT_ORDER_COL])
        )

        unclear_count = (
            int(decisions[OWNERSHIP_DECISION_COL].eq(UNCLEAR).sum())
            if not decisions.empty
            else 0
        )
        llm_rejections = len(decisions) - len(accepted_comments)
        self.add_stat("comments_total", len(comments))
        self.add_stat("comments_lexically_rejected", lexical_rejections)
        self.add_stat("comments_llm_evaluated", len(eligible_comments))
        self.add_stat("comments_llm_rejected", llm_rejections)
        self.add_stat("comments_unclear", unclear_count)
        self.add_stat("comments_accepted", len(accepted_comments))

        logger_extra = {
            "run_id": context.exp_name,
            "phase": "treatment_ownership_gate",
            "schema_id": "treatment_ownership.v1",
            "status": "complete",
            "n_comments": len(comments),
            "n_lexically_rejected": lexical_rejections,
            "n_llm_evaluated": len(eligible_comments),
            "n_llm_rejected": llm_rejections,
            "n_accepted": len(accepted_comments),
        }
        logger.info(
            "Treatment ownership gate retained %d of %d Reddit comments.",
            len(accepted_comments),
            len(comments),
            extra=logger_extra,
        )
        if len(self.data) < _MIN_USABLE_EVIDENCE:
            logger.warning(
                "Only %d report(s) remain after treatment ownership gating.",
                len(self.data),
                extra={**logger_extra, "status": "thin", "n_retained": len(self.data)},
            )
        return self.data


class OwnershipAwareSampleTYStage(SampleTYStage):
    """Apply the ownership gate's treatment after ordinary SampleTY extraction."""

    async def process(
        self, data: pd.DataFrame, context: PipelineContext
    ) -> pd.DataFrame:
        """Sample treatment/outcome, then enforce audited comment ownership."""
        ownership_input = OWNERSHIP_TREATMENT_COL in data.columns
        self.data = await super().process(data, context)
        if not ownership_input:
            return self.data
        if OWNERSHIP_TREATMENT_COL not in self.data.columns:
            raise RuntimeError(
                "SampleTY output is missing ownership fields. Use a fresh experiment "
                "name or clear stale extraction artifacts before resuming this pipeline."
            )

        treatment_options = list(context.experiment.options[TREATMENT_COL_NAME])
        override_mask = self.data[OWNERSHIP_DECISION_COL].eq(FIRSTHAND)
        invalid_treatments = self.data.loc[
            override_mask & ~self.data[OWNERSHIP_TREATMENT_COL].isin(treatment_options),
            OWNERSHIP_TREATMENT_COL,
        ]
        if not invalid_treatments.empty:
            raise ValueError(
                "Ownership gate returned unsupported treatment values: "
                f"{sorted(invalid_treatments.astype(str).unique())}."
            )

        self.data.loc[override_mask, TREATMENT_COL_NAME] = self.data.loc[
            override_mask, OWNERSHIP_TREATMENT_COL
        ]
        treatment_map = {
            treatment: index for index, treatment in enumerate(treatment_options)
        }
        discretized_column = f"{TREATMENT_COL_NAME}_discretized"
        self.data.loc[override_mask, discretized_column] = self.data.loc[
            override_mask, OWNERSHIP_TREATMENT_COL
        ].map(treatment_map)
        self.data[discretized_column] = self.data[discretized_column].astype(int)
        self.add_stat("ownership_treatment_overrides", int(override_mask.sum()))
        return self.data
