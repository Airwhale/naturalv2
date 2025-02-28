import os
from ast import literal_eval
from typing import Any, Literal, Optional

from naturalv2.evals.clinical_trial import (
    ArmGroup,
    BaselineMeasure,
    ClinicalTrial,
    MeasureGroup,
    Measurement,
    Outcome,
    OutcomeMeasure,
    OutcomeMeasureType,
    Reference,
)
from naturalv2.utils import (
    check_binary_endpoint,
    check_noncontrol,
    check_nonplacebo,
    get_nested_value,
)


class Experiment:
    def __init__(
        self,
        data_path: str,
        nct_id: str,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        self.trial = ClinicalTrial.from_json_file(
            os.path.join(data_path, f"{nct_id}.json")
        )
        self.split = split

        self.trial_path = data_path
        self.nct_id = self.trial.protocolSection.identificationModule.nctId
        self.title = self.trial.protocolSection.identificationModule.briefTitle
        self.date: Optional[str] = (
            get_nested_value(
                self.trial, "protocolSection.statusModule.completionDateStruct.date"
            )
            if self.split == "test"
            else get_nested_value(
                self.trial,
                "protocolSection.statusModule.resultsFirstPostDateStruct.date",
            )
        )
        references: Optional[list[Reference]] = get_nested_value(
            self.trial, "protocolSection.referencesModule.references"
        )
        self.references: list[str] = (
            [ref.citation for ref in references if ref.citation] if references else []
        )

        self._set_outcome_treatment_effects()

        baseline_measures: Optional[list[BaselineMeasure]] = get_nested_value(
            self.trial, "resultsSection.baselineCharacteristicsModule.measures"
        )
        self.covariate_names: list[str] = [
            base.title for base in baseline_measures or []
        ]
        self.inclusion_criteria: Optional[str] = get_nested_value(
            self.trial, "protocolSection.eligibilityModule.eligibilityCriteria"
        )

        self.data_dump: list[
            str
        ] = []  # list of paths to relevant data dumps, one per source
        self.curated_data_path = ""  # path to curated data
        self.treatment_common_names: list[str] = []
        self.outcome_common_names: list[str] = []
        self.extended_covariate_names: list[
            str
        ] = []  # inclusion-related binary variables
        self.options: dict[str, Any] = {}
        self.question_prompts: dict[str, str] = {}

        self.set_transforms()

    def _set_outcome_treatment_effects(self) -> None:
        self.outcome_treatment = []
        if (
            self.split == "test"
        ):  # use enpdoints and arms to find outcome_treatment pairs
            primary_outcomes: Optional[list[Outcome]] = get_nested_value(
                self.trial, "protocolSection.outcomesModule.primaryOutcomes"
            )
            arm_groups: Optional[list[ArmGroup]] = get_nested_value(
                self.trial, "protocolSection.armsInterventionsModule.armGroups"
            )

            outcomes: list[Outcome] = [
                outcome
                for outcome in primary_outcomes or []
                if check_binary_endpoint(outcome.measure)
            ]
            arms: list[ArmGroup] = [
                arm
                for arm in arm_groups or []
                if check_noncontrol(arm.type)
                and check_nonplacebo(arm.interventionNames)
            ]

            for outcome in outcomes:
                for i, arm1 in enumerate(arms):
                    for j, arm2 in enumerate(arms):
                        if i < j:
                            self.outcome_treatment.append(
                                (outcome.measure, (arm1.label, arm2.label))
                            )

            self.outcome_names: list[str] = [out.measure for out in outcomes]
            self.treatment_names: list[str] = [arm.label for arm in arms]
        else:  # use result information to find outcome-treatment pairs
            trial_outcome_measures: Optional[list[OutcomeMeasure]] = get_nested_value(
                self.trial,
                "resultsSection.outcomeMeasuresModule.outcomeMeasures",
            )

            outcome_measures: list[OutcomeMeasure] = [
                result
                for result in trial_outcome_measures or []
                if result.type == OutcomeMeasureType.PRIMARY
                and check_binary_endpoint(result.title)
            ]

            self.effect_sizes = []
            for result in outcome_measures:
                measure_groups: list[MeasureGroup] = [
                    cohort
                    for cohort in result.groups or []
                    if check_nonplacebo([cohort.title])
                ]

                for i, cohort1 in enumerate(measure_groups):
                    for j, cohort2 in enumerate(measure_groups):
                        if i < j:
                            measure1: Optional[Measurement] = result.get_group_stats(
                                cohort1
                            )
                            measure2: Optional[Measurement] = result.get_group_stats(
                                cohort2
                            )

                            denom1 = literal_eval(
                                cohort1.extract_denom_value_by_id(result.denoms)
                            )
                            denom2 = literal_eval(
                                cohort2.extract_denom_value_by_id(result.denoms)
                            )

                            if (
                                measure1 is not None
                                and measure2 is not None
                                and denom1 > 0
                                and denom2 > 0
                            ):
                                effect1: float = literal_eval(measure1.value)
                                effect2: float = literal_eval(measure2.value)
                            else:
                                continue

                            # divide by cohort size or 100 if result is a percentage
                            unit = (
                                result.unitOfMeasure.lower()
                                if result.unitOfMeasure
                                else ""
                            )
                            effect1 = (
                                effect1 / 100 if "percent" in unit else effect1 / denom1
                            )
                            effect2 = (
                                effect2 / 100 if "percent" in unit else effect2 / denom2
                            )
                            effect_size = effect2 - effect1

                            self.outcome_treatment.append(
                                (result.title, (cohort1.title, cohort2.title))
                            )
                            # always cohort2 - cohort1
                            self.effect_sizes.append(effect_size)

            self.outcome_names = [out.title for out in outcome_measures]
            self.treatment_names = [arm.title for arm in measure_groups]

    def hard_filter_ty(self, extractions):
        for name in self.treatment_names + self.outcome_names:
            extractions = extractions[extractions[name].isin(self.options[name])]
        return extractions

    def hard_filter_inclusion(self, extractions):
        for name in self.extended_covariate_names:
            extractions = extractions[
                extractions[name].lower().isin(["yes", "unknown"])
            ]
            extractions = extractions[
                extractions[name].lower().isin(["yes", "unknown"])
            ]
        return extractions

    def discretize(self, extractions):
        return

    def set_transforms(self):
        self.numerical_repr = {}
        self.language_repr = {}
