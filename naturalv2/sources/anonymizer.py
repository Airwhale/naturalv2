import logging
from collections import Counter
from typing import Optional

import pandas as pd
from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine
from presidio_anonymizer import AnonymizerEngine, OperatorConfig
from presidio_structured import (
    PandasAnalysisBuilder,
    PandasDataProcessor,
    StructuredEngine,
)
from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)


def _get_sample_size(
    pop_size: int, z: float = 1.96, p: float = 0.5, e: float = 0.01
) -> int:
    """
    Calculate the sample size needed for a given population size, confidence level,
    proportion of success, and margin of error.
    """
    return int(
        (pop_size * z**2 * p * (1 - p)) / ((pop_size - 1) * e**2 + z**2 * p * (1 - p))
    )


class Anonymizer:
    """
    A class for anonymizing sensitive information in text data.

    This class uses Presidio's AnalyzerEngine and AnonymizerEngine to detect and anonymize
    sensitive entities such as credit card numbers, email addresses, and personal identifiers.
    """

    ENTITIES = [
        "CREDIT_CARD",
        "CRYPTO",
        "EMAIL_ADDRESS",
        "IBAN_CODE",
        "IP_ADDRESS",
        "PERSON",
        "PHONE_NUMBER",
        "MEDICAL_LICENSE",
        "US_BANK_NUMBER",
        "US_DRIVER_LICENSE",
        "US_ITIN",
        "US_PASSPORT",
        "US_SSN",
        "UK_NHS",
        "UK_NINO",
        "ES_NIF",
        "ES_NIE",
        "IT_FISCAL_CODE",
        "IT_DRIVER_LICENSE",
        "IT_VAT_CODE",
        "IT_PASSPORT",
        "IT_IDENTITY_CARD",
        "PL_PESEL",
        "SG_NRIC_FIN",
        "SG_UEN",
        "AU_ABN",
        "AU_ACN",
        "AU_TFN",
        "AU_MEDICARE",
        "IN_PAN",
        "IN_AADHAAR",
        "IN_VEHICLE_REGISTRATION",
        "IN_VOTER",
        "IN_PASSPORT",
        "FI_PERSONAL_IDENTITY_CODE",
    ]

    def __init__(self, score_threshold: float = 0.7) -> None:
        self._score_threshold = score_threshold
        self._analyzer = AnalyzerEngine(
            default_score_threshold=score_threshold, supported_languages=["en"]
        )
        self._anonymizer = AnonymizerEngine()

        self.operators = {}
        for entity in self.ENTITIES:
            self.operators[entity] = OperatorConfig(
                operator_name="replace",
                params={"new_value": f"<{entity}>"},
            )

    def anonymize_text(self, text: str) -> tuple[str, dict[str, int]]:
        results = self._analyzer.analyze(
            text=text, language="en", entities=self.ENTITIES
        )
        anon_result = self._anonymizer.anonymize(text=text, analyzer_results=results)

        entity_stats = Counter()

        for item in anon_result.items:
            entity = item.entity_type
            entity_stats[entity] += 1

        return anon_result.text, dict(entity_stats)

    def anonymize_dataframe(
        self,
        df: pd.DataFrame,
        exclude_cols: list[str] = None,
        unstructured_text_cols: list[str] = None,
        data_source_name: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 1,
    ) -> pd.DataFrame:
        exclude_cols, unstructured_text_cols = self._validate_anonymize_dataframe_args(
            exclude_cols, unstructured_text_cols, batch_size, num_workers
        )

        # the structured engine will only handle (int, float, bool, str) dtypes,
        # but we only care about numeric and string dtypes for structured analysis,
        # so we exclude any other dtypes from structured analysis
        for col in df.columns:
            if col not in exclude_cols and (
                not pd.api.types.is_numeric_dtype(df[col])
                and not pd.api.types.is_string_dtype(df[col])
            ):
                exclude_cols.append(col)

        df_copy = df.copy()
        df_entity_stats = Counter()
        df_for_structured = df.drop(
            columns=exclude_cols + unstructured_text_cols, errors="ignore"
        )
        # fill NaN values based on the dtype of the column
        for col in df_for_structured.columns:
            if pd.api.types.is_numeric_dtype(df_for_structured[col]):
                df_for_structured[col].fillna(0, inplace=True)
            elif pd.api.types.is_string_dtype(df_for_structured[col]):
                df_for_structured[col].fillna("", inplace=True)

        tqdm.write(
            f"Anonymizing {data_source_name if data_source_name else 'DataFrame'}"
        )
        structured_analysis = PandasAnalysisBuilder(
            analyzer=self._analyzer,
            n_process=num_workers,
            batch_size=batch_size,
        ).generate_analysis(
            df_for_structured,
            selection_strategy="mixed",
            mixed_strategy_threshold=0.5,
            n=_get_sample_size(len(df_for_structured)),
        )

        # convert columns that appear in structured_analysis to string type in df_copy
        for col in structured_analysis.entity_mapping:
            df_copy[col] = df_copy[col].astype(str)

        pandas_engine = StructuredEngine(data_processor=PandasDataProcessor())
        anonymized_df = pandas_engine.anonymize(
            df_copy,
            structured_analysis=structured_analysis,
            operators=self.operators,
        )

        for col, entity_type in structured_analysis.entity_mapping.items():
            non_nan_count = df_copy[col].notna().sum()  # before filling NaN values
            if non_nan_count > 0:
                df_entity_stats[entity_type] += non_nan_count

        tqdm.write(
            f"[{data_source_name if data_source_name else 'DataFrame'}] "
            f"Anonymizing unstructured text columns: {', '.join(unstructured_text_cols)}"
        )
        batch_analyzer = BatchAnalyzerEngine(self._analyzer)
        for col in unstructured_text_cols:
            if col in df_copy.columns and (pd.api.types.is_string_dtype(df_copy[col])):
                anonymized_col, col_stats = self._anonymize_column(
                    df_copy.loc[:, col],
                    batch_analyzer,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                anonymized_df.loc[:, col] = anonymized_col
                df_entity_stats.update(col_stats)
            else:
                tqdm.write(
                    f"Column `{col}` does not exist in the DataFrame or is not a "
                    "string type. Skipping anonymization for this column."
                )

        logging.info(
            f"Anonymization stats for {data_source_name if data_source_name else 'DataFrame'}:"
        )
        self._log_anonymization_stats(df_entity_stats)

        return anonymized_df

    def _anonymize_column(
        self,
        col: pd.Series,
        batch_analyzer: BatchAnalyzerEngine,
        batch_size: int = 1,
        num_workers: int = 1,
    ) -> tuple[pd.Series, dict[str, int]]:
        col_stats_agg = Counter()

        col_data = col.fillna("").astype(str).to_list()
        if not col_data:
            tqdm.write(
                f"Column {col.name} is empty or contains only NaN values. "
                "Skipping anonymization and returning original column."
            )
            return col, dict(col_stats_agg)

        try:
            analyzer_results = batch_analyzer.analyze_iterator(
                col_data,
                language="en",
                batch_size=batch_size,
                n_process=num_workers,
                entities=self.ENTITIES,
            )
        except Exception as e:
            tqdm.write(
                f"Error analyzing column {col.name}: {e}. "
                "Skipping anonymization and returning original column."
            )
            return col, dict(col_stats_agg)

        processed_text = []
        for original_text, results in zip(col_data, analyzer_results):
            if results:
                try:
                    anonymizer_engine_result = self._anonymizer.anonymize(
                        text=original_text, analyzer_results=results
                    )

                    processed_text.append(anonymizer_engine_result.text)

                    for item in anonymizer_engine_result.items:
                        col_stats_agg[item.entity_type] += 1
                except Exception as e:
                    tqdm.write(
                        f"Error anonymizing text in column {col.name}: {e}. "
                        "Returning original text."
                    )
                    processed_text.append(original_text)
                    continue
            else:
                processed_text.append(original_text)

        return pd.Series(
            processed_text, index=col.index, dtype=col.dtype, name=col.name
        ), dict(col_stats_agg)

    def _validate_anonymize_dataframe_args(
        self,
        exclude_cols: Optional[list[str]],
        unstructured_text_cols: Optional[list[str]],
        batch_size: int,
        num_workers: int,
    ):
        if num_workers <= 0:
            raise ValueError(
                "Expected ``num_workers`` to be a positive integer greater than 0 "
                f"but got {num_workers}"
            )
        if batch_size <= 0:
            raise ValueError(
                "Expected ``batch_size`` to be a positive integer greater than 0 "
                f"but got {batch_size}"
            )

        if exclude_cols is None:
            exclude_cols = []
        if unstructured_text_cols is None:
            unstructured_text_cols = []

        if not isinstance(exclude_cols, list) or not all(
            isinstance(col, str) for col in exclude_cols
        ):
            raise ValueError(
                "Expected `exclude_cols` to be a list of strings but got "
                f"{type(exclude_cols)}"
            )
        if not isinstance(unstructured_text_cols, list) or not all(
            isinstance(col, str) for col in unstructured_text_cols
        ):
            raise ValueError(
                "Expected `unstructured_text_cols` to be a list of strings but got "
                f"{type(unstructured_text_cols)}"
            )

        overlap = set(exclude_cols) & set(unstructured_text_cols)
        if overlap:
            raise ValueError(
                "Expected `exclude_cols` and `unstructured_text_cols` to be disjoint "
                f"sets but found overlap: {overlap}"
            )
        return exclude_cols, unstructured_text_cols

    def _log_anonymization_stats(self, df_entity_stats: Counter) -> None:
        if df_entity_stats:
            stats_table = pd.DataFrame(
                list(df_entity_stats.items()), columns=["Entity", "Count"]
            ).sort_values(by="Count", ascending=False)
        else:
            stats_table = pd.DataFrame(columns=["Entity", "Count"])

        stats_table.loc[len(stats_table)] = ["-" * 5, "-" * 5]
        stats_table.loc[len(stats_table)] = ["Total", sum(df_entity_stats.values())]
        stats_table.loc[len(stats_table)] = ["Score Threshold", self._score_threshold]

        logging.info(stats_table.to_string(index=False))
