"""Tests unitarios sobre las funciones de preparación y análisis de monitoreo/drift."""
import pandas as pd

from monitoring.drift_report import (
    append_model_predictions,
    compute_feature_shift_summary,
    prepare_features,
)
from src.config import FEATURE_COLUMNS


def test_prepare_features_cleans_total_charges(valid_customer_payload):
    """prepare_features debe seleccionar FEATURE_COLUMNS y convertir espacios vacíos a NaN."""
    row_empty = dict(valid_customer_payload, customerID="CUST-1", TotalCharges=" ")
    df = pd.DataFrame([row_empty])
    cleaned = prepare_features(df)
    assert list(cleaned.columns) == FEATURE_COLUMNS
    assert pd.isna(cleaned.loc[0, "TotalCharges"])


def test_compute_feature_shift_summary(loaded_artifact, valid_customer_payload):
    """El resumen estadístico debe calcular diferencias numéricas y categóricas."""
    ref_row = dict(valid_customer_payload, tenure=36, MonthlyCharges=65.0)
    cur_row = dict(valid_customer_payload, tenure=12, MonthlyCharges=95.0)

    ref_df = append_model_predictions(
        prepare_features(pd.DataFrame([ref_row])), loaded_artifact.pipeline
    )
    cur_df = append_model_predictions(
        prepare_features(pd.DataFrame([cur_row])), loaded_artifact.pipeline
    )

    summary = compute_feature_shift_summary(ref_df, cur_df)
    assert summary["reference_rows"] == 1
    assert summary["current_rows"] == 1
    assert summary["numeric_shifts"]["tenure"]["abs_diff"] == -24.0
    assert summary["numeric_shifts"]["MonthlyCharges"]["abs_diff"] == 30.0
