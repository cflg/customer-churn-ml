"""Tests unitarios sobre el pipeline de preprocesamiento."""
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS
from src.features.preprocessing import build_preprocessor


def test_preprocessor_fit_transform_and_missing_values(valid_customer_payload):
    """El preprocesador debe imputar faltantes en TotalCharges y transformar sin error."""
    row_complete = dict(valid_customer_payload)
    row_missing = dict(valid_customer_payload, tenure=0, TotalCharges=np.nan)
    row_alt = dict(
        valid_customer_payload,
        gender="Female",
        Contract="Two year",
        InternetService="DSL",
        MonthlyCharges=45.0,
        TotalCharges=1200.0,
    )

    df = pd.DataFrame([row_complete, row_missing, row_alt])[FEATURE_COLUMNS]
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 3
    assert not np.isnan(transformed.toarray() if hasattr(transformed, "toarray") else transformed).any()


def test_preprocessor_handles_unknown_categories(valid_customer_payload):
    """OneHotEncoder con handle_unknown='ignore' no debe fallar ante categorías nuevas."""
    df_train = pd.DataFrame([valid_customer_payload])[FEATURE_COLUMNS]
    preprocessor = build_preprocessor()
    preprocessor.fit(df_train)

    unseen_row = dict(valid_customer_payload, Contract="Unseen-Contract-Type")
    df_unseen = pd.DataFrame([unseen_row])[FEATURE_COLUMNS]
    transformed = preprocessor.transform(df_unseen)
    assert transformed.shape[0] == 1
