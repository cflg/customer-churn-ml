"""Tests de inferencia local y reglas de negocio de nivel de riesgo."""
import pytest

from app.inference import predict_single, to_dataframe
from app.schemas import PredictionRequest
from src.config import FEATURE_COLUMNS
from src.inference.predict import predict_df, risk_level


@pytest.mark.parametrize(
    "probability,expected_level",
    [
        (0.0, "LOW"),
        (0.15, "LOW"),
        (0.299, "LOW"),
        (0.30, "MEDIUM"),
        (0.45, "MEDIUM"),
        (0.599, "MEDIUM"),
        (0.60, "HIGH"),
        (0.85, "HIGH"),
        (1.0, "HIGH"),
    ],
)
def test_risk_level_thresholds(probability, expected_level):
    """Verifica la asignación de nivel de riesgo según los umbrales definidos."""
    assert risk_level(probability) == expected_level


def test_to_dataframe_preserves_feature_columns(valid_customer_payload):
    """to_dataframe debe devolver exactamente las 19 columnas esperadas por el pipeline."""
    req = PredictionRequest(**valid_customer_payload)
    df = to_dataframe(req)
    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) == 1


def test_predict_single_contract(loaded_artifact, valid_customer_payload):
    """predict_single debe devolver predicción, probabilidad en [0, 1], riesgo y versión."""
    result = predict_single(
        pipeline=loaded_artifact.pipeline,
        request=valid_customer_payload,
        model_version=loaded_artifact.model_version,
        request_id="test-req-123",
    )
    assert set(result.keys()) == {
        "prediction",
        "probability",
        "risk_level",
        "model_version",
        "request_id",
    }
    assert result["prediction"] in {"Yes", "No"}
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["model_version"] == "1"
    assert result["request_id"] == "test-req-123"


def test_predict_df_batch(loaded_artifact, sample_customer_df):
    """predict_df debe agregar las columnas de salida manteniendo el largo del lote."""
    out = predict_df(loaded_artifact.pipeline, sample_customer_df)
    assert len(out) == 1
    assert {"prediction", "probability", "risk_level"}.issubset(out.columns)
