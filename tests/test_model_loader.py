"""Tests de carga del modelo serializado y su metadata de trazabilidad."""
import pytest

from app.model_loader import load_model_artifact
from src.inference.predict import load_model


def test_model_artifact_loads_correctly(loaded_artifact):
    """El artefacto debe existir, deserializarse y exponer predict y predict_proba."""
    assert loaded_artifact is not None
    assert hasattr(loaded_artifact.pipeline, "predict")
    assert hasattr(loaded_artifact.pipeline, "predict_proba")
    assert loaded_artifact.model_version == "1"
    assert loaded_artifact.model_name == "customer-churn-classifier"
    assert loaded_artifact.run_id != "unknown"


def test_load_model_fallback_default_path():
    """src.inference.predict.load_model() debe cargar el artefacto por defecto."""
    model = load_model()
    assert hasattr(model, "predict")
    assert "Yes" in list(model.classes_)
    assert "No" in list(model.classes_)


def test_load_model_missing_file_raises(tmp_path):
    """Si la ruta del modelo no existe, debe lanzar FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model_artifact(model_path=tmp_path / "non_existent.joblib")
