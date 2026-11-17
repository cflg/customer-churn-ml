"""Responsabilidad de localizar y cargar el artefacto del modelo y su metadata."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import joblib

from src.config import ROOT_DIR
from src.inference.predict import load_model

DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "churn_pipeline.joblib"
DEFAULT_METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"


@dataclass(frozen=True)
class ModelArtifact:
    """Contenedor inmutable del pipeline cargado y su trazabilidad."""

    pipeline: Any
    model_name: str
    model_version: str
    run_id: str
    source: str


def load_model_artifact(
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    model_uri: str | None = None,
    run_id: str | None = None,
) -> ModelArtifact:
    """
    Carga el pipeline de inferencia y su metadata de trazabilidad.

    Prioridad de resolución:
    1. Argumentos explícitos (`model_uri`, `run_id`, `model_path`).
    2. Variables de entorno (`MODEL_URI`, `MLFLOW_RUN_ID`, `MODEL_PATH`).
    3. Artefacto empaquetado por defecto en `models/churn_pipeline.joblib`.
    """
    resolved_uri = model_uri or os.getenv("MODEL_URI")
    resolved_run_id = run_id or os.getenv("MLFLOW_RUN_ID")
    resolved_model_path = Path(
        model_path or os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    )
    resolved_meta_path = Path(
        metadata_path or os.getenv("MODEL_METADATA_PATH", str(DEFAULT_METADATA_PATH))
    )

    metadata: dict[str, Any] = {}
    if resolved_meta_path.exists():
        with open(resolved_meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

    model_name = os.getenv(
        "MODEL_NAME", metadata.get("model_name", "customer-churn-classifier")
    )
    model_version = os.getenv(
        "MODEL_VERSION", str(metadata.get("model_version", "1"))
    )
    meta_run_id = metadata.get("run_id", "unknown")

    if resolved_uri or resolved_run_id:
        pipeline = load_model(run_id=resolved_run_id, model_uri=resolved_uri)
        source = resolved_uri or f"runs:/{resolved_run_id}/model"
        return ModelArtifact(
            pipeline=pipeline,
            model_name=model_name,
            model_version=model_version,
            run_id=resolved_run_id or meta_run_id,
            source=source,
        )

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el artefacto del modelo en {resolved_model_path}"
        )

    pipeline = joblib.load(resolved_model_path)
    return ModelArtifact(
        pipeline=pipeline,
        model_name=model_name,
        model_version=model_version,
        run_id=meta_run_id,
        source=str(resolved_model_path),
    )
