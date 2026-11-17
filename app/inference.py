"""Lógica de inferencia desacoplada de la capa HTTP."""
from typing import Any
import uuid

import pandas as pd

from app.schemas import PredictionRequest
from src.config import FEATURE_COLUMNS
from src.inference.predict import predict_df


def to_dataframe(request: PredictionRequest | dict[str, Any]) -> pd.DataFrame:
    """Transforma un PredictionRequest o diccionario validado en un DataFrame de 1 fila."""
    payload = (
        request.model_dump()
        if isinstance(request, PredictionRequest)
        else dict(request)
    )
    df = pd.DataFrame([payload])
    return df[FEATURE_COLUMNS]


def predict_single(
    pipeline: Any,
    request: PredictionRequest | dict[str, Any],
    model_version: str = "1",
    request_id: str | None = None,
) -> dict[str, Any]:
    """Ejecuta el pipeline completo para un cliente y devuelve el contrato de salida."""
    df_input = to_dataframe(request)
    result_df = predict_df(pipeline, df_input)
    row = result_df.iloc[0]

    return {
        "prediction": str(row["prediction"]),
        "probability": round(float(row["probability"]), 4),
        "risk_level": str(row["risk_level"]),
        "model_version": str(model_version),
        "request_id": request_id or str(uuid.uuid4()),
    }
