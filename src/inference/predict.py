"""
Carga un modelo desde MLflow (por run_id o por Model Registry) y predice
sobre un DataFrame con las columnas de FEATURE_COLUMNS.

Uso:
    python -m src.inference.predict --run-id <RUN_ID> --input data/raw/customer_churn_historical.csv
"""
import argparse

import mlflow
import pandas as pd

from src.config import FEATURE_COLUMNS

RISK_THRESHOLDS = {"LOW": 0.3, "MEDIUM": 0.6}  # >=0.6 => HIGH


def risk_level(probability: float) -> str:
    if probability < RISK_THRESHOLDS["LOW"]:
        return "LOW"
    if probability < RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "HIGH"


def load_model(run_id: str = None, model_uri: str = None):
    if model_uri:
        return mlflow.sklearn.load_model(model_uri)
    if run_id:
        return mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    raise ValueError("Se necesita --run-id o --model-uri")


def predict_df(model, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS]
    proba = model.predict_proba(X)[:, list(model.classes_).index("Yes")]
    pred = model.predict(X)
    out = df.copy()
    out["prediction"] = pred
    out["probability"] = proba
    out["risk_level"] = [risk_level(p) for p in proba]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--model-uri")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    model = load_model(run_id=args.run_id, model_uri=args.model_uri)
    df = pd.read_csv(args.input)
    result = predict_df(model, df)
    result.to_csv(args.output, index=False)
    print(f"Predicciones guardadas en {args.output}")


if __name__ == "__main__":
    main()
