"""
Entrena un modelo (baseline / logreg / rf), lo evalúa contra el test set
y registra parámetros, métricas y el modelo en MLflow.

Ejemplos:
    python -m src.training.train --model baseline
    python -m src.training.train --model logreg --C 1.0
    python -m src.training.train --model rf --n-estimators 300 --max-depth 8 --register
"""
import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COL,
    RANDOM_STATE,
)
from src.evaluation.metrics import compute_metrics
from src.features.preprocessing import build_preprocessor

MODEL_REGISTRY_NAME = "customer-churn-classifier"
EXPERIMENT_NAME = "customer-churn"


def build_model(name: str, args):
    if name == "baseline":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        return LogisticRegression(
            C=args.C,
            max_iter=1000,
            class_weight="balanced" if args.balanced else None,
            random_state=RANDOM_STATE,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight="balanced" if args.balanced else None,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Modelo desconocido: {name}")


def get_params(name: str, args) -> dict:
    if name == "baseline":
        return {"strategy": "most_frequent"}
    if name == "logreg":
        return {"C": args.C, "class_weight": args.balanced}
    if name == "rf":
        return {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "class_weight": args.balanced,
        }
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "logreg", "rf"], required=True)
    parser.add_argument("--C", type=float, default=1.0, help="Regularización para logreg")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--balanced", action="store_true", help="class_weight=balanced")
    parser.add_argument("--register", action="store_true", help="Registrar en Model Registry")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COL]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", build_model(args.model, args)),
        ]
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=args.run_name or args.model):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index("Yes")]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        params = get_params(args.model, args)

        mlflow.log_param("model_type", args.model)
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        registered_name = MODEL_REGISTRY_NAME if args.register else None
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=registered_name,
        )

        run_id = mlflow.active_run().info.run_id
        print(f"run_id={run_id} model={args.model} params={params}")
        print("metrics:", metrics)


if __name__ == "__main__":
    main()
