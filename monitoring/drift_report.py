"""
Generación de reporte de Data Drift y Prediction Drift comparando el dataset
histórico de referencia contra el lote de producción (`customer_churn_current.csv`).

Uso:
    python -m monitoring.drift_report
"""
import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.model_loader import load_model_artifact
from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    ROOT_DIR,
)
from src.inference.predict import predict_df

DEFAULT_CURRENT_PATH = ROOT_DIR / "data" / "production" / "customer_churn_current.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "monitoring" / "reports"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona FEATURE_COLUMNS y convierte TotalCharges a numérico."""
    clean = df[FEATURE_COLUMNS].copy()
    clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce")
    return clean


def append_model_predictions(df: pd.DataFrame, pipeline: Any) -> pd.DataFrame:
    """Ejecuta el pipeline sobre el lote y agrega probabilidad y clase predicha."""
    scored = predict_df(pipeline, df)
    out = df.copy()
    out["predicted_churn"] = scored["prediction"]
    out["predicted_probability"] = scored["probability"]
    out["risk_level"] = scored["risk_level"]
    return out


def compute_feature_shift_summary(
    ref_df: pd.DataFrame, cur_df: pd.DataFrame
) -> dict[str, Any]:
    """Calcula estadísticas comparativas por variable y sobre las predicciones."""
    numeric_summary = {}
    for col in NUMERIC_FEATURES + ["predicted_probability"]:
        if col in ref_df.columns and col in cur_df.columns:
            ref_mean = float(ref_df[col].mean())
            cur_mean = float(cur_df[col].mean())
            numeric_summary[col] = {
                "reference_mean": round(ref_mean, 4),
                "current_mean": round(cur_mean, 4),
                "abs_diff": round(cur_mean - ref_mean, 4),
                "rel_diff_pct": round(
                    ((cur_mean - ref_mean) / ref_mean * 100.0) if ref_mean else 0.0,
                    2,
                ),
            }

    categorical_summary = {}
    for col in CATEGORICAL_FEATURES + ["predicted_churn", "risk_level"]:
        if col in ref_df.columns and col in cur_df.columns:
            ref_dist = (
                ref_df[col].value_counts(normalize=True).round(4).to_dict()
            )
            cur_dist = (
                cur_df[col].value_counts(normalize=True).round(4).to_dict()
            )
            all_cats = sorted(set(ref_dist.keys()) | set(cur_dist.keys()))
            max_shift = max(
                abs(cur_dist.get(k, 0.0) - ref_dist.get(k, 0.0)) for k in all_cats
            )
            categorical_summary[col] = {
                "reference_distribution": ref_dist,
                "current_distribution": cur_dist,
                "max_category_shift_pct": round(float(max_shift * 100.0), 2),
            }

    return {
        "reference_rows": len(ref_df),
        "current_rows": len(cur_df),
        "numeric_shifts": numeric_summary,
        "categorical_shifts": categorical_summary,
    }


def generate_evidently_report(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Genera el reporte HTML y extrae las métricas de drift usando Evidently."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=ref_df[FEATURE_COLUMNS],
        current_data=cur_df[FEATURE_COLUMNS],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "data_drift_report.html"
    report.save_html(str(html_path))

    report_dict = report.as_dict()
    drifted_columns = []
    share_of_drifted_columns = 0.0
    dataset_drift = False

    for metric in report_dict.get("metrics", []):
        result = metric.get("result", {})
        if "drift_by_columns" in result:
            dataset_drift = bool(result.get("dataset_drift", False))
            share_of_drifted_columns = float(
                result.get("share_of_drifted_columns", 0.0)
            )
            for col_name, col_info in result["drift_by_columns"].items():
                if col_info.get("drift_detected"):
                    drifted_columns.append(
                        {
                            "column": col_name,
                            "stattest_name": col_info.get("stattest_name"),
                            "drift_score": round(
                                float(col_info.get("drift_score", 0.0)), 6
                            ),
                        }
                    )

    try:
        rel_html = str(html_path.relative_to(ROOT_DIR))
    except ValueError:
        rel_html = str(html_path)

    return {
        "html_report": rel_html,
        "dataset_drift": dataset_drift,
        "share_of_drifted_columns": round(share_of_drifted_columns, 4),
        "drifted_columns_count": len(drifted_columns),
        "drifted_columns": drifted_columns,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta el análisis de Data Drift con Evidently."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=RAW_DATA_PATH,
        help="Ruta al CSV histórico de referencia",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=DEFAULT_CURRENT_PATH,
        help="Ruta al CSV actual de producción",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio de salida para los reportes",
    )
    args = parser.parse_args()

    ref_raw = pd.read_csv(args.reference)
    cur_raw = pd.read_csv(args.current)

    ref_features = prepare_features(ref_raw)
    cur_features = prepare_features(cur_raw)

    artifact = load_model_artifact()
    ref_scored = append_model_predictions(ref_features, artifact.pipeline)
    cur_scored = append_model_predictions(cur_features, artifact.pipeline)

    evidently_summary = generate_evidently_report(
        ref_features, cur_features, args.output_dir
    )
    shift_summary = compute_feature_shift_summary(ref_scored, cur_scored)

    full_summary = {
        "model_name": artifact.model_name,
        "model_version": artifact.model_version,
        "run_id": artifact.run_id,
        "evidently": evidently_summary,
        "distributions": shift_summary,
    }

    summary_path = args.output_dir / "data_drift_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, ensure_ascii=False)

    print(f"Reporte HTML guardado en: {evidently_summary['html_report']}")
    print(f"Resumen JSON guardado en: {summary_path}")
    print(
        f"Dataset Drift detectado: {evidently_summary['dataset_drift']} "
        f"({evidently_summary['drifted_columns_count']}/{len(FEATURE_COLUMNS)} features)"
    )


if __name__ == "__main__":
    main()
