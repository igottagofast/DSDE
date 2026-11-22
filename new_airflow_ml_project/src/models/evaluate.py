"""Model evaluation, feature importance, threshold tuning, and yearly reporting."""
import json
from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.settings import EVALUATION_DIR, THRESHOLD_GRID


def extract_feature_importance(model_obj, feature_names: List[str]) -> pd.DataFrame:
    """Extract sorted feature importances or absolute linear coefficients."""
    if hasattr(model_obj, "feature_importances_"):
        importances = model_obj.feature_importances_
    elif hasattr(model_obj, "named_steps") and hasattr(
        model_obj.named_steps.get("lr", None), "coef_"
    ):
        importances = np.abs(model_obj.named_steps["lr"].coef_[0])
    elif hasattr(model_obj, "coef_"):
        importances = np.abs(model_obj.coef_[0])
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    return (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_threshold_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = None,
) -> pd.DataFrame:
    """Evaluate classification metrics across probability decision thresholds."""
    grid = thresholds if thresholds is not None else THRESHOLD_GRID
    rows = []
    for th in grid:
        y_pred = (y_prob >= th).astype(int)
        rows.append(
            {
                "Threshold": float(th),
                "Positive_Rate": float(np.mean(y_pred)),
                "Accuracy": float(accuracy_score(y_true, y_pred)),
                "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def analyze_yearly_performance(
    test_dates: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Break down evaluation metrics across calendar years in the test split."""
    years = pd.to_datetime(test_dates).dt.year
    records = []
    for yr in sorted(years.unique()):
        mask = (years == yr).to_numpy()
        y_t = y_true[mask]
        y_p = y_prob[mask]
        y_pred = (y_p >= threshold).astype(int)
        auc = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else np.nan
        records.append(
            {
                "Year": int(yr),
                "Samples": int(np.sum(mask)),
                "Actual_Positive_Rate": float(np.mean(y_t)),
                "Pred_Positive_Rate": float(np.mean(y_pred)),
                "ROC-AUC": auc,
                "Accuracy": float(accuracy_score(y_t, y_pred)),
                "Precision": float(precision_score(y_t, y_pred, zero_division=0)),
                "Recall": float(recall_score(y_t, y_pred, zero_division=0)),
                "F1-Score": float(f1_score(y_t, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(records)


def _predict_probabilities(model_obj, X_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model_obj, "predict_proba"):
        return model_obj.predict_proba(X_test)[:, 1]
    return model_obj.predict(X_test).astype(float)


def evaluate_saved_models(
    trained_results: Iterable[dict],
    split_info: dict,
    run_date: str,
) -> dict:
    """Evaluate all serialized models against test split and select best per target."""
    test_path = Path(split_info["test_path"])
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_path}")

    test_df = pd.read_parquet(test_path)
    feature_cols = split_info["feature_cols"]
    target_cols = split_info["target_cols"]
    X_test = test_df[feature_cols]

    trained_list = list(trained_results)
    if not trained_list:
        raise ValueError("No trained model results provided for evaluation.")

    eval_dir = EVALUATION_DIR / run_date
    eval_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    importance_frames = []
    best_models: Dict[str, dict] = {}

    for target_col in target_cols:
        y_true = test_df[target_col].to_numpy()
        target_rows = []

        for item in trained_list:
            model_name = item["model_name"]
            artifact_path = item["artifacts"][target_col]
            clf = joblib.load(artifact_path)

            y_prob = _predict_probabilities(clf, X_test)
            y_pred = clf.predict(X_test)

            auc = (
                float(roc_auc_score(y_true, y_prob))
                if len(np.unique(y_true)) > 1
                else np.nan
            )
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            row = {
                "Target": target_col,
                "Model": model_name,
                "ROC-AUC": auc,
                "Accuracy": float(accuracy_score(y_true, y_pred)),
                "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "Artifact_Path": artifact_path,
            }
            target_rows.append(row)
            comparison_rows.append(row)

        df_target = pd.DataFrame(target_rows).sort_values(
            by=["ROC-AUC", "Accuracy"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

        best_row = df_target.iloc[0]
        best_model_path = best_row["Artifact_Path"]
        best_models[target_col] = {
            "model_name": str(best_row["Model"]),
            "roc_auc": None if pd.isna(best_row["ROC-AUC"]) else float(best_row["ROC-AUC"]),
            "accuracy": float(best_row["Accuracy"]),
            "f1_score": float(best_row["F1-Score"]),
            "artifact_path": best_model_path,
        }

        best_clf = joblib.load(best_model_path)
        df_imp = extract_feature_importance(best_clf, feature_cols)
        if not df_imp.empty:
            df_imp.insert(0, "Target", target_col)
            df_imp.insert(1, "Best_Model", str(best_row["Model"]))
            df_imp["Rank"] = range(1, len(df_imp) + 1)
            importance_frames.append(df_imp)

    df_comparison = (
        pd.DataFrame(comparison_rows)
        .drop(columns=["Artifact_Path"])
        .sort_values(by=["Target", "ROC-AUC", "Accuracy"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    comparison_path = eval_dir / "model_comparison.csv"
    df_comparison.to_csv(comparison_path, index=False)

    df_importances = (
        pd.concat(importance_frames, ignore_index=True)
        if importance_frames
        else pd.DataFrame(columns=["Target", "Best_Model", "Feature", "Importance", "Rank"])
    )
    feature_importance_path = eval_dir / "feature_importance.csv"
    df_importances.to_csv(feature_importance_path, index=False)

    summary_payload = {
        "run_date": run_date,
        "test_rows": int(len(test_df)),
        "test_date_range": split_info.get("test_date_range"),
        "best_models": best_models,
    }
    summary_path = eval_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "run_date": run_date,
        "eval_dir": str(eval_dir),
        "comparison_path": str(comparison_path),
        "feature_importance_path": str(feature_importance_path),
        "summary_path": str(summary_path),
        "best_models": best_models,
    }


def run_post_eval_analysis(eval_summary: dict, split_info: dict, run_date: str) -> dict:
    """Run threshold sensitivity and yearly robustness breakdown on best models."""
    test_df = pd.read_parquet(split_info["test_path"])
    feature_cols = split_info["feature_cols"]
    X_test = test_df[feature_cols]
    test_dates = test_df["Date"]
    eval_dir = Path(eval_summary["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    threshold_frames = []
    yearly_frames = []
    report_lines = [
        f"# Model Evaluation Report ({run_date})",
        "",
        f"- **Train Split**: {split_info.get('train_rows')} rows ({split_info.get('train_date_range')})",
        f"- **Test Split**: {split_info.get('test_rows')} rows ({split_info.get('test_date_range')})",
        f"- **Features**: {len(feature_cols)} columns",
        "",
        "## Best Model per Target",
        "",
        "| Target | Best Model | ROC-AUC | Accuracy | F1-Score |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for target_col, meta in eval_summary["best_models"].items():
        clf = joblib.load(meta["artifact_path"])
        y_true = test_df[target_col].to_numpy()
        y_prob = _predict_probabilities(clf, X_test)

        df_th = evaluate_threshold_grid(y_true, y_prob, thresholds=THRESHOLD_GRID)
        df_th.insert(0, "Target", target_col)
        df_th.insert(1, "Best_Model", meta["model_name"])
        threshold_frames.append(df_th)

        df_yr = analyze_yearly_performance(test_dates, y_true, y_prob, threshold=0.50)
        df_yr.insert(0, "Target", target_col)
        df_yr.insert(1, "Best_Model", meta["model_name"])
        yearly_frames.append(df_yr)

        auc_str = f"{meta['roc_auc']:.4f}" if meta["roc_auc"] is not None else "N/A"
        report_lines.append(
            f"| {target_col} | {meta['model_name']} | {auc_str} | "
            f"{meta['accuracy']:.4f} | {meta['f1_score']:.4f} |"
        )

    df_thresholds = pd.concat(threshold_frames, ignore_index=True)
    threshold_path = eval_dir / "threshold_tuning.csv"
    df_thresholds.to_csv(threshold_path, index=False)

    df_yearly = pd.concat(yearly_frames, ignore_index=True)
    yearly_path = eval_dir / "yearly_performance.csv"
    df_yearly.to_csv(yearly_path, index=False)

    report_lines.extend(
        [
            "",
            "## Generated Artifacts",
            "",
            f"- `model_comparison.csv`: `{eval_summary['comparison_path']}`",
            f"- `feature_importance.csv`: `{eval_summary['feature_importance_path']}`",
            f"- `threshold_tuning.csv`: `{threshold_path}`",
            f"- `yearly_performance.csv`: `{yearly_path}`",
        ]
    )
    report_path = eval_dir / "evaluation_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "run_date": run_date,
        "threshold_path": str(threshold_path),
        "yearly_path": str(yearly_path),
        "report_path": str(report_path),
    }
