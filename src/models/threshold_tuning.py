"""
Threshold Tuning and Yearly Breakdown Analysis
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_threshold_grid(y_true: np.ndarray, y_prob: np.ndarray, thresholds: list = None) -> pd.DataFrame:
    """
    ทดสอบประสิทธิภาพที่ระดับ Threshold ต่างๆ (Sensitivity / Specificity Analysis)
    """
    if thresholds is None:
        thresholds = [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60]

    rows = []
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        rows.append({
            'Threshold': th,
            'Positive_Rate': np.mean(y_pred),
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0)
        })

    return pd.DataFrame(rows)


def analyze_yearly_performance(
    test_dates: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> pd.DataFrame:
    """
    วิเคราะห์ประสิทธิภาพโมเดลแยกตามรายปี (Yearly Robustness Breakdown)
    """
    years = pd.to_datetime(test_dates).dt.year
    unique_years = sorted(years.unique())

    records = []
    for yr in unique_years:
        mask = (years == yr)
        y_t = y_true[mask]
        y_p = y_prob[mask]
        y_pred = (y_p >= threshold).astype(int)

        auc = roc_auc_score(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan
        records.append({
            'Year': yr,
            'Samples': int(np.sum(mask)),
            'Actual_Positive_Rate': np.mean(y_t),
            'Pred_Positive_Rate': np.mean(y_pred),
            'ROC-AUC': auc,
            'Accuracy': accuracy_score(y_t, y_pred),
            'Precision': precision_score(y_t, y_pred, zero_division=0),
            'Recall': recall_score(y_t, y_pred, zero_division=0),
            'F1-Score': f1_score(y_t, y_pred, zero_division=0)
        })

    return pd.DataFrame(records)
