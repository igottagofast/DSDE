"""
Evaluation Metrics, Diagnostics, and Model Benchmarking
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from src.models.train import get_models_dict


def evaluate_models_on_targets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_test: pd.DataFrame,
    random_state: int = 42
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Tuple[str, object]]]:
    """
    เทรนและประเมินผลโมเดลทั้งหมดกับทุกตัวแปรเป้าหมาย พร้อมคืนค่าตารางคะแนนและโมเดลที่ดีที่สุด
    """
    results_per_target = {}
    best_models_per_target = {}

    for target_col in Y_train.columns:
        models = get_models_dict(random_state=random_state)
        records = []
        best_score = -1
        best_name = None
        best_model_obj = None

        y_tr = Y_train[target_col]
        y_te = Y_test[target_col]

        for model_name, clf in models.items():
            clf.fit(X_train, y_tr)

            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_te, y_prob)
            else:
                auc = np.nan

            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_te, y_pred)
            prec = precision_score(y_te, y_pred, zero_division=0)
            rec = recall_score(y_te, y_pred, zero_division=0)
            f1 = f1_score(y_te, y_pred, zero_division=0)

            records.append({
                'Model': model_name,
                'ROC-AUC': auc,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1
            })

            if not np.isnan(auc) and auc > best_score:
                best_score = auc
                best_name = model_name
                best_model_obj = clf

        df_metrics = pd.DataFrame(records).sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)
        results_per_target[target_col] = df_metrics
        best_models_per_target[target_col] = (best_name, best_model_obj)

    return results_per_target, best_models_per_target


def get_feature_importance(best_model_obj, feature_names: list) -> pd.DataFrame:
    """
    ดึงค่า Feature Importance หรือ Logistic Regression Coefficients
    """
    if hasattr(best_model_obj, 'feature_importances_'):
        importances = best_model_obj.feature_importances_
    elif hasattr(best_model_obj, 'named_steps') and hasattr(best_model_obj.named_steps.get('lr', None), 'coef_'):
        importances = np.abs(best_model_obj.named_steps['lr'].coef_[0])
    elif hasattr(best_model_obj, 'coef_'):
        importances = np.abs(best_model_obj.coef_[0])
    else:
        return pd.DataFrame()

    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    return df_imp
