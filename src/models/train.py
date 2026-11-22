"""
Model Definitions and Training Pipeline
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


def time_based_train_test_split(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    dates: pd.Series,
    split_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    แบ่งชุดข้อมูลแบบอิงลำดับเวลา (Time-Based Train-Test Split) โดยไม่สลับลำดับ (No Shuffling)
    """
    df_prep = pd.concat([dates.reset_index(drop=True), X.reset_index(drop=True), Y.reset_index(drop=True)], axis=1)
    df_prep['Date'] = pd.to_datetime(df_prep['Date'])
    df_prep = df_prep.sort_values(by='Date').reset_index(drop=True)

    dates_sorted = df_prep['Date']
    X_sorted = df_prep[X.columns]
    Y_sorted = df_prep[Y.columns]

    split_idx = int(len(df_prep) * split_ratio)

    X_train = X_sorted.iloc[:split_idx].reset_index(drop=True)
    X_test = X_sorted.iloc[split_idx:].reset_index(drop=True)
    Y_train = Y_sorted.iloc[:split_idx].reset_index(drop=True)
    Y_test = Y_sorted.iloc[split_idx:].reset_index(drop=True)
    train_dates = dates_sorted.iloc[:split_idx].reset_index(drop=True)
    test_dates = dates_sorted.iloc[split_idx:].reset_index(drop=True)

    return X_train, X_test, Y_train, Y_test, train_dates, test_dates


def get_models_dict(random_state: int = 42) -> Dict:
    """
    คืนค่า Dictionary ของโมเดลจำแนกประเภท (Classifiers) ที่ใช้ในการทดลอง
    """
    clfs = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        'Random Forest': RandomForestClassifier(random_state=random_state, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state)
    }

    if XGBClassifier is not None:
        clfs['XGBoost'] = XGBClassifier(random_state=random_state, eval_metric='logloss', n_jobs=-1)

    if LGBMClassifier is not None:
        clfs['LightGBM'] = LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1)

    return clfs
