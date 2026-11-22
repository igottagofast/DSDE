"""Model training utilities for parallel Airflow tasks."""
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config.settings import MODELS_DIR, RANDOM_STATE


def slugify_model_name(model_name: str) -> str:
    """Convert display model name into filesystem-safe slug."""
    return model_name.strip().lower().replace(" ", "_")


def build_classifier(model_name: str, random_state: int = RANDOM_STATE):
    """Instantiate classifier (preserving scikit-learn Pipeline inside Logistic Regression)."""
    if model_name == "Logistic Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=random_state)),
        ])
    if model_name == "Random Forest":
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)
    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=random_state)
    if model_name == "XGBoost":
        return XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )
    if model_name == "LightGBM":
        return LGBMClassifier(
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def train_and_save_model(model_name: str, split_info: dict, run_date: str) -> dict:
    """Train one classifier across all targets and serialize .joblib artifacts."""
    train_path = Path(split_info["train_path"])
    if not train_path.exists():
        raise FileNotFoundError(f"Training split not found: {train_path}")

    train_df = pd.read_parquet(train_path)
    feature_cols = split_info["feature_cols"]
    target_cols = split_info["target_cols"]

    X_train = train_df[feature_cols]
    model_dir = MODELS_DIR / run_date
    model_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify_model_name(model_name)

    artifacts: Dict[str, str] = {}
    for target_col in target_cols:
        y_train = train_df[target_col]
        clf = build_classifier(model_name=model_name, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)

        artifact_path = model_dir / f"{slug}__{target_col}.joblib"
        joblib.dump(clf, artifact_path)
        artifacts[target_col] = str(artifact_path)

    return {
        "model_name": model_name,
        "artifacts": artifacts,
    }
