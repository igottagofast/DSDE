"""Time-based chronological train/test splitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import ALL_TARGETS, PROCESSED_DATA_DIR, TRAIN_TEST_SPLIT_RATIO


def run_time_based_split(
    features_path: str,
    run_date: str,
    split_ratio: float = TRAIN_TEST_SPLIT_RATIO,
) -> dict:
    """Split feature dataset chronologically without shuffling and save Parquet splits."""
    src_path = Path(features_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Features dataset not found: {src_path}")

    df = pd.read_parquet(src_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    target_cols = [col for col in ALL_TARGETS if col in df.columns]
    if not target_cols:
        raise ValueError(f"No target columns from {ALL_TARGETS} found in dataset.")

    feature_cols = [col for col in df.columns if col not in (["Date"] + target_cols)]
    if not feature_cols:
        raise ValueError("No feature columns found in dataset.")

    # Convert continuous 5-day cumulative returns into binary classification labels
    df_cls = df.copy()
    for target in target_cols:
        df_cls[target] = (df_cls[target] > 0).astype(int)

    split_idx = int(len(df_cls) * split_ratio)
    if split_idx <= 0 or split_idx >= len(df_cls):
        raise ValueError(f"Invalid split index {split_idx} for dataset length {len(df_cls)}")

    train_df = df_cls.iloc[:split_idx].reset_index(drop=True)
    test_df = df_cls.iloc[split_idx:].reset_index(drop=True)

    if not (train_df["Date"].max() < test_df["Date"].min()):
        raise ValueError(
            "Temporal leakage detected: max train date is not strictly before min test date."
        )

    numeric_cols = feature_cols + target_cols
    for split_name, split_df in (("train", train_df), ("test", test_df)):
        if split_df[numeric_cols].isna().any().any():
            raise ValueError(f"NaN values detected in {split_name} split.")
        if np.isinf(split_df[numeric_cols].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite values detected in {split_name} split.")

    splits_dir = PROCESSED_DATA_DIR / run_date / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_path = splits_dir / "train.parquet"
    test_path = splits_dir / "test.parquet"
    meta_path = splits_dir / "split_metadata.json"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    train_range = [
        train_df["Date"].min().strftime("%Y-%m-%d"),
        train_df["Date"].max().strftime("%Y-%m-%d"),
    ]
    test_range = [
        test_df["Date"].min().strftime("%Y-%m-%d"),
        test_df["Date"].max().strftime("%Y-%m-%d"),
    ]

    metadata = {
        "run_date": run_date,
        "split_ratio": split_ratio,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_date_range": train_range,
        "test_date_range": test_range,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "run_date": run_date,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_date_range": train_range,
        "test_date_range": test_range,
    }
