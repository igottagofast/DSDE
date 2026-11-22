"""Data cleaning and standardization utilities for Airflow intermediate outputs."""
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import COLUMN_MAPPING, INTERMEDIATE_DATA_DIR, NON_EQUITY_DATASETS


def safe_to_numeric(series: pd.Series) -> pd.Series:
    """Convert numeric strings with commas or '%' suffixes safely into floats."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.rstrip("%")
    )
    return pd.to_numeric(cleaned, errors="coerce")


def standardize_columns(df: pd.DataFrame, column_mapping: Optional[dict] = None) -> pd.DataFrame:
    """Strip BOM/whitespace from headers and map Thai columns to English."""
    mapping = column_mapping if column_mapping is not None else COLUMN_MAPPING
    df_out = df.copy()
    df_out.columns = [str(col).lstrip("\ufeff").strip() for col in df_out.columns]
    rename_map = {k: v for k, v in mapping.items() if k in df_out.columns}
    if rename_map:
        df_out = df_out.rename(columns=rename_map)
    return df_out


def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Date column, drop invalid dates, deduplicate, and sort ascending."""
    df_out = df.copy()
    if "Date" in df_out.columns:
        df_out["Date"] = pd.to_datetime(df_out["Date"], errors="coerce")
    elif isinstance(df_out.index, pd.DatetimeIndex):
        df_out = df_out.reset_index().rename(columns={df_out.index.name or "index": "Date"})
        df_out["Date"] = pd.to_datetime(df_out["Date"], errors="coerce")
    else:
        for candidate in ("date", "วันเดือนปี", "Date_Parsed", "observation_date"):
            if candidate in df_out.columns:
                df_out = df_out.rename(columns={candidate: "Date"})
                df_out["Date"] = pd.to_datetime(df_out["Date"], errors="coerce")
                break

    if "Date" not in df_out.columns:
        raise KeyError(f"Date column not found in columns: {list(df_out.columns)}")

    return (
        df_out.dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )


def _compute_daily_return(df: pd.DataFrame) -> pd.Series:
    """Convert Pct_Change into decimal daily return, falling back to Close pct_change."""
    if "Pct_Change" not in df.columns:
        return df["Close"].pct_change()

    raw_pct = df["Pct_Change"]
    has_percent_sign = raw_pct.astype(str).str.contains("%", na=False).any()
    numeric_pct = safe_to_numeric(raw_pct)
    if has_percent_sign or raw_pct.dtype == object:
        returns = numeric_pct / 100.0
    else:
        returns = numeric_pct

    fallback = df["Close"].pct_change()
    return returns.fillna(fallback)


def clean_single_dataframe(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Clean one raw financial dataframe into standardized intermediate format."""
    df_clean = standardize_columns(df)
    df_clean = ensure_date_column(df_clean)

    if dataset_name in NON_EQUITY_DATASETS and "Volume" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Volume"])

    for col in ("Close", "Open", "High", "Low"):
        if col in df_clean.columns:
            df_clean[col] = safe_to_numeric(df_clean[col])

    df_clean = df_clean.dropna(subset=["Close"]).reset_index(drop=True)
    df_clean["Return"] = _compute_daily_return(df_clean)

    # Ensure string columns have uniform dtype for Parquet serialization
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].astype(str)

    return df_clean


def run_cleaning_pipeline(manifest_path: str, run_date: str) -> dict:
    """Clean all raw CSVs listed in manifest and persist Parquet files."""
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets", {})

    output_dir = INTERMEDIATE_DATA_DIR / run_date
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_files = []
    for name, csv_path in sorted(datasets.items()):
        df_raw = pd.read_csv(csv_path, encoding="utf-8-sig")
        df_clean = clean_single_dataframe(df_raw, dataset_name=name)
        out_path = output_dir / f"df_{name}.parquet"
        df_clean.to_parquet(out_path, index=False)
        cleaned_files.append(str(out_path))

    return {
        "run_date": run_date,
        "intermediate_dir": str(output_dir),
        "cleaned_count": len(cleaned_files),
    }
