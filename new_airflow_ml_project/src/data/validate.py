"""Data quality gate for validating raw CSV datasets prior to cleaning."""
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config.settings import COLUMN_MAPPING, REQUIRED_DATASETS


def _standardize_header(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    df_out.columns = [str(col).lstrip("\ufeff").strip() for col in df_out.columns]
    rename_map = {k: v for k, v in COLUMN_MAPPING.items() if k in df_out.columns}
    if rename_map:
        df_out = df_out.rename(columns=rename_map)
    return df_out


def validate_raw_manifest(
    manifest_path: str,
    required_datasets: Optional[List[str]] = None,
) -> dict:
    """Validate dataset completeness, schema, and date parseability."""
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    datasets: Dict[str, str] = manifest.get("datasets", {})
    expected = required_datasets if required_datasets is not None else REQUIRED_DATASETS

    missing_required = sorted(set(expected) - set(datasets.keys()))
    if missing_required:
        raise ValueError(f"Missing required datasets in manifest: {missing_required}")

    errors = []
    summaries = {}
    for name, csv_path in sorted(datasets.items()):
        path_obj = Path(csv_path)
        if not path_obj.exists():
            errors.append(f"{name}: missing file at {csv_path}")
            continue

        try:
            df = pd.read_csv(path_obj, encoding="utf-8-sig")
        except Exception as exc:
            errors.append(f"{name}: failed to read CSV ({exc})")
            continue

        if df.empty:
            errors.append(f"{name}: CSV is empty")
            continue

        df = _standardize_header(df)
        missing_cols = [col for col in ("Date", "Close") if col not in df.columns]
        if missing_cols:
            errors.append(f"{name}: missing columns {missing_cols}; found {list(df.columns)}")
            continue

        parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
        valid_dates = int(parsed_dates.notna().sum())
        if valid_dates == 0:
            errors.append(f"{name}: zero valid parsed dates")
            continue

        summaries[name] = {
            "rows": int(len(df)),
            "valid_dates": valid_dates,
            "min_date": parsed_dates.min().strftime("%Y-%m-%d"),
            "max_date": parsed_dates.max().strftime("%Y-%m-%d"),
        }

    if errors:
        raise ValueError("Data validation failed:\n" + "\n".join(errors))

    report = {
        "status": "passed",
        "run_date": manifest.get("run_date"),
        "validated_count": len(summaries),
        "datasets": summaries,
    }
    report_path = manifest_file.parent / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "passed",
        "validated_count": len(summaries),
        "report_path": str(report_path),
    }
