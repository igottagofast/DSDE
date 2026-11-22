"""Raw dataset discovery and manifest creation for Airflow ingestion."""
import json
from pathlib import Path
from typing import Dict, Optional

from config.settings import INTERMEDIATE_DATA_DIR, RAW_CATEGORY_DIRS, RAW_DATA_DIR

PREFIX_STRIPS = (
    "ราคาย้อนหลัง หุ้น ",
    "ราคาย้อนหลังของ สัญญาซื้อขายล่วงหน้าน้ำมันดิบ ",
    "ราคาย้อนหลังของ สัญญาซื้อขายล่วงหน้าน้ำมัน",
    "ข้อมูลผลตอบแทนย้อนหลังของพันธบัตร ",
    "อัตราแลกเปลี่ยนย้อนหลัง ",
    "ข้อมูลย้อนหลังของ ดัชนี ",
    "ข้อมูลย้อนหลังของ ",
)

EXACT_OR_SUBSTRING_MAP = (
    ("สหรัฐอเมริกา 10 ปี", "US_10Y"),
    ("สหรัฐ 10 ปี", "US_10Y"),
    ("สหรัฐอเมริกา 30 ปี", "US_30Y"),
    ("สหรัฐ 30 ปี", "US_30Y"),
    ("ไทย 10 ปี", "TH_10Y"),
    ("ไทย 20 ปี", "TH_20Y"),
    ("Baltic Dry Index", "BDI"),
    ("CBOE Volatility Index", "VIX"),
    ("Philadelphia Semiconductor Index", "SOX"),
    ("S&P 500", "SP500"),
    ("SET_50", "SET50"),
    ("SET 50", "SET50"),
    ("US Dollar Index", "DXY"),
    ("เบรนท์", "Brent"),
    ("FM_RT_013_formatted", "TH_RATE"),
)


def normalize_dataset_name(filename: str) -> Optional[str]:
    """Normalize raw Thai/English CSV filename into canonical dataset key.

    Duplicate export copies ending with '-2' (e.g. 'US Dollar Index-2.csv')
    return None so they are skipped during ingestion.
    """
    stem = Path(filename).stem.strip()
    if stem.endswith("-2"):
        return None

    clean_name = stem
    for prefix in PREFIX_STRIPS:
        clean_name = clean_name.replace(prefix, "")

    for pattern, canonical in EXACT_OR_SUBSTRING_MAP:
        if pattern in clean_name:
            return canonical

    return clean_name.strip().replace(" ", "_")


def discover_raw_files(raw_dir: Path = RAW_DATA_DIR) -> Dict[str, str]:
    """Scan categorized raw data directories and return {dataset_name: path}."""
    discovered: Dict[str, str] = {}
    base_dir = Path(raw_dir)

    for folder_candidates in RAW_CATEGORY_DIRS.values():
        chosen_files = []
        for folder_name in folder_candidates:
            candidate_dir = base_dir / folder_name
            if candidate_dir.exists():
                csv_files = sorted(candidate_dir.glob("*.csv"))
                if csv_files:
                    chosen_files = csv_files
                    break

        for csv_path in chosen_files:
            dataset_name = normalize_dataset_name(csv_path.name)
            if dataset_name is None:
                continue
            discovered.setdefault(dataset_name, str(csv_path.resolve()))

    return dict(sorted(discovered.items()))


def run_ingest(run_date: str, raw_dir: Optional[str] = None) -> dict:
    """Create raw file manifest for the given Airflow run_date."""
    source_dir = Path(raw_dir).resolve() if raw_dir else RAW_DATA_DIR
    if not source_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {source_dir}")

    datasets = discover_raw_files(source_dir)
    if not datasets:
        raise FileNotFoundError(f"No CSV files discovered under: {source_dir}")

    run_dir = INTERMEDIATE_DATA_DIR / run_date
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "raw_manifest.json"

    payload = {
        "run_date": run_date,
        "raw_dir": str(source_dir),
        "file_count": len(datasets),
        "datasets": datasets,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "run_date": run_date,
        "manifest_path": str(manifest_path),
        "file_count": len(datasets),
        "datasets": sorted(datasets.keys()),
    }
