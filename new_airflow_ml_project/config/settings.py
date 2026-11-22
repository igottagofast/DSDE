"""Project configuration and global settings for the Airflow ML pipeline."""
import os
from pathlib import Path

# Base Paths (configurable via environment variables for Docker / local runs)
PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent)
).resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = Path(os.environ.get("RAW_DATA_DIR", DATA_DIR / "01_raw")).resolve()
INTERMEDIATE_DATA_DIR = DATA_DIR / "02_intermediate"
PROCESSED_DATA_DIR = DATA_DIR / "03_processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"

# Raw data subdirectories by category
RAW_CATEGORY_DIRS = {
    "import_stocks": ["หุ้นนำเข้า"],
    "export_stocks": ["หุ้นส่งออก"],
    "fx": ["อัตราแลกเปลี่ยน"],
    "us_interest": ["ดอกเบี้ย"],
    "th_bond": ["พันธบัตรไทย"],
    "oil": ["ราคาน้ำมัน"],
    "th_rate": ["ดอกเบี้ยอ้างอิงแห่งประเทศไทย"],
    "indices": ["ข้อมูลดัชนี", "ดัชนี"],
}

# Stock Universes
EXPORT_STOCKS_ALL = [
    "DELTA", "HANA", "KCE", "CPF", "TU", "STA", "BDMS", "BH", "AAI", "ITC",
]
IMPORT_STOCKS_ALL = [
    "PTTGC", "TOP", "IRPC", "IVL", "GPSC", "EGCO", "RATCH", "BCP", "COM7", "BGRIM",
]

# Filtered Stock Universes (8 stocks per group with full historical coverage)
EXPORT_STOCKS_8 = ["DELTA", "HANA", "KCE", "CPF", "TU", "STA", "BDMS", "BH"]
IMPORT_STOCKS_8 = ["PTTGC", "TOP", "IRPC", "IVL", "GPSC", "EGCO", "RATCH", "BCP"]

# Required datasets validated before cleaning & feature engineering
REQUIRED_DATASETS = (
    IMPORT_STOCKS_8
    + EXPORT_STOCKS_8
    + [
        "Brent",
        "US_10Y",
        "USD_THB",
        "TH_10Y",
        "TH_20Y",
        "SET50",
        "SP500",
        "SOX",
        "VIX",
        "DXY",
        "CNY_THB",
        "JPY_THB",
    ]
)

# Non-equity datasets where Volume column should be dropped
NON_EQUITY_DATASETS = [
    "TH_10Y",
    "TH_20Y",
    "USD_THB",
    "CNY_THB",
    "EUR_THB",
    "JPY_THB",
    "US_10Y",
    "US_30Y",
    "BDI",
    "VIX",
    "SOX",
    "SP500",
    "SET50",
    "DXY",
    "TH_RATE",
]

# Target Definitions
PORTFOLIO_TARGETS = ["Import_Avg", "Export_Avg"]
SINGLE_STOCK_TARGETS = ["BH_Pct", "BDMS_Pct", "EGCO_Pct", "BCP_Pct"]
ALL_TARGETS = PORTFOLIO_TARGETS + SINGLE_STOCK_TARGETS

# Standard Thai to English Column Mapping (covers both '% เปลี่ยน' and '% เปลี่ยนแปลง')
COLUMN_MAPPING = {
    "วันเดือนปี": "Date",
    "ล่าสุด": "Close",
    "ราคาเปิด": "Open",
    "สูงสุด": "High",
    "ต่ำสุด": "Low",
    "ปริมาณ": "Volume",
    "% เปลี่ยน": "Pct_Change",
    "% เปลี่ยนแปลง": "Pct_Change",
}

# Modeling & Evaluation Parameters
FORECAST_HORIZON_DAYS = 5
TRAIN_TEST_SPLIT_RATIO = 0.80
RANDOM_STATE = 42
MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM",
]
THRESHOLD_GRID = [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60]
