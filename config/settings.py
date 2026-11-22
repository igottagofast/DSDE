"""
Project Configuration and Global Settings
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "01_raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "02_intermediate"
PROCESSED_DATA_DIR = DATA_DIR / "03_processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Google Colab Fallback Paths (if running inside Google Colab)
COLAB_DRIVE_BASE = Path("/content/drive/MyDrive/Colab Notebooks/DSDN")
COLAB_IMPORT_PATH = COLAB_DRIVE_BASE / "หุ้นนำเข้า/*.csv"
COLAB_EXPORT_PATH = COLAB_DRIVE_BASE / "หุ้นส่งออก/*.csv"
COLAB_MACRO_PATH = COLAB_DRIVE_BASE / "ปัจจัย/*.csv"

# Stock Universes
EXPORT_STOCKS_ALL = [
    'DELTA', 'HANA', 'KCE', 'CPF', 'TU', 'STA', 'BDMS', 'BH', 'AAI', 'ITC'
]

IMPORT_STOCKS_ALL = [
    'PTTGC', 'TOP', 'IRPC', 'IVL', 'GPSC', 'EGCO', 'RATCH', 'BCP', 'COM7', 'BGRIM'
]

# Filtered Stock Universes (Post-screening: removed short-history stocks)
EXPORT_STOCKS_8 = ['DELTA', 'HANA', 'KCE', 'CPF', 'TU', 'STA', 'BDMS', 'BH']
IMPORT_STOCKS_8 = ['PTTGC', 'TOP', 'IRPC', 'IVL', 'GPSC', 'EGCO', 'RATCH', 'BCP']

# Target Definitions for Modeling
PORTFOLIO_TARGETS = ['Import_Avg', 'Export_Avg']
SINGLE_STOCK_TARGETS = ['BH_Pct', 'BDMS_Pct', 'EGCO_Pct', 'BCP_Pct']
ALL_TARGETS = PORTFOLIO_TARGETS + SINGLE_STOCK_TARGETS

# Initial Macro Factors
INITIAL_MACRO_FACTORS = [
    'USD_THB', 'US_10Y', 'US_30Y', 'BOT_Rate', 'WTI', 'Brent', 'TH_10Y', 'TH_20Y'
]

# Additional Factors (Phase 2)
CURRENCIES = ['CNY_THB', 'EUR_THB', 'JPY_THB', 'USD_THB']
INDICES = ['SET50', 'SP500', 'SOX', 'VIX', 'DXY']

# Standard Thai to English Column Mapping
COLUMN_MAPPING = {
    'วันเดือนปี': 'Date',
    'ล่าสุด': 'Close',
    'ราคาเปิด': 'Open',
    'สูงสุด': 'High',
    'ต่ำสุด': 'Low',
    'ปริมาณ': 'Volume',
    '% เปลี่ยนแปลง': 'Pct_Change'
}

# Modeling Parameters
TRAIN_TEST_SPLIT_RATIO = 0.80
RANDOM_STATE = 42
