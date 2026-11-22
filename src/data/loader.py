"""
Data Loading Utilities (Local Files & Google Drive Support)
"""
import os
import glob
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
from config.settings import (
    RAW_DATA_DIR,
    COLAB_DRIVE_BASE,
    COLUMN_MAPPING
)
from src.data.cleaner import standardize_columns, ensure_date_column


def get_data_dir() -> Path:
    """
    คืนค่า Path ของโฟลเดอร์เก็บข้อมูลดิบ โดยตรวจสอบว่าอยู่ใน Colab หรือ Local
    """
    if os.path.exists(COLAB_DRIVE_BASE):
        return COLAB_DRIVE_BASE
    return RAW_DATA_DIR


def load_csv_files(file_paths: List[str], prefix: str = "df_") -> Dict[str, pd.DataFrame]:
    """
    โหลดไฟล์ CSV หลายไฟล์และคืนค่าเป็น Dictionary {ชื่อตัวแปร: DataFrame}
    พร้อมปรับชื่อคอลัมน์มาตรฐานและเรียงลำดับวันที่
    """
    dfs = {}
    for path in file_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        var_name = f"{prefix}{name}"
        try:
            df = pd.read_csv(path)
            df = standardize_columns(df, COLUMN_MAPPING)
            df = ensure_date_column(df)
            dfs[var_name] = df
        except Exception as e:
            print(f"⚠ ไม่สามารถอ่านไฟล์ {path}: {e}")
    return dfs


def load_stocks_group(group_dir_name: str, prefix: str = "df_") -> Dict[str, pd.DataFrame]:
    """
    โหลดกลุ่มหุ้นจากโฟลเดอร์ย่อย เช่น 'หุ้นนำเข้า' หรือ 'หุ้นส่งออก'
    """
    base = get_data_dir()
    search_path = os.path.join(base, group_dir_name, "*.csv")
    files = sorted(glob.glob(search_path))
    if not files:
        # ลองค้นหาแบบ fallback ใน RAW_DATA_DIR ตรงๆ
        search_path = os.path.join(RAW_DATA_DIR, group_dir_name, "*.csv")
        files = sorted(glob.glob(search_path))

    return load_csv_files(files, prefix=prefix)
