"""
Data Cleaning and Preprocessing Utilities
"""
import pandas as pd
import numpy as np


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    ตรวจสอบหาแถวที่ซ้ำซ้อน (Duplicate Rows) ใน DataFrame
    """
    duplicate_rows = df[df.duplicated()]
    total_duplicates = len(duplicate_rows)
    if total_duplicates > 0:
        print(f"⚠ พบแถวซ้ำทั้งหมด: {total_duplicates} แถว")
    else:
        print("✓ ไม่พบข้อมูลซ้ำซ้อน (No Duplicates)")
    return duplicate_rows


def check_and_show_nulls(df: pd.DataFrame):
    """
    ตรวจสอบหาค่าว่าง (Null/NaN) ใน DataFrame และพิมพ์สรุปจำนวนแถว
    """
    null_counts = df.isnull().sum()
    cols_with_null = null_counts[null_counts > 0]

    if cols_with_null.empty:
        print("✓ ไม่พบค่าว่างในข้อมูลนี้")
        return None

    null_rows = df[df.isnull().any(axis=1)]
    print(f"⚠ พบค่าว่างในคอลัมน์:")
    for col, count in cols_with_null.items():
        print(f"  - '{col}': {count} แถว")
    print(f"รวมแถวที่มีค่าว่างทั้งหมด: {len(null_rows)} แถว")
    return null_rows


def drop_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    ลบแถวที่มีค่าว่าง (Null/NaN) ใน DataFrame
    """
    initial_count = len(df)
    df_clean = df.dropna()
    deleted_count = initial_count - len(df_clean)
    if deleted_count > 0:
        print(f"✓ ลบค่าว่างสำเร็จ: ทั้งหมด {deleted_count} แถว")
    else:
        print("✓ ไม่พบค่าว่างในข้อมูลนี้")
    return df_clean


def drop_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """
    ลบคอลัมน์ที่ระบุออกจาก DataFrame อย่างปลอดภัย
    """
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    if existing_cols:
        df_clean = df.drop(columns=existing_cols)
        print(f"✓ ลบคอลัมน์สำเร็จ: {existing_cols}")
    else:
        print("✓ ไม่พบคอลัมน์ที่ระบุใน DataFrame (ไม่มีการลบ)")
        df_clean = df.copy()
    return df_clean


def fill_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    จัดการเติมค่าว่าง (Null/NaN) ด้วยวิธี Forward Fill (ffill) ตามด้วย Backward Fill (bfill)
    """
    initial_nulls = df.isnull().sum().sum()
    if initial_nulls == 0:
        return df.copy()

    df_filled = df.ffill().bfill()
    remaining_nulls = df_filled.isnull().sum().sum()
    print(f"✓ จัดการค่าว่างสำเร็จ (เริ่มต้น {initial_nulls} ค่า -> คงเหลือ {remaining_nulls} ค่า)")
    return df_filled


def safe_to_numeric(series: pd.Series) -> pd.Series:
    """
    แปลงคอลัมน์ข้อมูลต่าง ๆ (รองรับลูกน้ำ %, comma) ให้เป็นตัวเลขอย่างปลอดภัย
    """
    if series.dtype == 'object':
        return pd.to_numeric(
            series.astype(str).str.replace(',', '', regex=False).str.rstrip('%'),
            errors='coerce'
        )
    return pd.to_numeric(series, errors='coerce')


def standardize_columns(df: pd.DataFrame, column_mapping: dict = None) -> pd.DataFrame:
    """
    เปลี่ยนชื่อคอลัมน์ภาษาไทยเป็นภาษาอังกฤษมาตรฐานสากล
    """
    if column_mapping is None:
        from config.settings import COLUMN_MAPPING
        column_mapping = COLUMN_MAPPING

    df_out = df.copy()
    existing_map = {k: v for k, v in column_mapping.items() if k in df_out.columns}
    if existing_map:
        df_out = df_out.rename(columns=existing_map)
    return df_out


def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    แปลงคอลัมน์ Date เป็น Datetime และเรียงลำดับเวลาจากอดีตไปปัจจุบัน
    """
    d = df.copy()
    if "Date" in d.columns:
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    elif isinstance(d.index, pd.DatetimeIndex):
        d = d.reset_index().rename(columns={d.index.name or 'index': "Date"})
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    else:
        for cand in ["date", "วันเดือนปี", "Date_Parsed", "observation_date"]:
            if cand in d.columns:
                d = d.rename(columns={cand: "Date"})
                d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
                break
    return d.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
