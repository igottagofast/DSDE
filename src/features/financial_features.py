"""
Financial Domain Feature Engineering Functions
"""
import pandas as pd
import numpy as np
from src.data.cleaner import safe_to_numeric


def scale_correlation(corr_value, factor=10):
    """
    ฟังก์ชันสำหรับคูณสเกลค่า Correlation เพื่อขยายผลลัพธ์
    """
    if corr_value is None or pd.isna(corr_value):
        return corr_value
    return corr_value * factor


def calculate_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    คำนวณหาความผันผวนระหว่างวัน (Volatility) และทิศทางความเคลื่อนไหวราคาภายในวัน (Movement)
    สูตร:
      - Daily_Volatility = High - Low
      - Daily_Movement = Close - Open
    """
    df_out = df.copy()
    df_out['Date'] = pd.to_datetime(df_out['Date'])

    high = safe_to_numeric(df_out['High'])
    low = safe_to_numeric(df_out['Low'])
    close = safe_to_numeric(df_out['Close'])
    open_val = safe_to_numeric(df_out['Open'])

    df_out['Daily_Volatility'] = high - low
    df_out['Daily_Movement'] = close - open_val
    return df_out


def calculate_price_change_prev_day(df: pd.DataFrame, stock_name=None, close_col='Close') -> pd.DataFrame:
    """
    คำนวณส่วนต่างราคาปิดของวันนี้เทียบกับราคาปิดของวันทำการก่อนหน้า
    สูตร: Price_Change_Prev_Day = Close_today - Close_yesterday
    """
    df_out = df.copy()
    df_out['Date'] = pd.to_datetime(df_out['Date'])
    df_out = df_out.sort_values(by='Date')

    close_series = safe_to_numeric(df_out[close_col])
    df_out['Price_Change_Prev_Day'] = close_series.diff()

    if stock_name:
        df_out['Stock'] = stock_name

    return df_out.reset_index(drop=True)


def calculate_volatility_to_open(df: pd.DataFrame) -> pd.DataFrame:
    """
    คำนวณอัตราส่วนความผันผวนระหว่างวันเทียบราคาเปิด
    สูตร: Volatility_to_Open = (High - Low) / Open
    """
    df_out = df.copy()
    high = safe_to_numeric(df_out['High'])
    low = safe_to_numeric(df_out['Low'])
    open_val = safe_to_numeric(df_out['Open'])

    df_out['Volatility_to_Open'] = (high - low) / open_val
    return df_out


def calculate_relative_return(df_stock: pd.DataFrame, df_factor: pd.DataFrame, factor_name='FX', stock_name=None) -> pd.DataFrame:
    """
    คำนวณผลตอบแทนส่วนต่างเทียบกับปัจจัยภายนอก (เช่น อัตราแลกเปลี่ยน หรือดอกเบี้ย)
    สูตร: Relative_Return = Pct_Change_Stock - Pct_Change_Factor
    """
    df_stk_cp = df_stock.copy()
    df_fac_cp = df_factor.copy()

    df_stk_cp['Date'] = pd.to_datetime(df_stk_cp['Date'])
    df_fac_cp['Date'] = pd.to_datetime(df_fac_cp['Date'])

    df_stk_cp['Stock_Pct_Change'] = safe_to_numeric(df_stk_cp['Pct_Change'])
    df_fac_cp['Factor_Pct_Change'] = safe_to_numeric(df_fac_cp['Pct_Change'])

    merged = pd.merge(
        df_stk_cp[['Date', 'Stock_Pct_Change', 'Close', 'Open', 'High', 'Low']],
        df_fac_cp[['Date', 'Factor_Pct_Change']].rename(columns={'Factor_Pct_Change': f'{factor_name}_Pct_Change'}),
        on='Date',
        how='inner'
    )

    merged[f'Relative_Return_vs_{factor_name}'] = merged['Stock_Pct_Change'] - merged[f'{factor_name}_Pct_Change']

    if stock_name:
        merged['Stock'] = stock_name

    return merged.sort_values(by='Date').reset_index(drop=True)


def calculate_yield_slope(df_long: pd.DataFrame, df_short: pd.DataFrame, long_label='20Y', short_label='10Y') -> pd.DataFrame:
    """
    คำนวณความชันของเส้นอัตราผลตอบแทนพันธบัตร (Yield Curve Slope)
    สูตร: Yield_Slope = Yield_Long - Yield_Short
    """
    df_l = df_long.copy()
    df_s = df_short.copy()

    df_l['Date'] = pd.to_datetime(df_l['Date'])
    df_s['Date'] = pd.to_datetime(df_s['Date'])

    df_l['Yield_Long'] = safe_to_numeric(df_l['Close'])
    df_s['Yield_Short'] = safe_to_numeric(df_s['Close'])

    merged = pd.merge(
        df_l[['Date', 'Yield_Long']],
        df_s[['Date', 'Yield_Short']],
        on='Date',
        how='inner'
    )

    merged['Yield_Slope'] = merged['Yield_Long'] - merged['Yield_Short']
    merged = merged.rename(columns={
        'Yield_Long': f'Yield_{long_label}',
        'Yield_Short': f'Yield_{short_label}'
    })

    return merged.sort_values(by='Date').reset_index(drop=True)


def calculate_relative_strength(df_export: pd.DataFrame, df_import: pd.DataFrame, exp_stock_name='Export', imp_stock_name='Import') -> pd.DataFrame:
    """
    คำนวณอัตราส่วนความแข็งแกร่งเชิงสัมพัทธ์ระหว่างหุ้นส่งออกและนำเข้า
    สูตร: Relative_Strength_Ratio = Close_Export / Close_Import
    """
    df_exp = df_export.copy()
    df_imp = df_import.copy()

    df_exp['Date'] = pd.to_datetime(df_exp['Date'])
    df_imp['Date'] = pd.to_datetime(df_imp['Date'])

    df_exp['Export_Close'] = safe_to_numeric(df_exp['Close'])
    df_imp['Import_Close'] = safe_to_numeric(df_imp['Close'])

    merged = pd.merge(
        df_exp[['Date', 'Export_Close']],
        df_imp[['Date', 'Import_Close']],
        on='Date',
        how='inner'
    )

    merged['Relative_Strength_Ratio'] = merged['Export_Close'] / merged['Import_Close']
    return merged.sort_values(by='Date').reset_index(drop=True)


def convert_price_to_usd(df_stock: pd.DataFrame, df_usd_thb: pd.DataFrame) -> pd.DataFrame:
    """
    แปลงราคาปิดหุ้นจากสกุลเงินบาทเป็นดอลลาร์สหรัฐ (THB -> USD)
    สูตร: Stock_Close_USD = Stock_Close_THB / USD_THB_Rate
    """
    df_s = df_stock.copy()
    df_fx = df_usd_thb.copy()

    df_s['Date'] = pd.to_datetime(df_s['Date'])
    df_fx['Date'] = pd.to_datetime(df_fx['Date'])

    df_s['Close_THB'] = safe_to_numeric(df_s['Close'])
    df_fx['USD_THB_Rate'] = safe_to_numeric(df_fx['Close'])

    merged = pd.merge(
        df_s[['Date', 'Close_THB']],
        df_fx[['Date', 'USD_THB_Rate']],
        on='Date',
        how='inner'
    )

    merged['Stock_Close_USD'] = merged['Close_THB'] / merged['USD_THB_Rate']
    merged['Close'] = merged['Stock_Close_USD']
    return merged.sort_values(by='Date').reset_index(drop=True)
