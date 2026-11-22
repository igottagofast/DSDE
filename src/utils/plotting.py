"""
Visualization and Plotting Utilities with Thai Font Support
"""
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def setup_thai_font(font_path: str = "./Sarabun-Regular.ttf") -> fm.FontProperties:
    """
    ตั้งค่าฟอนต์ภาษาไทยสำหรับการแสดงผลใน Matplotlib / Seaborn
    """
    # ตรวจสอบหาไฟล์ฟอนต์ในตำแหน่งต่างๆ
    search_paths = [
        font_path,
        str(Path(__file__).resolve().parent.parent.parent / "Sarabun-Regular.ttf"),
        "/usr/share/fonts/truetype/sarabun/Sarabun-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Thonburi.ttc",
        "/Library/Fonts/Arial Unicode.ttf"
    ]

    selected_font = None
    for p in search_paths:
        if os.path.exists(p):
            selected_font = p
            break

    if selected_font:
        thai_font = fm.FontProperties(fname=selected_font)
        plt.rcParams['font.family'] = thai_font.get_name()
    else:
        thai_font = fm.FontProperties()

    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid")
    return thai_font


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Correlation Matrix", figsize: tuple = (14, 10), save_path: str = None):
    """
    พล็อตแผนภาพความร้อน (Heatmap) แสดงค่าสหสัมพันธ์
    """
    thai_font = setup_thai_font()
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, center=0)
    plt.title(title, fontproperties=thai_font, fontsize=15, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontproperties=thai_font)
    plt.yticks(rotation=0, fontproperties=thai_font)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_prediction_step(dates, actual, predicted, target_name: str, last_n: int = 60, save_path: str = None):
    """
    พล็อต Step Plot เปรียบเทียบ Actual vs Predicted ในช่วง N วันทำการล่าสุด
    """
    thai_font = setup_thai_font()
    dates_tail = dates[-last_n:]
    actual_tail = actual[-last_n:]
    pred_tail = predicted[-last_n:]

    plt.figure(figsize=(14, 5))
    plt.step(dates_tail, actual_tail, label='Actual Direction', where='mid', color='#1f77b4', linewidth=2)
    plt.step(dates_tail, pred_tail, label='Predicted Direction', where='mid', color='#ff7f0e', linestyle='--', linewidth=2)
    plt.title(f'การพยากรณ์ทิศทางจริงเทียบกับค่าทำนาย: {target_name} ({last_n} วันทำการล่าสุด)', fontproperties=thai_font, fontsize=14, fontweight='bold')
    plt.xlabel('วันที่', fontproperties=thai_font)
    plt.ylabel('ทิศทาง (0 = ลง, 1 = ขึ้น)', fontproperties=thai_font)
    plt.legend(prop=thai_font)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
