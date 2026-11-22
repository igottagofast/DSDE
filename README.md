# Thai Import & Export Stocks Predictive Pipeline (DSDE Project)

ระบบวิศวกรรมข้อมูลและการเรียนรู้ของเครื่อง (Data Engineering & Machine Learning Pipeline) สำหรับการวิเคราะห์และพยากรณ์ทิศทางราคาหุ้นกลุ่มนำเข้า (Import) และกลุ่มส่งออก (Export) ของไทย โดยอาศัยตัวแปรมหภาคและดัชนีตลาดโลก

---

## 🏛️ สถาปัตยกรรมโปรเจกต์ (Project Structure)

```text
newdsde/
├── README.md                      # ภาพรวมโครงการและคู่มือการใช้งาน
├── requirements.txt               # Dependencies ทั้งหมดของระบบ
├── .gitignore                     # กำหนดไฟล์ที่ไม่ต้อง Track ใน Git
│
├── config/                        # การตั้งค่าระบบและตัวแปรส่วนกลาง
│   ├── __init__.py
│   └── settings.py                # รายชื่อหุ้น, ตัวแปรกำหนด Target, Path กลาง
│
├── data/                          # แหล่งจัดเก็บข้อมูล (Medallion Architecture)
│   ├── 01_raw/                    # ข้อมูลดิบ (Raw CSVs)
│   ├── 02_intermediate/           # ข้อมูลหลังคลีน (Standardized Columns, Formatted Dates)
│   └── 03_processed/              # ข้อมูล Final Analytical Matrix พร้อมเทรน
│
├── notebooks/                     # ลำดับการทดลองแบบ Modular
│   ├── 01_data_cleaning.ipynb     # [Step 1] การโหลดและทำความสะอาดข้อมูลดิบ
│   ├── 02_exploratory_data_analysis.ipynb # [Step 2] การวิเคราะห์ Correlation & EDA
│   ├── 03_feature_engineering.ipynb       # [Step 3] วิศวกรรมฟีเจอร์ทางการเงินและอนุกรมเวลา
│   ├── 04_model_evaluation.ipynb          # [Step 4] Time-Based Train/Test Split และประเมินผลโมเดล
│   └── 05_reporting_insights.ipynb        # [Step 5] การจูน Threshold, ผลรายปี และบทสรุปผู้บริหาร
│
├── src/                           # Source Code โมดูลหลักสำหรับเรียกใช้งานซ้ำ
│   ├── data/
│   │   ├── cleaner.py             # ฟังก์ชันจัดการ Null, Duplicates, Standardizing
│   │   └── loader.py              # โหลดไฟล์แบบ Multi-source (Local & Drive)
│   ├── features/
│   │   ├── financial_features.py  # Yield slope, Relative strength, Volatility to Open
│   │   └── time_series_features.py# RSI, Lags, Rolling Volatility, Momentum
│   ├── models/
│   │   ├── train.py               # Time-Based Split, โมเดล Classifiers (LogReg, RF, XGB, LGBM)
│   │   ├── evaluate.py            # การประเมินผลแบบ Multi-target, Feature Importance
│   │   └── threshold_tuning.py    # Sensitivity Analysis และ Yearly Robustness Breakdown
│   └── utils/
│       └── plotting.py            # การตั้งค่าฟอนต์ภาษาไทยและฟังก์ชันพล็อตกราฟ
│
├── outputs/                       # โฟลเดอร์เก็บผลลัพธ์
│   ├── figures/                   # รูปภาพและแผนภูมิประกอบรายงานบทที่ 4
│   ├── models/                    # โมเดลที่ผ่านการเทรน
│   └── tables/                    # ตารางผลลัพธ์การทดลอง
│
└── docs/
    └── data_dictionary.md         # พจนานุกรมข้อมูลและรายละเอียดตัวแปร
```

---

## 🚀 วิธีการเริ่มต้นใช้งาน (Getting Started)

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. ลำดับการรันการทดลอง (Workflow Execution)
สามารถเปิดโฟลเดอร์ `notebooks/` และรันตามลำดับเลข:
1. `01_data_cleaning.ipynb`: ตรวจสอบและเตรียมความพร้อมข้อมูลหุ้นและปัจจัยมหภาค
2. `02_exploratory_data_analysis.ipynb`: สำรวจความสัมพันธ์ผ่าน Scatter Plot และ Correlation Heatmap
3. `03_feature_engineering.ipynb`: สกัดตัวแปรที่มีนัยสำคัญทางการเงินและตัวแปร Lagged ย้อนหลัง
4. `04_model_evaluation.ipynb`: เทรนโมเดลจำแนกประเภท (Classification) 5 อัลกอริทึม
5. `05_reporting_insights.ipynb`: วิเคราะห์ความเสถียรรายปีและการปรับจูน Decision Threshold เพื่อนำผลไปเขียนรายงานวิจัยบทที่ 4 และ 5

---

## 📊 ผลการทดลองสำคัญ (Key Findings)

* **ความสำเร็จของ Feature Engineering**: การเพิ่ม 23+ ฟีเจอร์ใหม่ช่วยเพิ่มค่า **ROC-AUC ในทุกตลาดเป้าหมาย** โดยกลุ่มพอร์ตโฟลิโอ `Export_Avg` เพิ่มขึ้นสูงสุด **+12.21%** (0.5421 $\rightarrow$ 0.6083) และ `Import_Avg` เพิ่มขึ้น **+9.83%** (0.5312 $\rightarrow$ 0.5834)
* **ปัจจัยชี้นำที่ทรงอิทธิพลที่สุด**: ดัชนีดอลลาร์สหรัฐ (`DXY`) และความผันผวนของอัตราแลกเปลี่ยน เป็นตัวแปรที่มี Feature Importance สูงสุดในการอธิบายการไหลเวียนของกระแสเงินทุน (Fund Flow) ในตลาดหุ้นไทย
* **Portfolio Diversification Effect**: การพยากรณ์พอร์ตเฉลี่ยมีประสิทธิภาพสูงกว่าและเสถียรกว่าหุ้นรายตัว เนื่องจากความเสี่ยงเฉพาะตัว (Unsystematic Risk) ได้รับการเจือจางลง ทำให้โมเดลตอบสนองต่อตัวแปรมหภาคเชิงโครงสร้างได้ชัดเจนขึ้น
