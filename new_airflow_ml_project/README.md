# Thai Import & Export Stocks ML Pipeline (Apache Airflow)

โปรเจกต์ Machine Learning Pipeline ที่ใช้ **Apache Airflow** เป็นตัวจัดการลำดับงาน (Workflow Orchestrator) ตั้งแต่การนำเข้าข้อมูลดิบ, ตรวจสอบคุณภาพข้อมูล, ทำความสะอาด, สร้างฟีเจอร์ทางการเงินและอนุกรมเวลา, แบ่งชุดข้อมูลแบบอิงเวลา (Time-Based Split), เทรนโมเดล 5 อัลกอริทึมแบบขนาน (Dynamic Task Mapping) จนถึงการประเมินผล (Model Evaluation) และวิเคราะห์ Threshold Tuning รายปี

---

## โครงสร้างโปรเจกต์ (Project Structure)

```text
new_airflow_ml_project/
├── docker-compose.yaml        # รัน Airflow (Webserver, Scheduler, Postgres, LocalExecutor)
├── Dockerfile                 # Apache Airflow 2.10.4 (Python 3.11) + ML dependencies
├── requirements.txt           # Python ML libraries (pandas, pyarrow, sklearn, xgboost, lightgbm)
├── .env.example               # ตัวอย่าง Environment Variables สำหรับ Docker Compose
├── config/
│   └── settings.py            # พาธกลาง, รายชื่อหุ้น, ตัวแปรเป้าหมาย, พารามิเตอร์โมเดล
├── dags/
│   └── stock_ml_pipeline.py   # ไฟล์นิยาม DAG หลัก (TaskFlow API + Dynamic Task Mapping)
├── src/
│   ├── data/
│   │   ├── ingest.py          # สแกนไฟล์ CSV ดิบและสร้าง Manifest
│   │   ├── validate.py        # Data Quality Gate ตรวจสอบความครบถ้วนและ Schema
│   │   └── cleaner.py         # ทำความสะอาดข้อมูล -> บันทึก Parquet ลง data/02_intermediate/
│   ├── features/
│   │   └── build_features.py  # รวมข้อมูล + สร้างฟีเจอร์ + สร้าง Target 5 วัน -> data/03_processed/
│   └── models/
│       ├── split.py           # Time-Based Train/Test Split (80/20 ไม่ shuffle)
│       ├── train.py           # เทรนโมเดลแยกตามอัลกอริทึม -> outputs/models/<ds>/
│       └── evaluate.py        # ประเมินผล, Feature Importance, Threshold Tuning & Yearly Report
├── data/
│   ├── 01_raw/                # ข้อมูลดิบ CSV
│   ├── 02_intermediate/       # ข้อมูลหลังคลีน (Parquet + Manifest แยกตามรอบรัน)
│   └── 03_processed/          # Analytical Dataset และ Train/Test Splits
├── outputs/
│   ├── models/                # ไฟล์โมเดล .joblib แยกตามวันที่รัน
│   └── evaluation/            # model_comparison.csv, feature_importance.csv, threshold_tuning.csv, evaluation_report.md
└── notebooks/                 # ใช้เฉพาะ EDA หรือทดลองฟีเจอร์ (ไม่รันใน Airflow)
```

---

## ลำดับการทำงานของ DAG (`thai_stock_ml_pipeline`)

```text
ingest_raw_data
      │
      ▼
validate_raw_data
      │
      ▼
clean_data
      │
      ▼
build_features
      │
      ▼
time_based_split
      │
      ├──► train_model [Logistic Regression] ─┐
      ├──► train_model [Random Forest]       ─┤
      ├──► train_model [Gradient Boosting]   ─┼──► evaluate_and_select_best
      ├──► train_model [XGBoost]             ─┤         │
      └──► train_model [LightGBM]            ─┘         ▼
                                                threshold_tuning_and_report
```

---

## วิธีการเริ่มต้นใช้งาน (Quickstart with Docker Compose)

### 1. เตรียมไฟล์ `.env` และโฟลเดอร์ `logs/`
```bash
cd new_airflow_ml_project
cp .env.example .env
mkdir -p logs
```
*(บน macOS สามารถใช้ `AIRFLOW_UID=50000` หรือปรับให้ตรงกับ UID ของผู้ใช้ได้)*

### 2. Build และ Start Apache Airflow
```bash
docker compose up -d --build
```

รอให้ `airflow-init` ทำงานเสร็จและ service `airflow-webserver` / `airflow-scheduler` สถานะ `healthy`:
```bash
docker compose ps
```

### 3. เปิด Airflow UI และสั่งรัน Pipeline
- เปิดเบราว์เซอร์ไปที่ `http://localhost:8080`
- เข้าสู่ระบบด้วย Username: `airflow` / Password: `airflow` (ตามค่าใน `.env`)
- ค้นหา DAG ชื่อ **`thai_stock_ml_pipeline`** → กดเปิด **Unpause** → กด **Trigger DAG**

---

## การทดสอบและดีบักราย Task (CLI Verification)

ตรวจสอบว่า Airflow อ่าน DAG ได้โดยไม่มี Import Error:
```bash
docker compose exec airflow-scheduler airflow dags list-import-errors
```

ทดสอบรันทีละ Task ด้วย `airflow tasks test`:
```bash
docker compose exec airflow-scheduler airflow tasks test thai_stock_ml_pipeline ingest_raw_data 2026-11-22
docker compose exec airflow-scheduler airflow tasks test thai_stock_ml_pipeline validate_raw_data 2026-11-22
```

---

## ผลลัพธ์ที่ได้จาก Pipeline

ผลลัพธ์จะถูกบันทึกแยกตามวันที่รัน (`<run_date>`, เช่น `2026-11-22`) เพื่อความ Idempotent:

- **Cleaned Data**: `data/02_intermediate/<run_date>/df_*.parquet`
- **Features & Splits**: `data/03_processed/<run_date>/features_dataset.parquet` และ `splits/{train,test}.parquet`
- **Trained Models**: `outputs/models/<run_date>/<model_slug>__<target>.joblib`
- **Evaluation Results**: `outputs/evaluation/<run_date>/`
  - `model_comparison.csv` — ตารางเปรียบเทียบ ROC-AUC, Accuracy, Precision, Recall, F1, Confusion Matrix
  - `feature_importance.csv` — อันดับความสำคัญของฟีเจอร์จาก Best Model ของแต่ละ Target
  - `threshold_tuning.csv` — ผล Sensitivity Analysis ตามเกณฑ์ Threshold
  - `yearly_performance.csv` — ผลประเมินความเสถียรรายปี
  - `evaluation_report.md` — รายงานสรุปผลพร้อมใช้งาน
