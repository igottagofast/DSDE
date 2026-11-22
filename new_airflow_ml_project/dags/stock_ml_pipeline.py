"""Apache Airflow DAG for the Thai Import & Export Stocks ML Pipeline."""
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

from config.settings import MODEL_NAMES

DEFAULT_ARGS = {
    "owner": "dsde_team",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="thai_stock_ml_pipeline",
    description="End-to-end Thai Import/Export stocks ML pipeline from raw CSV ingest to evaluation",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["ml", "stocks", "evaluation", "dsde"],
)
def stock_ml_pipeline():
    """Orchestrate data ingestion, validation, cleaning, feature engineering, training, and evaluation."""

    @task
    def ingest_raw_data() -> dict:
        from src.data.ingest import run_ingest

        ds = get_current_context()["ds"]
        return run_ingest(run_date=ds)

    @task
    def validate_raw_data(ingest_info: dict) -> dict:
        from src.data.validate import validate_raw_manifest

        return validate_raw_manifest(manifest_path=ingest_info["manifest_path"])

    @task
    def clean_data(ingest_info: dict, validation_info: dict) -> dict:
        from src.data.cleaner import run_cleaning_pipeline

        if validation_info.get("status") != "passed":
            raise ValueError(f"Validation did not pass: {validation_info}")

        ds = get_current_context()["ds"]
        return run_cleaning_pipeline(
            manifest_path=ingest_info["manifest_path"],
            run_date=ds,
        )

    @task
    def build_features(clean_info: dict) -> dict:
        from src.features.build_features import run_feature_pipeline

        ds = get_current_context()["ds"]
        return run_feature_pipeline(
            intermediate_dir=clean_info["intermediate_dir"],
            run_date=ds,
        )

    @task
    def time_based_split(features_info: dict) -> dict:
        from src.models.split import run_time_based_split

        ds = get_current_context()["ds"]
        return run_time_based_split(
            features_path=features_info["features_path"],
            run_date=ds,
        )

    @task
    def train_model(model_name: str, split_info: dict) -> dict:
        from src.models.train import train_and_save_model

        ds = get_current_context()["ds"]
        return train_and_save_model(
            model_name=model_name,
            split_info=split_info,
            run_date=ds,
        )

    @task
    def evaluate_and_select_best(trained_models: list, split_info: dict) -> dict:
        from src.models.evaluate import evaluate_saved_models

        ds = get_current_context()["ds"]
        return evaluate_saved_models(
            trained_results=trained_models,
            split_info=split_info,
            run_date=ds,
        )

    @task
    def threshold_tuning_and_report(eval_summary: dict, split_info: dict) -> dict:
        from src.models.evaluate import run_post_eval_analysis

        ds = get_current_context()["ds"]
        return run_post_eval_analysis(
            eval_summary=eval_summary,
            split_info=split_info,
            run_date=ds,
        )

    ingest_info = ingest_raw_data()
    validation_info = validate_raw_data(ingest_info)
    clean_info = clean_data(ingest_info, validation_info)
    features_info = build_features(clean_info)
    split_info = time_based_split(features_info)

    trained_models = train_model.partial(split_info=split_info).expand(
        model_name=MODEL_NAMES
    )
    eval_summary = evaluate_and_select_best(trained_models, split_info)
    threshold_tuning_and_report(eval_summary, split_info)


stock_ml_pipeline()
