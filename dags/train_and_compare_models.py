"""Airflow DAG: train all 4 forecasting models, promote the best XGBoost
version, and refresh the model-comparison dashboard.

This is the orchestration piece the project didn't have before: training.py
and model_comparison.py previously only ran when someone manually typed the
command. This DAG schedules that (default: daily) and can also be triggered
manually right after a new dataset is uploaded via the app UI.

Runs inside the mlflowcomp-airflow-* containers (see docker-compose.yml /
Dockerfile.airflow), which mount the whole project at /opt/airflow/project
with PYTHONPATH set so `from src.forecasting... import ...` works directly.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _train(model_name: str, **_):
    from src.forecasting.training import train_one_model

    result = train_one_model(model_name)
    print(f"{model_name}: MAE={result['mae']:.3f} run_id={result['run_id']}")
    return result


def _promote_best_model(**_):
    from src.forecasting.training import promote_best_xgboost_version

    version = promote_best_xgboost_version()
    if version is None:
        raise RuntimeError("Could not determine a best XGBoost version to promote.")
    print(f"Promoted cpu_forecast version {version} to alias 'champion'.")


def _run_model_comparison(**_):
    from src.forecasting.model_comparison import run_comparison

    results = run_comparison()
    print(f"Model comparison finished in {results['generated_in_seconds']}s. MAE summary: {results['mae_summary']}")


default_args = {
    "owner": "aiops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="train_and_compare_models",
    description="Train arima/sarima/xgboost/gru in parallel, promote the best model, refresh the comparison dashboard.",
    default_args=default_args,
    schedule=timedelta(days=1),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["aiops", "forecasting"],
) as dag:
    train_tasks = [
        PythonOperator(
            task_id=f"train_{name}",
            python_callable=_train,
            op_kwargs={"model_name": name},
        )
        for name in ("arima", "sarima", "xgboost", "gru")
    ]

    promote = PythonOperator(task_id="promote_best_model", python_callable=_promote_best_model)
    compare = PythonOperator(task_id="refresh_model_comparison", python_callable=_run_model_comparison)

    train_tasks >> promote >> compare
