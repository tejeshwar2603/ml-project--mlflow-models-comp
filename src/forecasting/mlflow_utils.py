import mlflow
import mlflow.pyfunc
import pandas as pd
import os
from pathlib import Path


def start_run(experiment_name="cpu_forecasting"):
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()


def log_experiment(run_name, params, metrics, artifacts=None, model=None, model_name=None):
    with start_run(experiment_name=params.get("experiment_name", "cpu_forecasting")) as run:
        mlflow.set_tag("run_name", run_name)
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))
        artifacts = artifacts or {}
        for artifact_name, artifact_path in artifacts.items():
            if os.path.exists(artifact_path):
                if os.path.isdir(artifact_path):
                    mlflow.log_artifacts(artifact_path, artifact_name)
                else:
                    mlflow.log_artifact(artifact_path, artifact_name)
        if model is not None:
            mlflow.pyfunc.log_model("model", python_model=model, artifacts=None)
            if model_name:
                mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
        return run.info.run_id


def save_prediction_csv(predictions, output_path):
    df = pd.DataFrame(predictions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return str(output_path)
