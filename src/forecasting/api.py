import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc


def normalize_model_uri(uri: str) -> str:
    """Normalize MLflow model registry URIs.

    Convert legacy URIs like "models:/<name>/versions/<id>" to
    "models:/<name>/<id>" so MLflow model loading accepts them.
    """
    if uri.startswith("models:/") and "/versions/" in uri:
        parts = uri.split("/")
        if len(parts) == 4 and parts[0] == "models:" and parts[2] == "versions":
            version = parts[3]
            if version.isdigit():
                return f"{parts[0]}/{parts[1]}/{version}"
    return uri


class PredictRequest(BaseModel):
    server_id: str
    horizon: int = 1
    features: dict


class PredictResponse(BaseModel):
    server_id: str
    horizon: int
    prediction: float
    confidence_lower: float | None = None
    confidence_upper: float | None = None


def create_app(model_uri: str | None = None):
    app = FastAPI(title="CPU Forecasting API")
    logger = logging.getLogger("src.forecasting.api")
    uri = normalize_model_uri(model_uri or os.getenv("FORECAST_MODEL_URI", "models:/cpu_forecast/Production"))
    logger.info("Loading MLflow model from URI: %s", uri)
    print(f"Loading MLflow model from URI: {uri}")
    try:
        model = mlflow.pyfunc.load_model(uri)
    except Exception as exc:
        model = None
        load_error = str(exc)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_uri": uri}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        nonlocal model
        if model is None:
            raise HTTPException(status_code=500, detail=f"Model could not be loaded: {load_error}")
        payload = pd.DataFrame([request.features])
        try:
            score = model.predict(payload)
            if hasattr(score, "tolist"):
                score = float(score.tolist()[0])
            else:
                score = float(score)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return PredictResponse(
            server_id=request.server_id,
            horizon=request.horizon,
            prediction=score,
            confidence_lower=None,
            confidence_upper=None,
        )

    return app
