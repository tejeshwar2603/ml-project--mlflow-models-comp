import os
import re
import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
from .chatbot import AIOpsChatbot, _llm_configured
from .forecast_service import ForecastService
from .rag import DEFAULT_VECTOR_STORE_PATH, build_vector_store_from_environment


FEATURE_COLUMNS = [
    "cpu_utilization",
    "ram_utilization",
    "disk_utilization",
    "network_utilization",
    "cpu_utilization_lag_1",
    "cpu_utilization_lag_3",
    "cpu_utilization_lag_7",
    "cpu_utilization_lag_14",
    "cpu_utilization_lag_30",
    "cpu_utilization_roll_mean_3",
    "cpu_utilization_roll_std_3",
    "cpu_utilization_roll_min_3",
    "cpu_utilization_roll_max_3",
    "cpu_utilization_roll_mean_7",
    "cpu_utilization_roll_std_7",
    "cpu_utilization_roll_min_7",
    "cpu_utilization_roll_max_7",
    "cpu_ram_ratio",
    "cpu_disk_ratio",
    "day_of_week",
    "month",
    "is_weekend",
]


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


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5
    analysis_mode: str = "general"
    include_prediction: bool = False
    prediction_request: PredictRequest | None = None
    ml_output: dict | None = None


class ForecastRequest(BaseModel):
    server_id: str
    horizon_days: int = 7
    model: str = "auto"


class ChatResponse(BaseModel):
    answer: str
    analysis_mode: str
    risk: dict
    recommendation: str
    action_plan: list[str]
    related_incidents: list[str]
    runbooks: list[str]
    jira_ticket_draft: dict
    executive_summary: str
    sources: list[dict]
    ml_output: dict | None = None


def _ensure_vector_store(path: str | Path) -> None:
    store_path = Path(path)
    if store_path.exists():
        return
    logger = logging.getLogger("src.forecasting.api")
    logger.warning("Vector store missing at %s; building from local artifacts...", store_path)
    build_vector_store_from_environment(store_path)


def create_app(model_uri: str | None = None):
    app = FastAPI(title="CPU Forecasting API")
    static_dir = Path(__file__).resolve().parent / "static"
    logger = logging.getLogger("src.forecasting.api")
    vector_store_path = os.getenv("RAG_VECTOR_STORE_PATH", str(DEFAULT_VECTOR_STORE_PATH))
    try:
        _ensure_vector_store(vector_store_path)
    except Exception as exc:
        logger.warning("Could not build vector store automatically: %s", exc)
    uri = normalize_model_uri(model_uri or os.getenv("FORECAST_MODEL_URI", "models:/cpu_forecast/Production"))
    logger.info("Loading MLflow model from URI: %s", uri)
    print(f"Loading MLflow model from URI: {uri}")
    # Import mlflow lazily so the API can start in minimal environments
    model = None
    load_error = None
    try:
        import mlflow.pyfunc as _mlflow_pyfunc

        try:
            model = _mlflow_pyfunc.load_model(uri)
        except Exception as exc:
            model = None
            load_error = str(exc)
            # MLflow 3.x removed model-registry "stages" (Production/Staging/etc) in
            # favor of aliases, so legacy "models:/<name>/<stage>" URIs like the
            # default above now fail to resolve on any version. Fall back to the
            # latest registered version so the API still starts with a usable model.
            match = re.match(r"^models:/([^/]+)/(Production|Staging|Archived|None)$", uri)
            if match:
                registered_name = match.group(1)
                try:
                    from mlflow import MlflowClient

                    client = MlflowClient()
                    versions = client.search_model_versions(f"name='{registered_name}'")
                    if versions:
                        latest = max(versions, key=lambda v: int(v.version))
                        fallback_uri = f"models:/{registered_name}/{latest.version}"
                        model = _mlflow_pyfunc.load_model(fallback_uri)
                        load_error = None
                        uri = fallback_uri
                        logger.warning(
                            "Stage-based URI %s not resolvable under this MLflow version; loaded latest version instead: %s",
                            match.group(0),
                            fallback_uri,
                        )
                except Exception as fallback_exc:
                    load_error = f"{load_error}; fallback to latest version also failed: {fallback_exc}"
    except Exception:
        # mlflow not installed or import failed; continue with model=None
        model = None
        load_error = "mlflow not available in environment"
    chatbot = None
    chatbot_error = None
    try:
        chatbot = AIOpsChatbot(vector_store_path)
    except Exception as exc:
        chatbot_error = str(exc)

    forecast_service = ForecastService(xgb_model=model, xgb_load_error=load_error)

    @app.get("/")
    def root():
        return {
            "message": "AIOps Chatbot API",
            "status": "running",
            "docs": "http://127.0.0.1:8001/docs",
            "endpoints": {
                "health": "GET /health",
                "chat": "POST /chat",
                "capacity_summary": "POST /llm/capacity-summary",
                "draft_ticket": "POST /llm/draft-ticket",
                "executive_report": "POST /llm/executive-report",
                "root_cause": "POST /llm/root-cause",
                "servers": "GET /servers",
                "model_metrics": "GET /models/metrics",
                "forecast": "POST /forecast",
                "capacity_overview": "GET /capacity-overview",
                "model_comparison": "GET /model-comparison",
            },
            "ui": "GET /ui",
        }

    @app.get("/health")
    def health():
        from .chatbot import DEFAULT_GROK_MODEL, _grok_api_key

        grok_key = _grok_api_key()
        llm_model = os.getenv("AIOPS_LLM_MODEL")
        if not llm_model:
            llm_model = DEFAULT_GROK_MODEL if grok_key else os.getenv("OPENAI_API_KEY") and "gpt-4.1-mini" or None
        return {
            "status": "ok",
            "model_uri": uri,
            "forecast_model_loaded": model is not None,
            "forecast_model_load_error": load_error,
            "llm_configured": _llm_configured(),
            "llm_provider": "grok" if grok_key else ("openai" if os.getenv("OPENAI_API_KEY") else None),
            "llm_model": llm_model,
            "vector_store_path": vector_store_path,
            "vector_store_ready": Path(vector_store_path).exists(),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        nonlocal model
        if model is None:
            raise HTTPException(status_code=500, detail=f"Model could not be loaded: {load_error}")
        missing = [col for col in FEATURE_COLUMNS if col not in request.features]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing feature columns: {missing}")
        payload = pd.DataFrame([request.features])[FEATURE_COLUMNS]
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

    @app.get("/servers")
    def list_servers():
        return {"servers": forecast_service.list_servers()}

    @app.get("/models/metrics")
    def model_metrics():
        return {"models": forecast_service.model_metrics()}

    @app.post("/forecast")
    def forecast(request: ForecastRequest):
        try:
            return forecast_service.forecast(
                request.server_id, horizon_days=request.horizon_days, model=request.model
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/capacity-overview")
    def capacity_overview(horizon_days: int = 7):
        return {"servers": forecast_service.fleet_overview(horizon_days=horizon_days)}

    @app.get("/model-comparison")
    def model_comparison():
        results_path = Path("artifacts/model_comparison/results.json")
        if not results_path.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "Model comparison analysis has not been run yet. Run "
                    "'python -m src.forecasting.model_comparison' to generate it "
                    "(takes a few minutes; also logs plots to MLflow)."
                ),
            )
        return json.loads(results_path.read_text())

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest):
        nonlocal chatbot
        if chatbot is None:
            raise HTTPException(
                status_code=500,
                detail=f"Chatbot vector store could not be loaded: {chatbot_error}",
            )
        ml_output = request.ml_output
        if request.include_prediction:
            if request.prediction_request is None:
                raise HTTPException(
                    status_code=400,
                    detail="prediction_request is required when include_prediction=true",
                )
            prediction = predict(request.prediction_request)
            ml_output = prediction.model_dump()
        try:
            result = chatbot.answer(
                request.question,
                ml_output=ml_output,
                top_k=request.top_k,
                analysis_mode=request.analysis_mode,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return ChatResponse(**result)

    @app.get("/ui", response_class=FileResponse)
    def ui():
        ui_path = static_dir / "chatbot_ui.html"
        if not ui_path.exists():
            raise HTTPException(status_code=404, detail="Chatbot UI not found.")
        return FileResponse(ui_path, media_type="text/html")

    @app.post("/llm/capacity-summary")
    def capacity_summary(request: ChatRequest):
        """Generate a capacity planning summary using the forecast and RAG."""
        nonlocal chatbot
        if chatbot is None:
            raise HTTPException(
                status_code=500,
                detail=f"Chatbot vector store could not be loaded: {chatbot_error}",
            )
        ml_output = request.ml_output
        if request.include_prediction and request.prediction_request:
            prediction = predict(request.prediction_request)
            ml_output = prediction.model_dump()
        try:
            result = chatbot.answer(
                request.question or "Summarize capacity planning recommendations for the next 7 days.",
                ml_output=ml_output,
                top_k=request.top_k,
                analysis_mode="capacity_planning",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return ChatResponse(**result)

    @app.post("/llm/draft-ticket")
    def draft_ticket(request: ChatRequest):
        """Draft a Jira ticket from the forecast and RAG context."""
        nonlocal chatbot
        if chatbot is None:
            raise HTTPException(
                status_code=500,
                detail=f"Chatbot vector store could not be loaded: {chatbot_error}",
            )
        ml_output = request.ml_output
        if request.include_prediction and request.prediction_request:
            prediction = predict(request.prediction_request)
            ml_output = prediction.model_dump()
        try:
            result = chatbot.answer(
                request.question or "Draft a Jira ticket for this capacity issue.",
                ml_output=ml_output,
                top_k=request.top_k,
                analysis_mode="jira_ticket",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        # Return just the Jira ticket draft for convenience
        ticket_draft = result.get("jira_ticket_draft", {})
        return {
            "ticket_draft": ticket_draft,
            "risk_level": result.get("risk", {}).get("level"),
            "recommendation": result.get("recommendation"),
            "sources": result.get("sources", []),
        }

    @app.post("/llm/executive-report")
    def executive_report(request: ChatRequest):
        """Generate an executive summary from forecast and RAG."""
        nonlocal chatbot
        if chatbot is None:
            raise HTTPException(
                status_code=500,
                detail=f"Chatbot vector store could not be loaded: {chatbot_error}",
            )
        ml_output = request.ml_output
        if request.include_prediction and request.prediction_request:
            prediction = predict(request.prediction_request)
            ml_output = prediction.model_dump()
        try:
            result = chatbot.answer(
                request.question or "Provide an executive summary of capacity risks and recommendations.",
                ml_output=ml_output,
                top_k=request.top_k,
                analysis_mode="executive_report",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return ChatResponse(**result)

    @app.post("/llm/root-cause")
    def root_cause(request: ChatRequest):
        """Identify likely root causes using forecast pattern and historical incidents."""
        nonlocal chatbot
        if chatbot is None:
            raise HTTPException(
                status_code=500,
                detail=f"Chatbot vector store could not be loaded: {chatbot_error}",
            )
        ml_output = request.ml_output
        if request.include_prediction and request.prediction_request:
            prediction = predict(request.prediction_request)
            ml_output = prediction.model_dump()
        try:
            result = chatbot.answer(
                request.question or "What is the likely root cause of this forecast pattern?",
                ml_output=ml_output,
                top_k=request.top_k,
                analysis_mode="root_cause",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return ChatResponse(**result)

    return app
