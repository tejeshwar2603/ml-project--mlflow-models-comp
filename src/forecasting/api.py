import os
import re
import json
import uuid
import logging
import threading
from pathlib import Path, PurePosixPath
from urllib.parse import unquote_plus
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
from . import object_storage
from .chatbot import AIOpsChatbot, _llm_configured
from .forecast_service import ForecastService
from .rag import DEFAULT_VECTOR_STORE_PATH, build_vector_store_from_environment

# Tracks background model-comparison analysis runs kicked off via
# POST /datasets/{id}/model-comparison, keyed by dataset_id. This is the
# in-process fallback for when Airflow (which the same analysis also runs
# automatically after every dataset upload/train - see dags/train_and_compare_models.py)
# isn't up or reachable; a single uvicorn worker process is assumed.
_comparison_jobs_lock = threading.Lock()
_comparison_jobs: dict[str, dict] = {}


def _run_comparison_job(dataset_id: str) -> None:
    from . import model_comparison

    try:
        model_comparison.run_comparison(dataset_id=dataset_id)
        with _comparison_jobs_lock:
            _comparison_jobs[dataset_id] = {"status": "done", "error": None}
    except Exception as exc:
        with _comparison_jobs_lock:
            _comparison_jobs[dataset_id] = {"status": "error", "error": str(exc)}


def _start_comparison_job(dataset_id: str) -> bool:
    """Returns False if a job for this dataset_id is already running."""
    with _comparison_jobs_lock:
        current = _comparison_jobs.get(dataset_id)
        if current and current["status"] == "running":
            return False
        _comparison_jobs[dataset_id] = {"status": "running", "error": None}
    threading.Thread(target=_run_comparison_job, args=(dataset_id,), daemon=True).start()
    return True


# Must match forecast_service.FEATURE_COLUMNS / training.FEATURE_COLUMNS exactly.
FEATURE_COLUMNS = [
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
    dataset_id: str = "synthetic"


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
            # "champion" alias set by training.promote_best_xgboost_version() if one
            # exists, and only then to the newest registered version. Preferring the
            # alias matters once ad-hoc dataset uploads start registering their own
            # versions (see /datasets/{id}/train) - those are deliberately NOT
            # promoted to champion, so "just take the highest version number" would
            # otherwise let a tiny test upload silently become the model serving
            # everyone else's forecasts on the next restart.
            match = re.match(r"^models:/([^/]+)/(Production|Staging|Archived|None)$", uri)
            if match:
                registered_name = match.group(1)
                try:
                    from mlflow import MlflowClient

                    client = MlflowClient()
                    fallback_uri = None
                    try:
                        registered = client.get_registered_model(registered_name)
                        if "champion" in (registered.aliases or {}):
                            fallback_uri = f"models:/{registered_name}@champion"
                    except Exception:
                        pass
                    if fallback_uri is None:
                        versions = client.search_model_versions(f"name='{registered_name}'")
                        if versions:
                            latest = max(versions, key=lambda v: int(v.version))
                            fallback_uri = f"models:/{registered_name}/{latest.version}"
                    if fallback_uri:
                        model = _mlflow_pyfunc.load_model(fallback_uri)
                        load_error = None
                        uri = fallback_uri
                        logger.warning(
                            "Stage-based URI %s not resolvable under this MLflow version; loaded %s instead.",
                            match.group(0),
                            fallback_uri,
                        )
                except Exception as fallback_exc:
                    load_error = f"{load_error}; alias/latest-version fallback also failed: {fallback_exc}"
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
                "model_comparison": "GET /model-comparison?dataset_id=synthetic",
                "trigger_model_comparison": "POST /datasets/{dataset_id}/model-comparison - run the full comparison analysis for a dataset in the background (also runs automatically via Airflow after upload/train)",
                "datasets": "GET /datasets",
                "upload_dataset": "POST /datasets/upload (multipart file: .xlsx/.xls/.csv) - now also trains + logs to MLflow automatically",
                "train_dataset": "POST /datasets/{dataset_id}/train - retrigger training/MLflow/Airflow for an already-uploaded dataset",
                "storage_events": "POST /storage/events - MinIO bucket-notification webhook target; not meant to be called directly",
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
    def list_servers(dataset_id: str = "synthetic"):
        try:
            return {"servers": forecast_service.list_servers(dataset_id=dataset_id)}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/servers/{server_id}/history")
    def server_history(server_id: str, dataset_id: str = "synthetic", days: int = 30):
        try:
            return forecast_service.historical_utilization(server_id, dataset_id=dataset_id, days=days)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/models/metrics")
    def model_metrics():
        return {"models": forecast_service.model_metrics()}

    @app.get("/datasets")
    def list_datasets():
        return {"datasets": forecast_service.list_datasets()}

    @app.post("/datasets/upload")
    async def upload_dataset(
        file: UploadFile = File(...),
        dataset_label: str | None = Form(None),
        dataset_id: str | None = Form(None),
    ):
        content = await file.read()
        filename = file.filename or "uploaded"
        try:
            df = forecast_service.read_tabular_bytes(filename, content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read file: {exc}")

        # Pass an explicit dataset_id (e.g. "kafka-live") to upsert/replace an existing
        # dataset in place, instead of minting a new upload-<uuid> id each call - this is
        # what the streaming consumer uses to keep one rolling "live" dataset up to date.
        resolved_id = dataset_id or f"upload-{uuid.uuid4().hex[:8]}"
        label = dataset_label or file.filename or resolved_id
        try:
            summary = forecast_service.register_uploaded_dataset(resolved_id, df, label=label)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # Uploading a dataset used to just make it available for live forecasting and
        # nothing else - no MLflow run, no Airflow trigger. That silent gap is exactly
        # what /datasets/{id}/train below closes; call it automatically here so
        # "I added a dataset" actually results in a visible MLflow run without a
        # second manual step.
        summary["training"] = _train_dataset_and_notify_airflow(resolved_id)
        return summary

    def _train_dataset_and_notify_airflow(target_dataset_id: str) -> dict:
        from . import training as training_module

        dataset_path = forecast_service.get_dataset_path(target_dataset_id)
        result = {"mlflow": None, "mlflow_error": None, "airflow_triggered": False, "airflow_error": None}
        if dataset_path:
            try:
                result["mlflow"] = training_module.train_on_dataset(target_dataset_id, dataset_path)
            except Exception as exc:
                result["mlflow_error"] = str(exc)
        else:
            result["mlflow_error"] = (
                f"'{target_dataset_id}' has no persisted file to train from (only uploaded/streamed "
                "datasets do). The built-in 'synthetic' dataset is trained via "
                "'python -m src.forecasting.training' instead."
            )

        airflow_url = os.getenv("AIRFLOW_API_URL", "http://localhost:8090")
        try:
            import requests

            resp = requests.post(
                f"{airflow_url}/api/v1/dags/train_and_compare_models/dagRuns",
                json={"conf": {"dataset_id": target_dataset_id, "dataset_path": dataset_path}},
                auth=(os.getenv("AIRFLOW_USER", "admin"), os.getenv("AIRFLOW_PASSWORD", "admin")),
                timeout=3,
            )
            if resp.status_code in (200, 201):
                result["airflow_triggered"] = True
            else:
                result["airflow_error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:
            result["airflow_error"] = f"Airflow not reachable at {airflow_url}: {exc}"
        return result

    @app.post("/datasets/{target_dataset_id}/train")
    def train_dataset(target_dataset_id: str):
        """Manually (re)trigger training for an already-uploaded dataset - the same
        thing /datasets/upload now does automatically, exposed separately so a
        dataset can be retrained later (e.g. after Kafka has streamed in more rows)
        without re-uploading."""
        try:
            forecast_service.dataset_summary(target_dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return _train_dataset_and_notify_airflow(target_dataset_id)

    def _dataset_id_from_object_key(key: str) -> str:
        # Deterministic per object path, not a random uuid - the same key landing
        # again (a real-time feed overwriting the same object) should upsert the
        # same dataset rather than minting a new one every time.
        stem = PurePosixPath(key).stem
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", stem).strip("-").lower()
        return f"minio-{slug}" if slug else f"minio-{uuid.uuid4().hex[:8]}"

    @app.post("/storage/events")
    async def storage_events(request: Request):
        """Webhook target MinIO's bucket notification calls the instant an object is
        created/overwritten in the configured bucket (see docker-compose.yml's
        minio-init service, which wires the notification target + event binding).
        Pulls the object's bytes back out via the S3 API and feeds them through the
        exact same read_tabular_bytes() -> register_uploaded_dataset() path a manual
        upload goes through - MinIO is a third source for one ingestion pipeline,
        not a separate one."""
        expected_token = os.getenv("MINIO_WEBHOOK_TOKEN")
        if expected_token:
            auth_header = request.headers.get("authorization", "")
            if auth_header != f"Bearer {expected_token}":
                raise HTTPException(status_code=401, detail="Invalid or missing webhook auth token.")

        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid webhook payload: {exc}")

        results = []
        for record in body.get("Records") or []:
            if not record.get("eventName", "").startswith("s3:ObjectCreated"):
                continue
            s3_info = record.get("s3", {})
            bucket = s3_info.get("bucket", {}).get("name")
            key = unquote_plus(s3_info.get("object", {}).get("key", ""))
            if not bucket or not key:
                continue
            try:
                content = object_storage.fetch_object(bucket, key)
                df = forecast_service.read_tabular_bytes(key, content)
                resolved_id = _dataset_id_from_object_key(key)
                summary = forecast_service.register_uploaded_dataset(resolved_id, df, label=f"minio://{bucket}/{key}")
                summary["training"] = _train_dataset_and_notify_airflow(resolved_id)
                results.append({"bucket": bucket, "key": key, "dataset_id": resolved_id, "status": "ok", "summary": summary})
            except Exception as exc:
                results.append({"bucket": bucket, "key": key, "status": "error", "error": str(exc)})
        return {"processed": len(results), "results": results}

    @app.post("/forecast")
    def forecast(request: ForecastRequest):
        try:
            return forecast_service.forecast(
                request.server_id,
                horizon_days=request.horizon_days,
                model=request.model,
                dataset_id=request.dataset_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/capacity-overview")
    def capacity_overview(horizon_days: int = 7, dataset_id: str = "synthetic"):
        return {"servers": forecast_service.fleet_overview(horizon_days=horizon_days, dataset_id=dataset_id)}

    @app.get("/model-comparison")
    def model_comparison(dataset_id: str = "synthetic"):
        suffix = "" if dataset_id == "synthetic" else f"_{dataset_id}"
        results_path = Path(f"artifacts/model_comparison/results{suffix}.json")
        if results_path.exists():
            # parse_constant guards against stale/edge-case results files containing
            # NaN/Infinity (e.g. a dataset with missing readings that hit a study
            # that doesn't fully guard against it) - Starlette's JSON response can't
            # serialize those back out, so turn them into null here instead of 500ing.
            return json.loads(results_path.read_text(), parse_constant=lambda _: None)

        with _comparison_jobs_lock:
            job = dict(_comparison_jobs.get(dataset_id) or {})
        if job.get("status") == "running":
            return {"status": "running", "dataset_id": dataset_id}
        if job.get("status") == "error":
            return {"status": "error", "dataset_id": dataset_id, "message": job.get("error")}

        cli_arg = "" if dataset_id == "synthetic" else f" {dataset_id}"
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model comparison analysis has not been run yet for dataset '{dataset_id}'. "
                f"POST /datasets/{dataset_id}/model-comparison to generate it in the background "
                f"(takes a few minutes; also logs plots to MLflow), or run "
                f"'python -m src.forecasting.model_comparison{cli_arg}' from the CLI."
            ),
        )

    @app.post("/datasets/{target_dataset_id}/model-comparison")
    def trigger_model_comparison(target_dataset_id: str):
        if target_dataset_id != "synthetic":
            try:
                forecast_service.dataset_summary(target_dataset_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
        started = _start_comparison_job(target_dataset_id)
        return {"status": "started" if started else "already_running", "dataset_id": target_dataset_id}

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
