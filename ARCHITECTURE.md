# CPU Forecasting & AIOps Platform — Architecture

Time-series CPU-utilization forecasting (ARIMA / SARIMA / XGBoost / GRU) with an
MLflow-tracked training pipeline, Airflow orchestration, three dataset ingestion
paths (manual upload, Kafka streaming, MinIO object storage), and a chatbot
dashboard for capacity-planning questions. This describes the actual current
wiring of this repository, not an idealized target.

**Stack**: FastAPI `:8000` · MLflow `:5000` · Postgres `:5433` · Airflow `:8090` ·
Kafka `:9095` · Zookeeper `:2182` · MinIO `:9010`/`:9011` · Docker Compose

## System diagram

```mermaid
flowchart TB
    subgraph ING["INGESTION"]
        PROD["producer.py<br/>simulated server-metrics"]
        TOPIC[["Kafka topic<br/>server-metrics"]]
        CONS["consumer.py<br/>buffers, flushes every N msgs"]
        MINIO[("MinIO bucket<br/>cpu-utilization-datasets")]
    end

    subgraph SRV["SERVING API — :8000"]
        API["api.py<br/>FastAPI routes"]
        FS["forecast_service.py<br/>dataset registry, live forecast"]
    end

    subgraph CORE["ML CORE"]
        FEAT["features.py<br/>lag/rolling, imputation"]
        MODELS["models.py<br/>ARIMA · SARIMA · XGBoost · GRU"]
        TRAIN["training.py<br/>fit, evaluate, promote champion"]
        COMP["model_comparison.py<br/>7-tab offline benchmark"]
    end

    subgraph REG["TRACKING & REGISTRY"]
        MLF["MLflow server :5000<br/>cpu_forecast registry, @champion alias"]
        PG[("Postgres :5433<br/>mlflow + airflow DBs")]
    end

    subgraph ORCH["ORCHESTRATION"]
        DAG["train_and_compare_models DAG<br/>train x4 >> promote >> compare"]
        AF["Airflow webserver+scheduler :8090"]
    end

    subgraph UI["CHATBOT + DASHBOARD"]
        HTML["chatbot_ui.html — served at /ui"]
        BOT["chatbot.py + rag.py<br/>Groq/Llama + local vector store"]
    end

    PROD --> TOPIC --> CONS -- "POST /datasets/upload" --> API
    MINIO -- "bucket-notification webhook<br/>POST /storage/events" --> API
    API -- "GetObject (boto3)" --> MINIO
    API --> FS
    FS --> FEAT --> MODELS
    API -- "POST /datasets/upload, /train" --> TRAIN
    API -- "POST .../model-comparison" --> COMP
    TRAIN --> MLF
    COMP --> MLF
    MLF --> PG
    API -- "notify (best-effort)" --> DAG
    DAG --> AF
    AF --> TRAIN
    AF --> COMP
    FS -- "GET /servers, /forecast, /capacity-overview" --> HTML
    API --> BOT --> HTML
    MLF -. "@champion model" .-> FS

    classDef ing fill:#0d7d72,stroke:#0d7d72,color:#fff
    class MINIO ing
```

Solid arrows are calls that happen on every request; the dotted arrow is the model
registry handing the currently-promoted XGBoost version back to the serving layer at
process startup. Airflow's path is redundant with the API's own in-process
training/comparison calls — it exists so the same pipeline can also run on a daily
schedule with no HTTP request involved. MinIO's arrow is bidirectional: it pushes a
webhook the instant an object lands, and the API pulls that object's bytes back out
over the S3 API in response.

## Request & data flow, end to end

| # | Step | Where |
|---|------|-------|
| 1 | A dataset arrives via one of three paths: a human uploads a CSV/XLSX, `consumer.py` flushes a batch of Kafka readings, or a `.csv`/`.xlsx` object is created/overwritten in the MinIO bucket (MinIO's own bucket notification fires the instant this happens). | `POST /datasets/upload`, `POST /storage/events` |
| 2 | Schema-validated, timestamp-parsed, deduplicated, persisted to disk, and registered in the in-memory dataset map. | `forecast_service.register_uploaded_dataset()` |
| 3 | All 4 models are fit synchronously in the request (arima, sarima, xgboost, gru) against that one dataset; XGBoost gets registered as a new MLflow model version (never promoted to champion). | `training.train_on_dataset()` |
| 4 | Airflow's REST API is notified (best-effort, 3s timeout) to also run the full pipeline for this dataset on its own schedule/queue. | `_train_dataset_and_notify_airflow()` |
| 5 | A user (or the Airflow DAG) can separately trigger the full 7-tab offline benchmark for that dataset — walk-forward horizons, dataset-size learning curves, null-handling robustness, a Diebold-Mariano significance test, a per-server heatmap. | `POST /datasets/{id}/model-comparison` |
| 6 | Live forecasts and fleet capacity recommendations are served per-dataset, per-server, on demand — ARIMA/SARIMA refit on the fly, XGBoost runs a recursive multi-day loop off the registered `@champion` model. | `POST /forecast`, `GET /capacity-overview` |
| 7 | The dashboard renders all of the above, plus a Groq/Llama-backed chatbot that answers capacity questions using a local RAG vector store built from predictions and docs. | `GET /ui`, `POST /chat` |

## Layers

### 1. Ingestion — Kafka & MinIO

Three ways a dataset enters the system, all converging on the same
`register_uploaded_dataset()` → train → notify-Airflow pipeline:

- `src/streaming/producer.py` — publishes synthetic readings to the Kafka `server-metrics` topic.
- `src/streaming/consumer.py` — batches messages and POSTs them to `/datasets/upload`.
- **MinIO** (`:9010` API / `:9011` console) — an S3-compatible object store for
  "drop a file, it just gets ingested" real-time feeds. A `.csv`/`.xlsx` object
  created or overwritten in the `cpu-utilization-datasets` bucket fires a bucket
  notification webhook at `POST /storage/events` (bearer-token authenticated via
  `MINIO_WEBHOOK_TOKEN`); the handler fetches the object's bytes back out over the
  S3 API and feeds them through the identical parsing/registration path a manual
  upload uses. The dataset ID is derived from the object key (`live/foo.csv` →
  `minio-foo`), so the same key being overwritten repeatedly — the real-time-feed
  case — upserts one dataset in place rather than creating a new one each time.
  - `src/forecasting/object_storage.py` — the boto3 S3 client (`fetch_object()`).
  - `src/forecasting/api.py`'s `POST /storage/events` — the webhook handler.
  - `docker-compose.yml`'s `minio` service — registers the webhook notification
    target via `MINIO_NOTIFY_WEBHOOK_*` env vars at startup (not a runtime
    `mc admin config set` + restart, which needs an interactive TTY that doesn't
    exist in a non-interactive init container).
  - `docker-compose.yml`'s `minio-init` service — one-shot: creates the bucket,
    binds `s3:ObjectCreated:*` events for `.csv`/`.xlsx` to that already-registered
    webhook target.

### 2. Serving API — FastAPI, `:8000`

Every HTTP route: forecasting, dataset management, model comparison, chat, and the
dashboard's static file.

- `src/forecasting/api.py` — all routes (see table below).
- `src/forecasting/forecast_service.py` — the dataset registry (in-memory + persisted
  CSVs under `artifacts/uploaded_datasets/`), live per-server forecasting, fleet
  overview / rightsizing recommendations.

### 3. ML Core — features, models, training

The actual forecasting algorithms and the feature engineering that feeds them.

- `src/forecasting/features.py` — `build_features()`: lag features (1/3/7/14/30
  days), rolling mean/std/min/max (3/7/14/30-day windows), calendar features
  (day-of-week, month, weekend flag), and `validate_and_fill()` for per-server
  interpolate/ffill/bfill imputation. **Does not** include the raw
  cpu/ram/disk/network readings or any ratio derived from them as model inputs —
  those leak the label (see Known state below).
- `src/forecasting/models.py` — `ARIMAForecaster` / `SARIMAForecaster` (order chosen
  per-series via ADF stationarity + ACF, not a fixed `(p,d,q)`), `XGBoostForecaster`
  (the production model), `GRUForecaster` / `LSTMForecaster` / `TFTForecaster`.
- `src/forecasting/training.py` — `train_one_model()`, `train_and_evaluate()`,
  `train_on_dataset()`, and `promote_best_xgboost_version()` (champion selection —
  see Known state).
- `src/forecasting/evaluation.py` — MAE/MSE/RMSE/MAPE/SMAPE/R²/explained-variance,
  `compare_models()`.
- `src/forecasting/model_comparison.py` — the 7-tab offline benchmark
  (`run_comparison(dataset_id=...)`), runnable against the built-in synthetic
  dataset or any uploaded one.

`FEATURE_COLUMNS` (the exact, current model input — defined identically in both
`training.py` and `api.py`):

```
cpu_utilization_lag_1, cpu_utilization_lag_3, cpu_utilization_lag_7,
cpu_utilization_lag_14, cpu_utilization_lag_30,
cpu_utilization_roll_mean_3, cpu_utilization_roll_std_3,
cpu_utilization_roll_min_3, cpu_utilization_roll_max_3,
cpu_utilization_roll_mean_7, cpu_utilization_roll_std_7,
cpu_utilization_roll_min_7, cpu_utilization_roll_max_7,
day_of_week, month, is_weekend
```

### 4. Tracking & Registry — MLflow, `:5000`

Every training run, metric, and model version. `cpu_forecast` is the registered
model name; `@champion` is the alias live serving actually loads (MLflow 3.x
aliases replace the deprecated Production/Staging stages).

- Backend store: `postgresql+psycopg2://airflow:airflow@localhost:5433/mlflow`
- Artifact root: `./mlruns`

### 5. Orchestration — Airflow, `:8090`

One DAG: train all 4 models in parallel, promote the best XGBoost version, refresh
the comparison dashboard. Runs daily, or on-demand per dataset (conf carries
`dataset_id`/`dataset_path` from a manual trigger).

- `dags/train_and_compare_models.py`

### 6. Dashboard & Chatbot

Single-page dashboard (forecasts, fleet overview, 7-tab model comparison) plus an
LLM chatbot grounded in a local RAG vector store.

- `src/forecasting/static/chatbot_ui.html` — served at `GET /ui`.
- `src/forecasting/chatbot.py` — `AIOpsChatbot`, Groq/Llama integration.
- `src/forecasting/rag.py` — `VectorStore`, local docs + Confluence/Jira loaders.

## API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Model load status, LLM config, vector store readiness. |
| POST | `/predict` | Raw feature-vector inference — caller supplies exactly `FEATURE_COLUMNS`. |
| POST | `/forecast` | Live per-server forecast (`model`: `auto`/`arima`/`sarima`/`xgboost`). |
| GET | `/capacity-overview` | Fleet-wide rightsizing recommendations for a dataset. |
| GET | `/servers`, `/servers/{id}/history` | Server list / historical utilization for a dataset. |
| GET | `/models/metrics` | MAE comparison summary across trained models. |
| GET | `/datasets` | List registered datasets (synthetic + uploaded). |
| POST | `/datasets/upload` | Upload a CSV/XLSX; auto-trains + notifies Airflow. |
| POST | `/datasets/{id}/train` | Re-trigger training for an already-uploaded dataset. |
| GET | `/model-comparison` | Read the 7-tab offline benchmark for a dataset (`?dataset_id=`). |
| POST | `/datasets/{id}/model-comparison` | Run the offline benchmark in the background. |
| POST | `/storage/events` | MinIO bucket-notification webhook target — bearer-token authenticated, not meant to be called directly. |
| POST | `/chat`, `/llm/*` | Chatbot Q&A, capacity summaries, ticket drafts, root-cause, exec reports. |
| GET | `/ui` | The dashboard. |

## Running it

```bash
# 1. Infra: Postgres, Kafka/Zookeeper, Airflow webserver+scheduler, MinIO
docker compose up -d

# 2. MLflow tracking server (same Postgres backend as Airflow)
python -m mlflow server \
  --backend-store-uri postgresql+psycopg2://airflow:airflow@localhost:5433/mlflow \
  --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# 3. Train the baseline synthetic models + register/promote XGBoost
python -m src.forecasting.training

# 4. Start the API (loads .env, sets FORECAST_MODEL_URI)
python run_api.py
```

The dashboard is then at `http://localhost:8000/ui`, MLflow at
`http://localhost:5000`, Airflow at `http://localhost:8090`, and the MinIO console
at `http://localhost:9011` (`minioadmin`/`minioadmin`) — drop a `.csv` into the
`cpu-utilization-datasets` bucket there and it'll appear under `/datasets` within
moments, no upload button required.

Example `/predict` payload (must match `FEATURE_COLUMNS` exactly):

```json
{
  "server_id": "server-001",
  "horizon": 1,
  "features": {
    "cpu_utilization_lag_1": 34.0,
    "cpu_utilization_lag_3": 30.2,
    "cpu_utilization_lag_7": 25.8,
    "cpu_utilization_lag_14": 22.1,
    "cpu_utilization_lag_30": 20.0,
    "cpu_utilization_roll_mean_3": 33.2,
    "cpu_utilization_roll_std_3": 2.1,
    "cpu_utilization_roll_min_3": 31.0,
    "cpu_utilization_roll_max_3": 35.8,
    "cpu_utilization_roll_mean_7": 30.1,
    "cpu_utilization_roll_std_7": 3.4,
    "cpu_utilization_roll_min_7": 28.0,
    "cpu_utilization_roll_max_7": 33.5,
    "day_of_week": 2,
    "month": 7,
    "is_weekend": 0
  }
}
```

Most live use goes through `POST /forecast` instead (server_id + horizon_days +
dataset_id), which builds this feature vector for you from a server's history.

## Known state

### Fixed

- **Feature leakage** (`features.py`) — raw cpu/ram/disk/network + ratio features
  were dropped from the model input; they leaked the label, producing unrealistic
  near-zero error. Only lag/rolling/calendar features remain.
- **Hardcoded ARIMA/SARIMA order** (`models.py`) — order is now chosen per-series
  via ADF stationarity + ACF, not a single `(p,d,q)` for every server.
- **NaN leakage in model comparison** (`model_comparison.py`) — real uploaded
  datasets with missing readings produced NaN in the walk-forward/dataset-size
  studies, which then failed to even serialize as JSON. Now imputed the same way
  the main pipeline already does.
- **Champion promotion bug** (`training.py`) — `promote_best_xgboost_version()`
  compared MAE across every registered version regardless of dataset or
  feature-schema era, letting a stale pre-fix model silently become "champion" and
  break live serving with a feature-names mismatch. Now filtered to
  schema-matching baseline runs only.
- **Chart.js crash** (`chatbot_ui.html`) — charts were recreated on dataset switch
  without destroying the previous instance. The Model Comparison tab is now
  dataset-aware with a run/poll flow.
- **Duplicated file-parsing logic** (`api.py`) — the CSV/XLSX parsing in
  `/datasets/upload` is now a single shared `ForecastService.read_tabular_bytes()`,
  reused by the new MinIO webhook handler, so the two ingestion paths can't
  silently drift on what file types/parsing they accept.

### Still open

- **Dataset registry is in-memory only** (`forecast_service.py`) — an uploaded
  dataset disappears from the picker on every API restart, even though its CSV and
  past results are still on disk.
- **Airflow queue can wedge** (`dags/train_and_compare_models.py`) — the DAG's
  `max_active_runs: 1` means one orphaned "running" entry (e.g. from a crash
  mid-task) silently blocks every queued run behind it, with no alert.
- **GIL contention on long comparison runs** (`api.py`) — the in-process
  background thread for `POST .../model-comparison` shares the GIL with request
  handling; a long real-dataset run (~10+ min) can make other API calls sluggish.
  A subprocess or a real task queue (Celery/RQ) would isolate this.
- **Daily-cadence assumption** (`model_comparison.py`) — the 30-day test split
  assumes daily-cadence data; an hourly dataset gets ~700 points crammed under 30
  date labels, and ARIMA/SARIMA get scored on a much longer horizon than the chart
  implies.
- **MinIO's webhook target is a fixed host/port** (`docker-compose.yml`) —
  `MINIO_NOTIFY_WEBHOOK_ENDPOINT_datasets` hardcodes `host.docker.internal:8000`.
  If port 8000 is taken by something else on the host (as happened while building
  this), or the API runs on a different port, the webhook silently can't reach it -
  MinIO logs the delivery failure but there's no alert surfaced anywhere else.
- **No dead-letter/retry visibility for missed webhook deliveries** — if
  `/storage/events` is down when an object lands, that object is never retried or
  flagged; the dataset simply never appears. Re-uploading the same object (or a
  periodic reconciliation job) would be the only way to recover.

## Notes

- `mlruns/` stores MLflow experiment metadata pointers locally; the actual backend
  store is Postgres (see above) — sqlite-over-Docker-bind-mount was dropped after
  it hit `disk I/O error` from inside the Airflow containers.
- You can extend the model set with LightGBM, TCN, hyperparameter tuning, SHAP
  explainability, or confidence-interval generation.
