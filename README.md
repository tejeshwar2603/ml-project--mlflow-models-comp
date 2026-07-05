# Enterprise AIOps Copilot Forecasting

Production-ready CPU forecasting pipeline with classical, machine learning, and deep learning models.

## What is included

- Data ingestion and validation
- Feature engineering with lag, rolling, and calendar features
- Time-based train/validation/test split and rolling window evaluation
- Model wrappers for ARIMA, SARIMA, XGBoost, GRU, LSTM, TFT
- MLflow experiment tracking with metrics, parameters, artifacts, and model registry
- FastAPI prediction service using the registered production model
- RAG chatbot endpoint that can answer from local docs, Confluence pages, Jira issues, and ML forecast output

## Install

```bash
pip install -r requirements.txt
```

## Run training

```bash
python -m src.forecasting.training
```

## Run API

```bash
python -m src.forecasting.app
```

or

```bash
python run_api.py
```

## Build RAG vector store

LangChain is not required. This project uses a small local vector store based on scikit-learn TF-IDF embeddings. Set any of these sources, then build the store:

```bash
set RAG_LOCAL_PATHS=README.md;ARCHITECTURE.md
set ATLASSIAN_BASE_URL=https://your-company.atlassian.net
set ATLASSIAN_EMAIL=you@company.com
set ATLASSIAN_API_TOKEN=your_api_token
set CONFLUENCE_SPACE_KEY=AIOPS
set JIRA_JQL=project = AIOPS ORDER BY updated DESC
python -m src.forecasting.build_vector_store
```

Then start the API and call `POST /chat`. If `OPENAI_API_KEY` is set, the chatbot uses the OpenAI-compatible model configured by `AIOPS_LLM_MODEL`; otherwise it returns a retrieved-context answer with sources.

Example chat request:

```json
{
  "question": "Which servers are likely to exceed 90% CPU next week and what runbooks should we follow?",
  "analysis_mode": "capacity_planning",
  "include_prediction": true,
  "prediction_request": {
    "server_id": "App-101",
    "horizon": 7,
    "features": {
      "cpu_utilization": 72,
      "ram_utilization": 68,
      "disk_utilization": 55,
      "network_utilization": 32,
      "cpu_utilization_lag_1": 70,
      "cpu_utilization_lag_3": 68,
      "cpu_utilization_lag_7": 63,
      "cpu_utilization_lag_14": 59,
      "cpu_utilization_lag_30": 55,
      "cpu_utilization_roll_mean_3": 69,
      "cpu_utilization_roll_std_3": 2,
      "cpu_utilization_roll_min_3": 67,
      "cpu_utilization_roll_max_3": 71,
      "cpu_utilization_roll_mean_7": 65,
      "cpu_utilization_roll_std_7": 4,
      "cpu_utilization_roll_min_7": 60,
      "cpu_utilization_roll_max_7": 70,
      "cpu_ram_ratio": 1.06,
      "cpu_disk_ratio": 1.31,
      "day_of_week": 1,
      "month": 7,
      "is_weekend": false
    }
  }
}
```

The `/chat` response includes:

- `answer`: narrative guidance for the operator
- `risk`: forecast risk assessment
- `recommendation`: practical next steps
- `action_plan`: checklist items
- `related_incidents`: any Jira issues found in the retrieved context
- `runbooks`: any Confluence runbooks found in the retrieved context
- `jira_ticket_draft`: a draft ticket summary and description
- `executive_summary`: a concise leader-facing summary

Supported `analysis_mode` values:

- `general`
- `capacity_planning`
- `incident_prevention`
- `jira_ticket`
- `executive_report`
- `root_cause`
- `change_impact`

The chatbot response includes the narrative answer, retrieved sources, ML output, risk level, recommendation, action plan, Jira ticket draft, and executive summary.

## Notes

- Training uses synthetic sample data by default.
- MLflow artifacts are stored in `mlruns/` under the project directory.
- The REST API exposes a `/predict` endpoint for daily horizon predictions.
- The `/chat` endpoint retrieves relevant chunks from the vector store and can include `/predict` output as LLM context.
- The `/chat` endpoint retrieves relevant chunks from the vector store and can include `/predict` output as LLM context.
- See `ARCHITECTURE.md` for full workflow, architecture details, and step-by-step run instructions.

Optional: use `.env.example`

- Copy `.env.example` to `.env` and fill in your credentials.
- `.env` is ignored by git (see `.gitignore`) so you won't accidentally commit secrets.

Example:

```bash
copy .env.example .env   # Windows cmd
cp .env.example .env     # PowerShell / Unix
# then edit .env and run
python run_api.py
```
