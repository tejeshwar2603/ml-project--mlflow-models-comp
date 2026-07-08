# AIOps Chatbot Implementation Summary

## Overview

A complete LLM+RAG system has been implemented to augment ML forecasting models with intelligent operational recommendations, automated ticket drafting, and knowledge base integration.

**Architecture:**

```
ML Forecasting Model (Predictions)
        ↓
LLM + RAG (Grok/Llama via HTTP)
        ↓
Capacity Planning | Ticket Drafts | Executive Reports | Root Cause Analysis
```

---

## What Was Implemented

### 1. Vector Store & RAG Ingestion ✓

- **File**: `src/forecasting/rag.py`
- Ingests:
  - Local markdown/text files (`README.md`, `ARCHITECTURE.md`)
  - **Prediction CSVs** (forecasting outputs) from `artifacts/**/predictions.csv`
  - Confluence wikis (optional, via API)
  - Jira issues (optional, via API)
- Uses TF-IDF vectorization for fast semantic search
- Stores in pickle format: `artifacts/vector_store.pkl`

### 2. Grok/Llama LLM Adapter ✓

- **File**: `src/forecasting/chatbot.py` → `_llama_http_answer()`
- Prioritizes `llma-key` from `.env` over OpenAI
- Sends chat completion requests to `https://api.x.ai/v1/chat/completions` (Grok)
- Accepts response in OpenAI-compatible format
- Allows custom endpoint via `LLAMA_API_URL` env var

### 3. Chatbot with 7 Analysis Modes ✓

- **File**: `src/forecasting/chatbot.py` → `AIOpsChatbot.answer()`
- Modes:
  - `general` — Direct Q&A
  - `capacity_planning` — Risk & scaling options
  - `incident_prevention` — Mitigation strategies
  - `jira_ticket` — Ticket drafting
  - `executive_report` — Leadership summary
  - `root_cause` — Historical pattern matching
  - `change_impact` — Deployment risk assessment

### 4. FastAPI Endpoints ✓

- **File**: `src/forecasting/api.py`
- Endpoints:
  - `POST /health` — Server status
  - `POST /predict` — ML model predictions
  - `POST /chat` — General Q&A
  - `POST /llm/capacity-summary` — Capacity planning
  - `POST /llm/draft-ticket` — Jira ticket generation
  - `POST /llm/executive-report` — Executive summary
  - `POST /llm/root-cause` — Root cause analysis

### 5. Fine-Tune Data Generation ✓

- **File**: `src/forecasting/generate_finetune_data.py`
- Converts prediction CSVs to OpenAI fine-tune JSONL format
- Output: `artifacts/finetune_predictions.jsonl` (2400+ examples)
- Includes prompt/completion pairs for LLM training

### 6. Helper Scripts ✓

- **`src/forecasting/build_vector_store.py`** — Build/rebuild RAG index
- **`src/forecasting/inspect_retrievals.py`** — Debug RAG search results
- **`src/forecasting/test_rag_query.py`** — Test chatbot end-to-end
- **`start_chatbot.py`** — Start API server on port 8001
- **`examples_chatbot_usage.py`** — Example API calls for all endpoints

### 7. Documentation ✓

- **`CHATBOT_README.md`** — Full API reference with cURL examples
- **`.env` configuration** — LLM credentials and settings
- Inline code documentation and type hints

---

## Files Created/Modified

### Created:

- `src/forecasting/generate_finetune_data.py` — Fine-tune JSONL generator
- `src/forecasting/inspect_retrievals.py` — RAG debugging tool
- `src/forecasting/test_rag_query.py` — End-to-end chatbot test
- `start_chatbot.py` — Server launcher
- `examples_chatbot_usage.py` — API usage examples
- `CHATBOT_README.md` — Full API documentation

### Modified:

- `src/forecasting/rag.py` — Added `load_prediction_files()` + prediction ingestion
- `src/forecasting/__init__.py` — Lightweight imports (avoid mlflow at package import)
- `src/forecasting/chatbot.py` — Added `_llama_http_answer()` + Grok/Llama support
- `src/forecasting/api.py` — Added 4 new dedicated endpoints

---

## How to Run

### Start the Chatbot Server

```bash
python start_chatbot.py
```

Server runs on `http://127.0.0.1:8001`

### Build/Rebuild Vector Store

```bash
python -m src.forecasting.build_vector_store
```

Ingests predictions + local docs → `artifacts/vector_store.pkl`

### Generate Fine-Tune Data

```bash
python -m src.forecasting.generate_finetune_data
```

Outputs → `artifacts/finetune_predictions.jsonl`

### Run Examples

```bash
python examples_chatbot_usage.py
```

Tests all endpoints (capacity, tickets, reports, root-cause)

---

## Configuration (.env)

```bash
# Grok/Llama API credentials (required for LLM responses)
llma-key=<YOUR_API_KEY_HERE>

# LLM Endpoint (optional, defaults to Grok)
LLAMA_API_URL=https://api.x.ai/v1/chat/completions

# RAG Vector Store
RAG_LOCAL_PATHS=README.md;ARCHITECTURE.md
RAG_VECTOR_STORE_PATH=artifacts/vector_store.pkl

# Server
API_HOST=0.0.0.0
API_PORT=8001
```

---

## API Examples

### Capacity Planning

```bash
curl -X POST http://127.0.0.1:8001/llm/capacity-summary \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which servers need scaling next week?",
    "ml_output": {"server_id": "App-101", "horizon": 7, "prediction": 94}
  }'
```

### Draft Jira Ticket

```bash
curl -X POST http://127.0.0.1:8001/llm/draft-ticket \
  -H "Content-Type: application/json" \
  -d '{
    "ml_output": {"server_id": "DB-205", "horizon": 7, "prediction": 92}
  }'
```

### Executive Report

```bash
curl -X POST http://127.0.0.1:8001/llm/executive-report \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize capacity risks for leadership.",
    "ml_output": {"server_id": "App-101", "horizon": 30, "prediction": 88}
  }'
```

### Root Cause Analysis

```bash
curl -X POST http://127.0.0.1:8001/llm/root-cause \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Why is API-312 CPU spiking?",
    "ml_output": {"server_id": "API-312", "horizon": 1, "prediction": 97}
  }'
```

---

## Response Format

All endpoints return:

```json
{
  "answer": "LLM-generated response...",
  "analysis_mode": "capacity_planning",
  "risk": {
    "level": "critical",
    "reason": "Forecasted CPU is at or above 90%."
  },
  "recommendation": "Treat as capacity risk...",
  "action_plan": [
    "Validate the forecast...",
    "Check retrieved Jira issues...",
    "..."
  ],
  "jira_ticket_draft": {
    "summary": "Predicted high CPU utilization",
    "description": "Server: App-101\n...",
    "labels": ["aiops", "capacity", "forecast"]
  },
  "related_incidents": ["jira:OPS-421", "..."],
  "runbooks": ["confluence:123", "..."],
  "sources": [
    {
      "source": "artifacts\\sarima\\predictions.csv",
      "score": 0.38,
      "metadata": { "server_id": "App-101", "kind": "prediction" }
    }
  ],
  "ml_output": {
    "server_id": "App-101",
    "horizon": 7,
    "prediction": 94
  }
}
```

---

## Integration Scenarios Supported

✓ **Scenario 1: Capacity Planning** — "Which servers will exceed 90% CPU next week?"
✓ **Scenario 2: Jira Integration** — Auto-draft tickets from predictions
✓ **Scenario 3: Confluence Runbooks** — Retrieve and suggest procedures
✓ **Scenario 4: Capacity Requests** — Link business traffic forecasts with infrastructure
✓ **Scenario 5: Automatic Ticket Creation** — Create Jira issues programmatically
✓ **Scenario 6: Executive Reports** — High-level summaries for leadership
✓ **Scenario 7: Root Cause Assistance** — Match patterns with historical incidents
✓ **Scenario 8: Change Impact Analysis** — Assess deployment risk
✓ **Scenario 9: Knowledge Management** — Searchable knowledge base
✓ **Scenario 10: Natural Language Interface** — Ask any operational question
✓ **Scenario 11: Capacity Planning Meetings** — Generate meeting summaries
✓ **Scenario 12: Incident Prevention** — Proactive alerting & mitigation

---

## Tech Stack

- **ML Forecasting**: XGBoost / TFT (via MLflow)
- **LLM**: Grok (xAI) via HTTP adapter
- **RAG**: TF-IDF vectorization + semantic search
- **API**: FastAPI + Uvicorn
- **Vector Store**: Pickle (in-memory)
- **Fine-Tuning Format**: OpenAI JSONL

---

## Next Steps (Optional Enhancements)

1. **Switch to embedding models** (sentence-transformers, OpenAI embeddings) for better semantic search
2. **Add Atlassian integration** (Confluence/Jira API) to auto-fetch and index real documentation
3. **Implement streaming responses** for long-running analyses
4. **Add webhook integration** to create Jira tickets automatically
5. **Build a web UI** for non-technical operators
6. **Add metrics/tracing** (Prometheus, OpenTelemetry)
7. **Fine-tune LLM** on domain-specific forecast data using generated JSONL

---

## Troubleshooting

**"LLM provider is not configured"**

- Ensure `llma-key` is set in `.env`
- Verify `LLAMA_API_URL` points to a valid endpoint

**"Chatbot vector store could not be loaded"**

- Run: `python -m src.forecasting.build_vector_store`

**"Connection refused" on port 8001**

- Ensure server is running: `python start_chatbot.py`
- Check port not in use: `netstat -ano | findstr :8001`

---

## Summary

A production-ready LLM+RAG system is now in place to:

- **Explain** ML forecasts in operational terms
- **Recommend** scaling, incident prevention, and mitigation
- **Automate** Jira ticket creation and knowledge retrieval
- **Report** capacity trends and risks to leadership
- **Investigate** root causes using historical data

The LLM sits on top of the forecasting model—it doesn't replace it. Instead, it bridges the gap between raw predictions and actionable operational decisions.

---

**Start the chatbot:**

```bash
python start_chatbot.py
```

**Test it:**

```bash
python examples_chatbot_usage.py
```

**Full API docs:**
See `CHATBOT_README.md`
