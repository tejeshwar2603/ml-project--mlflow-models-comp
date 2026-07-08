# AIOps Chatbot API

The forecasting chatbot combines ML predictions with Retrieval-Augmented Generation (RAG) to provide operational insights and automated recommendations.

## Running the Chatbot

```bash
python start_chatbot.py
```

The API will start on `http://127.0.0.1:8001` by default.

### Configuration

Set environment variables to customize behavior:

```bash
# LLM Provider (required for LLM responses)
llma-key=your_grok_api_key_here

# LLM API endpoint (defaults to Grok)
LLAMA_API_URL=https://api.x.ai/v1/chat/completions

# RAG Vector Store
RAG_VECTOR_STORE_PATH=artifacts/vector_store.pkl

# API Server
API_HOST=0.0.0.0
API_PORT=8001
```

## Endpoints

### Health Check

```bash
curl http://127.0.0.1:8001/health
```

**Response:**

```json
{
  "status": "ok",
  "model_uri": "models:/cpu_forecast/Production"
}
```

---

### `/chat` — General Q&A with Forecast + RAG

**POST** `http://127.0.0.1:8001/chat`

Answers operational questions using ML predictions and retrieved knowledge base.

**Request:**

```json
{
  "question": "Which servers will exceed 90% CPU next week?",
  "analysis_mode": "capacity_planning",
  "top_k": 5,
  "ml_output": {
    "server_id": "App-101",
    "horizon": 7,
    "prediction": 94
  }
}
```

**Response:**

```json
{
  "answer": "App-101 is expected to exceed 90% CPU...",
  "analysis_mode": "capacity_planning",
  "risk": {
    "level": "critical",
    "reason": "Forecasted CPU is at or above 90%."
  },
  "recommendation": "Treat as capacity risk: check active incidents...",
  "action_plan": [
    "Validate the forecast against current telemetry.",
    "Check retrieved Jira issues for similar incidents...",
    "..."
  ],
  "jira_ticket_draft": {
    "summary": "Predicted high CPU utilization",
    "description": "Server: App-101\nHorizon: 7 day(s)\n...",
    "labels": ["aiops", "capacity", "forecast"]
  },
  "sources": [
    {
      "source": "artifacts\\sarima\\predictions.csv",
      "score": 0.38,
      "metadata": { "server_id": "App-101", "kind": "prediction" }
    },
    "..."
  ]
}
```

---

### `/llm/capacity-summary` — Capacity Planning Summary

**POST** `http://127.0.0.1:8001/llm/capacity-summary`

Generates a capacity planning summary using the forecast and historical patterns.

**Request:**

```json
{
  "question": "Summarize capacity needs for the next 30 days.",
  "analysis_mode": "capacity_planning",
  "top_k": 5,
  "ml_output": {
    "server_id": "App-101",
    "horizon": 30,
    "prediction": 88
  }
}
```

**Response:** Same as `/chat` but optimized for capacity planning analysis.

---

### `/llm/draft-ticket` — Generate Jira Ticket

**POST** `http://127.0.0.1:8001/llm/draft-ticket`

Drafts a Jira ticket with summary, description, and suggested labels.

**Request:**

```json
{
  "question": "Draft a ticket for this high CPU forecast.",
  "top_k": 5,
  "ml_output": {
    "server_id": "DB-205",
    "horizon": 7,
    "prediction": 92
  }
}
```

**Response:**

```json
{
  "ticket_draft": {
    "summary": "Predicted high CPU utilization",
    "description": "Server: DB-205\nHorizon: 7 day(s)\nPredicted CPU: 92\nRisk: critical - Forecasted CPU is at or above 90%.\nSuggested action: Treat as capacity risk...",
    "labels": ["aiops", "capacity", "forecast"]
  },
  "risk_level": "critical",
  "recommendation": "Treat as capacity risk: check active incidents and planned changes...",
  "sources": [...]
}
```

---

### `/llm/executive-report` — Executive Summary

**POST** `http://127.0.0.1:8001/llm/executive-report`

Generates a business-friendly executive summary.

**Request:**

```json
{
  "question": "Summarize capacity risks for leadership.",
  "top_k": 5,
  "ml_output": {
    "server_id": "App-101",
    "horizon": 7,
    "prediction": 94
  }
}
```

**Response:** Same as `/chat` with `analysis_mode: executive_report`.

---

### `/llm/root-cause` — Root Cause Analysis

**POST** `http://127.0.0.1:8001/llm/root-cause`

Identifies likely root causes by comparing forecast patterns with historical incidents.

**Request:**

```json
{
  "question": "What is causing the CPU spike on App-101?",
  "top_k": 5,
  "ml_output": {
    "server_id": "App-101",
    "horizon": 7,
    "prediction": 94
  }
}
```

**Response:** Same as `/chat` with `analysis_mode: root_cause`.

---

## Analysis Modes

- **`general`** — Direct Q&A using context and forecast.
- **`capacity_planning`** — Explain capacity risk, affected services, and scaling options.
- **`incident_prevention`** — Identify incident risk and preventative mitigation.
- **`jira_ticket`** — Draft a Jira ticket for engineer review.
- **`executive_report`** — Summarize risk, business impact, and recommendations for leaders.
- **`root_cause`** — Compare patterns with incidents and runbooks to propose likely causes.
- **`change_impact`** — Assess whether a deployment/workload change may increase risk.

---

## Example cURL Commands

### Query with prediction included

```bash
curl -X POST http://127.0.0.1:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which servers will exceed 90% CPU next week?",
    "analysis_mode": "capacity_planning",
    "top_k": 5,
    "ml_output": {
      "server_id": "App-101",
      "horizon": 7,
      "prediction": 94
    }
  }'
```

### Draft a Jira ticket

```bash
curl -X POST http://127.0.0.1:8001/llm/draft-ticket \
  -H "Content-Type: application/json" \
  -d '{
    "ml_output": {
      "server_id": "DB-205",
      "horizon": 7,
      "prediction": 92
    }
  }'
```

### Ask for executive summary

```bash
curl -X POST http://127.0.0.1:8001/llm/executive-report \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize capacity risks for the business.",
    "ml_output": {
      "server_id": "App-101",
      "horizon": 30,
      "prediction": 88
    }
  }'
```

---

## Building the RAG Vector Store

The vector store indexes:

- **Local files**: `README.md`, `ARCHITECTURE.md` (configurable via `RAG_LOCAL_PATHS`)
- **Prediction CSVs**: all `artifacts/**/predictions.csv` files
- **Confluence** (optional): Atlassian wikis (if `CONFLUENCE_SPACE_KEY` and credentials set)
- **Jira** (optional): Issue history (if `JIRA_JQL` and credentials set)

Rebuild the vector store:

```bash
python -m src.forecasting.build_vector_store
```

---

## Generating Fine-Tune Data

Convert prediction CSVs to fine-tune examples for LLM training:

```bash
python -m src.forecasting.generate_finetune_data
```

Output: `artifacts/finetune_predictions.jsonl` (prompt/completion pairs).

---

## Architecture

```
Monitoring Systems (CPU, RAM, Disk, Network)
        ↓
Forecasting Model (XGBoost / TFT)
        ↓
ML Predictions (Next 7/30 Days)
        ↓
LLM + RAG
        ↓
Recommendations, Reports, Ticket Summaries, Action Plans
```

The LLM does not replace the forecasting model; it sits on top to explain predictions and automate operational workflows.

---

## Troubleshooting

**"LLM provider is not configured"**

- Ensure `llma-key` is set in `.env` or environment.
- Verify `LLAMA_API_URL` points to a working endpoint (defaults to Grok).

**"Chatbot vector store could not be loaded"**

- Run `python -m src.forecasting.build_vector_store` to build or rebuild the vector store.

**Port already in use**

- Set `API_PORT` to a different port (default 8001).

---

## Integration Scenarios

### Scenario 1: Capacity Planning Meeting

```bash
# Get a summary for leadership
curl -X POST http://127.0.0.1:8001/llm/executive-report \
  -d '{"question": "Summarize infrastructure capacity trends."}'
```

### Scenario 2: Automated Jira Ticket

```bash
# Draft a ticket from forecast
curl -X POST http://127.0.0.1:8001/llm/draft-ticket \
  -d '{"ml_output": {"server_id": "API-312", "prediction": 97, "horizon": 7}}'
```

### Scenario 3: Root Cause Investigation

```bash
# Identify likely causes
curl -X POST http://127.0.0.1:8001/llm/root-cause \
  -d '{"question": "Why is API-312 CPU spiking?", "ml_output": {"prediction": 97}}'
```

---

For more details, see `ARCHITECTURE.md` and the source code in `src/forecasting/`.
