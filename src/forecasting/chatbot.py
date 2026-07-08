import json
import os
from typing import Any

from .predictions import PredictionStore, format_prediction_answer
from .rag import DEFAULT_VECTOR_STORE_PATH, VectorStore


SYSTEM_PROMPT = """You are an Enterprise AIOps assistant.
Answer using the retrieved context, Jira/Confluence evidence, and the ML forecast.
If the context does not contain the answer, say what is missing instead of guessing.
Keep the answer operational and concise.
Return practical next steps, not only an explanation."""


ANALYSIS_MODES = {
    "general": "Answer the operator's question.",
    "capacity_planning": "Explain capacity risk, affected service impact, and scaling options.",
    "incident_prevention": "Identify likely incident risk and preventative mitigation.",
    "jira_ticket": "Draft a Jira ticket that an engineer can review and submit.",
    "executive_report": "Summarize risk, business impact, and recommended decision for leaders.",
    "root_cause": "Compare the metric pattern with retrieved incidents/runbooks and propose likely causes.",
    "change_impact": "Assess whether a planned deployment or workload change may increase risk.",
}


def _format_context(results: list[dict[str, Any]]) -> str:
    parts = []
    for idx, item in enumerate(results, start=1):
        parts.append(
            f"[{idx}] source={item['source']} score={item['score']:.3f}\n{item['text']}"
        )
    return "\n\n".join(parts)


def _format_insights(insights: dict[str, Any] | None) -> str:
    if not insights:
        return "none"
    if insights.get("type") == "capacity_scan":
        threshold = insights.get("threshold")
        horizon = insights.get("horizon_days")
        matches = insights.get("matches") or []
        top_servers = insights.get("top_servers") or []
        date_range = insights.get("date_range") or {}
        lines = [
            f"Capacity scan across all indexed servers (threshold={threshold}%, horizon={horizon} day(s), "
            f"forecast window {date_range.get('start', '?')} to {date_range.get('end', '?')}):",
        ]
        if matches:
            lines.append(f"Servers at/above {threshold}%:")
            for item in matches:
                lines.append(
                    f"- {item['server_id']}: peak {item['peak_prediction']}% on {item['peak_date']} (model={item['model']})"
                )
        else:
            lines.append(f"No servers are at/above {threshold}%.")
        if top_servers:
            lines.append("Highest predicted peaks across ALL indexed servers (for full ranking, not just threshold hits):")
            for item in top_servers:
                lines.append(f"- {item['server_id']}: {item['peak_prediction']}%")
        return "\n".join(lines)
    return json.dumps(insights, indent=2, default=str)


def _forecast_risk(ml_output: dict[str, Any] | None) -> dict[str, Any]:
    prediction = None
    if ml_output:
        prediction = ml_output.get("prediction")
    if prediction is None:
        return {"level": "unknown", "reason": "No ML prediction was supplied."}
    if prediction >= 90:
        return {"level": "critical", "reason": "Forecasted CPU is at or above 90%."}
    if prediction >= 80:
        return {"level": "high", "reason": "Forecasted CPU is at or above 80%."}
    if prediction >= 70:
        return {"level": "medium", "reason": "Forecasted CPU is elevated but below high-risk thresholds."}
    return {"level": "low", "reason": "Forecasted CPU is below common alert thresholds."}


def _build_operational_output(
    question: str,
    retrieved: list[dict[str, Any]],
    ml_output: dict[str, Any] | None,
    analysis_mode: str,
) -> dict[str, Any]:
    risk = _forecast_risk(ml_output)
    prediction = None if not ml_output else ml_output.get("prediction")
    server_id = None if not ml_output else ml_output.get("server_id")
    horizon = None if not ml_output else ml_output.get("horizon")
    evidence_sources = [item["source"] for item in retrieved]
    related_incidents = [item["source"] for item in retrieved if item["metadata"].get("kind") == "jira"]
    runbooks = [item["source"] for item in retrieved if item["metadata"].get("kind") == "confluence"]

    recommendation = "Review retrieved runbooks, related Jira incidents, and current utilization before taking action."
    if prediction is not None and prediction >= 90:
        recommendation = "Treat as capacity risk: check active incidents and planned changes, then scale or schedule mitigation."
    elif prediction is not None and prediction >= 80:
        recommendation = "Monitor closely and validate whether upcoming jobs, releases, or traffic increases explain the trend."

    action_plan = [
        "Validate the forecast against current CPU, RAM, disk, and network telemetry.",
        "Check retrieved Jira issues for similar incidents and previous fixes.",
        "Open the relevant Confluence runbook before making production changes.",
    ]
    if risk["level"] in {"critical", "high"}:
        action_plan.append("Prepare a capacity or scaling change for engineer approval.")

    ticket_summary = "Predicted high CPU utilization"
    ticket_description = (
        f"Server: {server_id or 'unknown'}\n"
        f"Horizon: {horizon or 'unknown'} day(s)\n"
        f"Predicted CPU: {prediction if prediction is not None else 'unknown'}\n"
        f"Risk: {risk['level']} - {risk['reason']}\n"
        f"Question: {question}\n"
        f"Evidence sources: {', '.join(evidence_sources) if evidence_sources else 'none'}\n"
        f"Suggested action: {recommendation}"
    )

    executive_summary = (
        f"Risk level is {risk['level']}. "
        f"{risk['reason']} "
        f"Recommended action: {recommendation}"
    )

    return {
        "analysis_mode": analysis_mode,
        "risk": risk,
        "recommendation": recommendation,
        "action_plan": action_plan,
        "related_incidents": related_incidents,
        "runbooks": runbooks,
        "jira_ticket_draft": {
            "summary": ticket_summary,
            "description": ticket_description,
            "labels": ["aiops", "capacity", "forecast"],
        },
        "executive_summary": executive_summary,
    }


DEFAULT_GROK_MODEL = "grok-4.3"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


def _grok_api_key() -> str | None:
    return (
        os.getenv("llma-key")
        or os.getenv("LLMA_KEY")
        or os.getenv("GROK_API_KEY")
        or os.getenv("XAI_API_KEY")
    )


def _llm_configured() -> bool:
    return bool(_grok_api_key() or os.getenv("OPENAI_API_KEY"))


def _fallback_prefix(llm_error: str | None = None) -> str:
    if llm_error:
        return f"LLM call failed ({llm_error}), so this answer uses indexed ML/RAG context instead.\n\n"
    if not _llm_configured():
        return "LLM provider is not configured, so this answer uses indexed ML/RAG context instead.\n\n"
    return ""


def _fallback_answer(
    question: str,
    retrieved: list[dict[str, Any]],
    ml_output: dict[str, Any] | None,
    analysis_mode: str,
    insights: dict[str, Any] | None = None,
    llm_error: str | None = None,
) -> str:
    prediction_answer = format_prediction_answer(question, ml_output, insights, analysis_mode)
    if prediction_answer:
        prefix = _fallback_prefix(llm_error)
        if prefix:
            prediction_answer = prefix + prediction_answer
        if retrieved:
            sources = ", ".join(item["source"] for item in retrieved[:3])
            prediction_answer += f"\n\nSupporting RAG sources: {sources}"
        return prediction_answer

    context = _format_context(retrieved)
    forecast = json.dumps(ml_output or {}, indent=2)
    if not retrieved and not ml_output and not insights:
        if llm_error:
            return f"LLM call failed ({llm_error}) and there is not enough indexed context or ML output to answer this yet."
        if not _llm_configured():
            return "I do not have enough indexed context or ML output to answer this yet."
        return "I do not have enough indexed context or ML output to answer this yet."
    mode_instruction = ANALYSIS_MODES.get(analysis_mode, ANALYSIS_MODES["general"])
    return (
        _fallback_prefix(llm_error)
        + f"Analysis mode: {analysis_mode} - {mode_instruction}\n\n"
        f"Question: {question}\n\n"
        f"ML forecast context:\n{forecast}\n\n"
        f"Top retrieved evidence:\n{context}"
    )


def _llama_http_answer(
    question: str,
    retrieved: list[dict[str, Any]],
    ml_output: dict[str, Any] | None,
    analysis_mode: str,
    llama_key: str,
    insights: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Call a Grok/Llama-compatible HTTP chat API using the provided key."""
    import urllib.request
    import urllib.error

    llama_url = os.getenv("LLAMA_API_URL", "https://api.x.ai/v1/chat/completions")
    model = os.getenv("AIOPS_LLM_MODEL", DEFAULT_GROK_MODEL)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Analysis mode: {analysis_mode}\n"
                f"Mode instruction: {ANALYSIS_MODES.get(analysis_mode, ANALYSIS_MODES['general'])}\n\n"
                f"Question:\n{question}\n\n"
                f"ML forecast output (single top match, if any):\n{json.dumps(ml_output or {}, indent=2)}\n\n"
                f"Fleet-wide ML insights (use this for questions about multiple/all servers):\n{_format_insights(insights)}\n\n"
                f"Retrieved context:\n{_format_context(retrieved)}"
            ),
        },
    ]
    
    payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.1}
    if model.startswith("grok-"):
        payload["reasoning_effort"] = os.getenv("AIOPS_LLM_REASONING_EFFORT", "none")
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {llama_key}",
        "User-Agent": "Mozilla/5.0 (compatible; AIOpsChatbot/1.0)",
    }
    
    try:
        req = urllib.request.Request(llama_url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        
        choices = body.get("choices") or []
        if choices:
            first = choices[0]
            if isinstance(first.get("message"), dict):
                content = first["message"].get("content")
                if content:
                    return content, None
            if first.get("text"):
                return first.get("text"), None
        
        if isinstance(body.get("text"), str):
            return body.get("text"), None
        if isinstance(body.get("response"), str):
            return body.get("response"), None
        return None, "Unexpected LLM response format"
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
            print(f"LLM HTTP error {e.code}: {err_body}")
            return None, f"HTTP {e.code}: {err_body}"
        except Exception:
            print(f"LLM HTTP error: {e}")
            return None, f"HTTP {e.code}"
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None, str(e)
    
    return None, "Empty LLM response"


def _openai_answer(
    question: str,
    retrieved: list[dict[str, Any]],
    ml_output: dict[str, Any] | None,
    analysis_mode: str,
    insights: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    llama_key = _grok_api_key()
    if llama_key:
        return _llama_http_answer(question, retrieved, ml_output, analysis_mode, llama_key, insights=insights)

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return None, None
    try:
        from openai import OpenAI
    except Exception as exc:
        return None, str(exc)

    model = os.getenv("AIOPS_LLM_MODEL", DEFAULT_OPENAI_MODEL)
    client = OpenAI(api_key=openai_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Analysis mode: {analysis_mode}\n"
                f"Mode instruction: {ANALYSIS_MODES.get(analysis_mode, ANALYSIS_MODES['general'])}\n\n"
                f"Question:\n{question}\n\n"
                f"ML forecast output (single top match, if any):\n{json.dumps(ml_output or {}, indent=2)}\n\n"
                f"Fleet-wide ML insights (use this for questions about multiple/all servers):\n{_format_insights(insights)}\n\n"
                f"Retrieved context:\n{_format_context(retrieved)}"
            ),
        },
    ]
    try:
        response = client.chat.completions.create(model=model, messages=messages, temperature=0.1)
        return response.choices[0].message.content or "", None
    except Exception as exc:
        return None, str(exc)


class AIOpsChatbot:
    def __init__(
        self,
        vector_store_path: str = str(DEFAULT_VECTOR_STORE_PATH),
        artifacts_dir: str | None = None,
    ) -> None:
        self.vector_store_path = vector_store_path
        self.vector_store = VectorStore.load(vector_store_path)
        self.prediction_store = PredictionStore(artifacts_dir or "artifacts")

    def answer(
        self,
        question: str,
        ml_output: dict[str, Any] | None = None,
        top_k: int = 5,
        analysis_mode: str = "general",
    ) -> dict[str, Any]:
        if analysis_mode not in ANALYSIS_MODES:
            raise ValueError(f"Unsupported analysis_mode: {analysis_mode}")
        resolved = self.prediction_store.resolve_context(question, ml_output)
        ml_output = resolved.get("ml_output")
        insights = resolved.get("insights")
        if ml_output is None and insights and insights.get("matches"):
            top_match = insights["matches"][0]
            ml_output = {
                "server_id": top_match["server_id"],
                "prediction": top_match["peak_prediction"],
                "horizon": insights.get("horizon_days", 7),
                "peak_date": top_match.get("peak_date"),
                "model": top_match.get("model"),
            }
        retrieved = self.vector_store.search(question, top_k=top_k)
        structured = _build_operational_output(question, retrieved, ml_output, analysis_mode)
        answer, llm_error = _openai_answer(question, retrieved, ml_output, analysis_mode, insights=insights)
        if answer is None:
            answer = _fallback_answer(
                question,
                retrieved,
                ml_output,
                analysis_mode,
                insights,
                llm_error=llm_error,
            )
        return {
            "answer": answer,
            **structured,
            "llm_configured": _llm_configured(),
            "llm_error": llm_error,
            "sources": [
                {
                    "source": item["source"],
                    "score": item["score"],
                    "metadata": item["metadata"],
                }
                for item in retrieved
            ],
            "ml_output": ml_output,
            "insights": insights,
        }
