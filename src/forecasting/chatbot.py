import json
import os
from typing import Any

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


def _fallback_answer(
    question: str,
    retrieved: list[dict[str, Any]],
    ml_output: dict[str, Any] | None,
    analysis_mode: str,
) -> str:
    context = _format_context(retrieved)
    forecast = json.dumps(ml_output or {}, indent=2)
    if not retrieved and not ml_output:
        return "I do not have enough indexed context or ML output to answer this yet."
    mode_instruction = ANALYSIS_MODES.get(analysis_mode, ANALYSIS_MODES["general"])
    return (
        "LLM provider is not configured, so this is a retrieved-context answer.\n\n"
        f"Analysis mode: {analysis_mode} - {mode_instruction}\n\n"
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
) -> str | None:
    """Call a Grok/Llama-compatible HTTP chat API using the provided key."""
    import urllib.request
    import urllib.error
    
    # Default to Grok's API endpoint (xAI), but allow override
    llama_url = os.getenv("LLAMA_API_URL", "https://api.x.ai/v1/chat/completions")
    model = os.getenv("AIOPS_LLM_MODEL", "grok-beta")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Analysis mode: {analysis_mode}\n"
                f"Mode instruction: {ANALYSIS_MODES.get(analysis_mode, ANALYSIS_MODES['general'])}\n\n"
                f"Question:\n{question}\n\n"
                f"ML forecast output:\n{json.dumps(ml_output or {}, indent=2)}\n\n"
                f"Retrieved context:\n{_format_context(retrieved)}"
            ),
        },
    ]
    
    payload = {"model": model, "messages": messages, "temperature": 0.1}
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {llama_key}",
    }
    
    try:
        req = urllib.request.Request(llama_url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        
        # Try OpenAI-like response: choices[0].message.content
        choices = body.get("choices") or []
        if choices:
            first = choices[0]
            if isinstance(first.get("message"), dict):
                content = first["message"].get("content")
                if content:
                    return content
            # Some providers return text directly
            if first.get("text"):
                return first.get("text")
        
        # As a last resort, try top-level 'text' or 'response'
        if isinstance(body.get("text"), str):
            return body.get("text")
        if isinstance(body.get("response"), str):
            return body.get("response")
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
            print(f"LLM HTTP error {e.code}: {err_body}")
        except Exception:
            print(f"LLM HTTP error: {e}")
        return None
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None
    
    return None


def _openai_answer(
    question: str,
    retrieved: list[dict[str, Any]],
    ml_output: dict[str, Any] | None,
    analysis_mode: str,
) -> str | None:
    # Prioritize Grok/Llama key over OpenAI
    llama_key = os.getenv("llma-key") or os.getenv("LLMA_KEY")
    if llama_key:
        # Use generic HTTP adapter for Grok/Llama
        return _llama_http_answer(question, retrieved, ml_output, analysis_mode, llama_key)
    
    # Fallback to OpenAI if available
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None

    model = os.getenv("AIOPS_LLM_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=openai_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Analysis mode: {analysis_mode}\n"
                f"Mode instruction: {ANALYSIS_MODES.get(analysis_mode, ANALYSIS_MODES['general'])}\n\n"
                f"Question:\n{question}\n\n"
                f"ML forecast output:\n{json.dumps(ml_output or {}, indent=2)}\n\n"
                f"Retrieved context:\n{_format_context(retrieved)}"
            ),
        },
    ]
    response = client.chat.completions.create(model=model, messages=messages, temperature=0.1)
    return response.choices[0].message.content or ""


class AIOpsChatbot:
    def __init__(self, vector_store_path: str = str(DEFAULT_VECTOR_STORE_PATH)) -> None:
        self.vector_store_path = vector_store_path
        self.vector_store = VectorStore.load(vector_store_path)

    def answer(
        self,
        question: str,
        ml_output: dict[str, Any] | None = None,
        top_k: int = 5,
        analysis_mode: str = "general",
    ) -> dict[str, Any]:
        if analysis_mode not in ANALYSIS_MODES:
            raise ValueError(f"Unsupported analysis_mode: {analysis_mode}")
        retrieved = self.vector_store.search(question, top_k=top_k)
        structured = _build_operational_output(question, retrieved, ml_output, analysis_mode)
        answer = _openai_answer(question, retrieved, ml_output, analysis_mode)
        if answer is None:
            answer = _fallback_answer(question, retrieved, ml_output, analysis_mode)
        return {
            "answer": answer,
            **structured,
            "sources": [
                {
                    "source": item["source"],
                    "score": item["score"],
                    "metadata": item["metadata"],
                }
                for item in retrieved
            ],
            "ml_output": ml_output,
        }
