import os
import json
from pathlib import Path
from glob import glob
import pandas as pd


def _classify_and_recommend(predicted, current=None):
    try:
        p = float(predicted)
    except Exception:
        return "unknown", "Insufficient prediction data to provide a recommendation."
    if p >= 90:
        return (
            "critical",
            "Forecasted CPU is >=90%: prepare immediate capacity changes or mitigation (scale replicas or add nodes).",
        )
    if p >= 80:
        return (
            "high",
            "Forecasted CPU is >=80%: monitor closely, validate scheduled jobs/releases, prepare scaling plan.",
        )
    if p >= 70:
        return (
            "medium",
            "CPU is elevated (>=70%): validate traffic patterns and jobs; consider preemptive scaling if trend continues.",
        )
    return ("low", "No immediate action required; continue monitoring.")


def find_prediction_files(env_var: str = "RAG_PREDICTION_PATHS") -> list[str]:
    paths = [p for p in os.getenv(env_var, "").split(os.pathsep) if p]
    if not paths:
        repo_artifacts = Path("artifacts")
        if repo_artifacts.exists():
            paths.extend([str(p) for p in repo_artifacts.rglob("predictions*.csv")])
    # also accept glob patterns
    resolved = []
    for p in paths:
        resolved.extend(glob(p))
    return list(dict.fromkeys(resolved))


def main(output: str = "artifacts/finetune_predictions.jsonl") -> None:
    files = find_prediction_files()
    if not files:
        print("No prediction files found to generate fine-tune data.")
        return
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for fp in files:
            try:
                df = pd.read_csv(fp)
            except Exception:
                continue
            for _, row in df.iterrows():
                server = str(row.get("server_id") or row.get("server") or row.get("host") or "unknown")
                date = str(row.get("date") or row.get("timestamp") or "")
                prediction = row.get("prediction") if "prediction" in row else row.get("predicted") if "predicted" in row else None
                current = row.get("current_cpu") if "current_cpu" in row else row.get("current") if "current" in row else None
                horizon = row.get("horizon") if "horizon" in row else None
                level, recommendation = _classify_and_recommend(prediction, current)
                prompt_parts = [f"Server: {server}"]
                if date:
                    prompt_parts.append(f"Date: {date}")
                if current is not None:
                    prompt_parts.append(f"Current CPU: {current}")
                if prediction is not None:
                    prompt_parts.append(f"Predicted CPU: {prediction}")
                if horizon is not None:
                    prompt_parts.append(f"Horizon: {horizon}")
                prompt = " | ".join(prompt_parts) + "\n\nQuestion: Based on this forecast, summarize the risk level and recommended next steps."
                completion = f"Risk: {level}. Recommendation: {recommendation}"
                # OpenAI fine-tune JSONL expects newline-terminated completion
                record = {"prompt": prompt, "completion": completion + "\n"}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print(f"Wrote {count} fine-tune examples to {out_path}")


if __name__ == "__main__":
    main()
