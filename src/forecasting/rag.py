import json
import os
import pickle
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_VECTOR_STORE_PATH = Path("artifacts") / "vector_store.pkl"


@dataclass
class Document:
    text: str
    source: str
    metadata: dict[str, Any]


def _clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


class VectorStore:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self.documents: list[Document] = []
        self.matrix = None

    def build(self, documents: list[Document]) -> "VectorStore":
        self.documents = [doc for doc in documents if doc.text.strip()]
        texts = [doc.text for doc in self.documents]
        if not texts:
            raise ValueError("No documents were provided for the vector store.")
        self.matrix = self.vectorizer.fit_transform(texts)
        return self

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.matrix is None or not self.documents:
            raise ValueError("Vector store is empty. Build or load it first.")
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        top_indexes = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indexes:
            if scores[idx] <= 0:
                continue
            doc = self.documents[int(idx)]
            results.append(
                {
                    "text": doc.text,
                    "source": doc.source,
                    "metadata": doc.metadata,
                    "score": float(scores[idx]),
                }
            )
        return results

    def save(self, path: str | Path = DEFAULT_VECTOR_STORE_PATH) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        return str(path)

    @staticmethod
    def load(path: str | Path = DEFAULT_VECTOR_STORE_PATH) -> "VectorStore":
        with Path(path).open("rb") as f:
            return pickle.load(f)


def _basic_auth_headers(email: str, token: str) -> dict[str, str]:
    import base64

    raw = f"{email}:{token}".encode("utf-8")
    return {"Authorization": "Basic " + base64.b64encode(raw).decode("ascii")}


def _get_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_local_documents(paths: list[str]) -> list[Document]:
    docs: list[Document] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        for file_path in ([path] if path.is_file() else path.rglob("*")):
            if file_path.suffix.lower() not in {".txt", ".md", ".json", ".csv"}:
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            for i, chunk in enumerate(chunk_text(text)):
                docs.append(Document(chunk, str(file_path), {"chunk": i, "kind": "local"}))
    return docs


def load_prediction_files(paths: list[str]) -> list[Document]:
    """Load prediction CSV files and produce small textual documents for RAG.

    Expected CSV columns (best-effort): server, server_id, date, prediction, current_cpu, horizon
    """
    docs: list[Document] = []
    for raw_path in paths:
        # expand globs and directories
        matches = []
        p = Path(raw_path)
        if p.is_dir():
            matches.extend([str(x) for x in p.rglob("predictions*.csv")])
        else:
            matches.extend(glob(raw_path))
        for file_path in matches:
            fp = Path(file_path)
            if not fp.exists():
                continue
            try:
                df = pd.read_csv(fp)
            except Exception:
                # skip unreadable files
                continue
            for _, row in df.iterrows():
                server = str(row.get("server_id") or row.get("server") or row.get("host") or "unknown")
                date = str(row.get("date") or row.get("timestamp") or "")
                prediction = (
                    row.get("prediction")
                    if "prediction" in row
                    else row.get("predicted_cpu_utilization")
                    if "predicted_cpu_utilization" in row
                    else row.get("predicted")
                    if "predicted" in row
                    else None
                )
                current = row.get("current_cpu") if "current_cpu" in row else row.get("current") if "current" in row else None
                horizon = row.get("horizon") if "horizon" in row else None
                text_parts = [f"Server: {server}"]
                if date:
                    text_parts.append(f"Date: {date}")
                if current is not None:
                    text_parts.append(f"Current CPU: {current}")
                if prediction is not None:
                    text_parts.append(f"Predicted CPU: {prediction}")
                if horizon is not None:
                    text_parts.append(f"Horizon: {horizon}")
                text = " | ".join(text_parts)
                for i, chunk in enumerate(chunk_text(text)):
                    docs.append(
                        Document(chunk, str(fp), {"server_id": server, "date": date, "chunk": i, "kind": "prediction"})
                    )
    return docs


def load_confluence_pages(
    base_url: str,
    email: str,
    api_token: str,
    space_key: str,
    limit: int = 25,
) -> list[Document]:
    headers = {"Accept": "application/json", **_basic_auth_headers(email, api_token)}
    query = urllib.parse.urlencode(
        {
            "spaceKey": space_key,
            "type": "page",
            "expand": "body.storage,version",
            "limit": limit,
        }
    )
    url = f"{base_url.rstrip('/')}/wiki/rest/api/content?{query}"
    payload = _get_json(url, headers)
    docs: list[Document] = []
    for page in payload.get("results", []):
        title = page.get("title", "Untitled Confluence page")
        body = page.get("body", {}).get("storage", {}).get("value", "")
        source = f"confluence:{page.get('id')}"
        for i, chunk in enumerate(chunk_text(f"{title}\n{body}")):
            docs.append(Document(chunk, source, {"title": title, "chunk": i, "kind": "confluence"}))
    return docs


def load_jira_issues(
    base_url: str,
    email: str,
    api_token: str,
    jql: str,
    limit: int = 50,
) -> list[Document]:
    headers = {"Accept": "application/json", **_basic_auth_headers(email, api_token)}
    query = urllib.parse.urlencode(
        {
            "jql": jql,
            "maxResults": limit,
            "fields": "summary,description,status,priority,issuetype",
        }
    )
    url = f"{base_url.rstrip('/')}/rest/api/3/search?{query}"
    payload = _get_json(url, headers)
    docs: list[Document] = []
    for issue in payload.get("issues", []):
        fields = issue.get("fields", {})
        key = issue.get("key", "UNKNOWN")
        text = json.dumps(fields.get("description", ""), ensure_ascii=False)
        title = fields.get("summary", "")
        status = fields.get("status", {}).get("name", "")
        source = f"jira:{key}"
        issue_text = f"{key} {title}\nStatus: {status}\n{text}"
        for i, chunk in enumerate(chunk_text(issue_text)):
            docs.append(Document(chunk, source, {"key": key, "summary": title, "chunk": i, "kind": "jira"}))
    return docs


def build_vector_store_from_environment(
    path: str | Path = DEFAULT_VECTOR_STORE_PATH,
) -> str:
    docs: list[Document] = []
    local_paths = [p for p in os.getenv("RAG_LOCAL_PATHS", "").split(os.pathsep) if p]
    docs.extend(load_local_documents(local_paths))

    # Ingest prediction CSVs into the RAG vector store
    pred_paths = [p for p in os.getenv("RAG_PREDICTION_PATHS", "").split(os.pathsep) if p]
    # default: search common artifacts path for predictions.csv files
    if not pred_paths:
        # try to find artifacts/**/predictions.csv relative to repo root
        repo_artifacts = Path("artifacts")
        if repo_artifacts.exists():
            pred_paths.extend([str(p) for p in repo_artifacts.rglob("predictions*.csv")])
    if pred_paths:
        docs.extend(load_prediction_files(pred_paths))

    confluence_space = os.getenv("CONFLUENCE_SPACE_KEY")
    jira_jql = os.getenv("JIRA_JQL")
    base_url = os.getenv("ATLASSIAN_BASE_URL")
    email = os.getenv("ATLASSIAN_EMAIL")
    token = os.getenv("ATLASSIAN_API_TOKEN")
    if base_url and email and token and confluence_space:
        docs.extend(load_confluence_pages(base_url, email, token, confluence_space))
    if base_url and email and token and jira_jql:
        docs.extend(load_jira_issues(base_url, email, token, jira_jql))

    return VectorStore().build(docs).save(path)

