import os
import json
from pathlib import Path

from .rag import VectorStore, DEFAULT_VECTOR_STORE_PATH


def main(server: str = "App-101", top_k: int = 5):
    path = os.getenv("RAG_VECTOR_STORE_PATH", str(DEFAULT_VECTOR_STORE_PATH))
    print(f"Loading vector store from {path}")
    vs = VectorStore.load(path)
    query = f"Server: {server}"
    results = vs.search(query, top_k=top_k)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
