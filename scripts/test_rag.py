import os
import sys
import json
from pathlib import Path
import importlib.util

# Use local docs for the vector store
os.environ.setdefault("RAG_LOCAL_PATHS", "README.md;ARCHITECTURE.md")

RAG_PATH = Path("src/forecasting/rag.py")
if not RAG_PATH.exists():
    print(f"rag.py not found at {RAG_PATH}. Aborting.")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("rag_local", str(RAG_PATH))
rag = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(rag)
except Exception as e:
    print("Failed to load rag module:", e)
    sys.exit(1)

try:
    print("Loading local documents and building vector store in-memory...")
    local_paths = [p for p in os.getenv("RAG_LOCAL_PATHS", "").split(os.pathsep) if p]
    if not local_paths:
        print("No RAG_LOCAL_PATHS set; defaulting to README.md and ARCHITECTURE.md")
        local_paths = ["README.md", "ARCHITECTURE.md"]
    docs = rag.load_local_documents(local_paths)
    if not docs:
        print("No local documents found to build vector store. Check RAG_LOCAL_PATHS.")
        sys.exit(1)
    vs = rag.VectorStore()
    vs.build(docs)
    print("Built vector store in-memory. Running sample query: 'high cpu utilization'\n")
    results = vs.search("high cpu utilization", top_k=5)
    print(json.dumps(results, indent=2))
except Exception as e:
    print("Error building or querying vector store:", e)
    sys.exit(1)
