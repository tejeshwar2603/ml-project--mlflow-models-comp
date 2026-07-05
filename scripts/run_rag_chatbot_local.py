import os
import sys
import json
import importlib.util
from dotenv import load_dotenv
import types
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAG_PATH = BASE / 'src' / 'forecasting' / 'rag.py'
CHATBOT_PATH = BASE / 'src' / 'forecasting' / 'chatbot.py'

load_dotenv(BASE / '.env')

def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    # Create minimal package modules so relative imports inside modules work
    src_mod = types.ModuleType('src')
    src_fore_mod = types.ModuleType('src.forecasting')
    src_mod.__path__ = [str(BASE / 'src')]
    src_fore_mod.__path__ = [str(BASE / 'src' / 'forecasting')]
    sys.modules['src'] = src_mod
    sys.modules['src.forecasting'] = src_fore_mod

    rag = load_module_from_path('src.forecasting.rag', RAG_PATH)
    sys.modules['src.forecasting.rag'] = rag
    chatbot = load_module_from_path('src.forecasting.chatbot', CHATBOT_PATH)
    sys.modules['src.forecasting.chatbot'] = chatbot
except Exception as e:
    print('Failed to load modules:', e)
    sys.exit(1)

# Build vector store in-memory from local docs
local_paths = [p for p in os.getenv('RAG_LOCAL_PATHS', '').split(os.pathsep) if p]
if not local_paths:
    local_paths = ['README.md', 'ARCHITECTURE.md']
print('Using local paths:', local_paths)

docs = rag.load_local_documents(local_paths)
if not docs:
    print('No documents found for vector store. Ensure README.md or ARCHITECTURE.md exist.')
    sys.exit(1)

vs = rag.VectorStore()
try:
    vs.build(docs)
except Exception as e:
    print('Error building vector store:', e)
    sys.exit(1)

query = os.getenv('RAG_TEST_QUERY', 'Which servers are likely to exceed 90% CPU next week?')
print('\nRunning query:', query)
results = vs.search(query, top_k=5)
print('\nRetrieved documents:')
print(json.dumps(results, indent=2)[:4000])

# Build ML output sample (no MLflow needed)
ml_output = {
    'server_id': 'App-101',
    'horizon': 7,
    'prediction': 94.0,
}

# Build structured operational output
structured = chatbot._build_operational_output(query, results, ml_output, analysis_mode='capacity_planning')
print('\nStructured output:')
print(json.dumps(structured, indent=2))

# Try LLM call path if key present
if os.getenv('OPENAI_API_KEY'):
    print('\nOPENAI_API_KEY found; attempting LLM call (may incur network usage)...')
    try:
        answer = chatbot._openai_answer(query, results, ml_output, analysis_mode='capacity_planning')
        print('\nLLM answer:\n')
        print(answer)
    except Exception as e:
        print('LLM call failed:', e)
else:
    print('\nNo OPENAI_API_KEY set; showing fallback answer:\n')
    print(chatbot._fallback_answer(query, results, ml_output, analysis_mode='capacity_planning'))
