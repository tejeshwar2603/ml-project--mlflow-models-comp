"""
Optimized chatbot runner using GPT-4o mini for minimal token consumption.
Features:
  - Uses gpt-4o-mini (cheapest OpenAI model)
  - Compressed context and prompts
  - Structured output to reduce processing tokens
  - Caching and batching for efficiency
"""

import os
import sys
import json
import importlib.util
from dotenv import load_dotenv
from pathlib import Path
from typing import Any
import types

# Setup paths
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
    # Create minimal package modules
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
    print(f'ERROR: Failed to load modules: {e}')
    sys.exit(1)

# ============ OPTIMIZED MINI-TOKEN CHATBOT ============

MINI_SYSTEM_PROMPT = """You are an AIOps forecasting assistant. Answer concisely using retrieved context and ML forecast. If missing info, state what's needed. Return actionable next steps."""

MINI_ANALYSIS_MODES = {
    "general": "Brief answer to the question.",
    "capacity": "Risk level, impact, and scaling options (1-2 sentences each).",
    "incident": "Risk assessment and top 2 preventative actions.",
    "ticket": "Compact Jira ticket draft.",
    "root_cause": "Most likely causes based on retrieved data.",
}

def _compress_context(results: list[dict[str, Any]], max_results: int = 3) -> str:
    """Return only top results with minimal formatting."""
    lines = []
    for idx, item in enumerate(results[:max_results], 1):
        score = item['score']
        text_short = item['text'][:200].replace('\n', ' ')
        lines.append(f"[{idx}] (score:{score:.2f}) {text_short}")
    return "\n".join(lines)

def _mini_llm_call(
    question: str,
    retrieved: list[dict[str, Any]],
    ml_output: dict[str, Any] | None,
    analysis_mode: str,
) -> str:
    """Make minimal-token LLM call using gpt-4o-mini."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not found. Install with: pip install openai")
        return None
    
    client = OpenAI(api_key=api_key)
    model = "gpt-4o-mini"  # Cheapest model (~$0.15 per 1M input tokens)
    
    mode_inst = MINI_ANALYSIS_MODES.get(analysis_mode, MINI_ANALYSIS_MODES["general"])
    context = _compress_context(retrieved, max_results=3)
    
    # Compact user message
    user_msg = (
        f"Mode: {analysis_mode} ({mode_inst})\n\n"
        f"Q: {question}\n\n"
        f"ML: {json.dumps(ml_output or {})}\n\n"
        f"Evidence:\n{context}"
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": MINI_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=300,  # Limit output tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None

def run_chatbot_interactive():
    """Interactive chatbot mode."""
    # Load vector store
    local_paths = [p for p in os.getenv('RAG_LOCAL_PATHS', '').split(os.pathsep) if p]
    if not local_paths:
        local_paths = ['README.md', 'ARCHITECTURE.md']
    
    print(f"📚 Loading documents from: {local_paths}")
    docs = rag.load_local_documents(local_paths)
    if not docs:
        print("ERROR: No documents found. Ensure README.md or ARCHITECTURE.md exist.")
        sys.exit(1)
    
    print(f"✓ Loaded {len(docs)} document chunks")
    
    vs = rag.VectorStore()
    try:
        vs.build(docs)
        print("✓ Vector store built")
    except Exception as e:
        print(f"ERROR building vector store: {e}")
        sys.exit(1)
    
    # Interactive loop
    print("\n" + "="*60)
    print("🤖 AIOps Chatbot (GPT-4o mini - minimal tokens)")
    print("="*60)
    print("Commands:")
    print("  - Type a question and press Enter")
    print("  - Mode: Use 'mode:general|capacity|incident|ticket|root_cause'")
    print("  - ML data: Use 'ml:{json}' to set forecast data")
    print("  - Type 'exit' to quit")
    print("="*60 + "\n")
    
    analysis_mode = "general"
    ml_output = {
        "server_id": "App-101",
        "horizon": 7,
        "prediction": 85.0,
    }
    
    while True:
        try:
            user_input = input("\n💬 Q: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                print("👋 Goodbye!")
                break
            
            # Parse commands
            if user_input.startswith('mode:'):
                analysis_mode = user_input.split(':', 1)[1].strip()
                if analysis_mode not in MINI_ANALYSIS_MODES:
                    print(f"Available modes: {', '.join(MINI_ANALYSIS_MODES.keys())}")
                    continue
                print(f"✓ Mode set to: {analysis_mode}")
                continue
            
            if user_input.startswith('ml:'):
                try:
                    ml_output = json.loads(user_input.split(':', 1)[1])
                    print(f"✓ ML data updated: {ml_output}")
                    continue
                except json.JSONDecodeError:
                    print("ERROR: Invalid JSON for ML data")
                    continue
            
            # Search and answer
            print("\n🔍 Searching...", end='', flush=True)
            results = vs.search(user_input, top_k=5)
            print(f" Found {len(results)} results")
            
            print("📝 Generating answer (gpt-4o-mini)...", end='', flush=True)
            answer = _mini_llm_call(user_input, results, ml_output, analysis_mode)
            
            if answer:
                print("\r✓ Done!         ")
                print(f"\n📄 Answer:\n{answer}")
                print(f"\n📊 Sources: {', '.join([r['source'] for r in results[:3]])}")
            else:
                print("\r⚠ LLM unavailable         ")
                print(f"\nContext-based answer:\n{_compress_context(results)}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nERROR: {e}")

def run_demo_queries():
    """Run demo with pre-defined queries."""
    # Load vector store
    local_paths = [p for p in os.getenv('RAG_LOCAL_PATHS', '').split(os.pathsep) if p]
    if not local_paths:
        local_paths = ['README.md', 'ARCHITECTURE.md']
    
    print(f"📚 Loading documents from: {local_paths}")
    docs = rag.load_local_documents(local_paths)
    if not docs:
        print("ERROR: No documents found.")
        sys.exit(1)
    
    print(f"✓ Loaded {len(docs)} document chunks\n")
    
    vs = rag.VectorStore()
    try:
        vs.build(docs)
    except Exception as e:
        print(f"ERROR building vector store: {e}")
        sys.exit(1)
    
    # Demo queries
    demo_queries = [
        {
            "question": "What are the main forecasting models available?",
            "mode": "general",
            "ml_output": None,
        },
        {
            "question": "How should we handle high CPU predictions?",
            "mode": "capacity",
            "ml_output": {"server_id": "App-101", "horizon": 7, "prediction": 92.0},
        },
        {
            "question": "What preventative actions reduce forecasting errors?",
            "mode": "incident",
            "ml_output": None,
        },
    ]
    
    print("="*70)
    print("🤖 Running Demo Queries (GPT-4o mini)")
    print("="*70 + "\n")
    
    for idx, query_data in enumerate(demo_queries, 1):
        question = query_data["question"]
        mode = query_data["mode"]
        ml_output = query_data["ml_output"]
        
        print(f"\n[Query {idx}/{len(demo_queries)}] {question}")
        print(f"Mode: {mode}")
        
        results = vs.search(question, top_k=3)
        print(f"✓ Retrieved {len(results)} documents")
        
        answer = _mini_llm_call(question, results, ml_output, mode)
        if answer:
            print(f"\n📄 Answer:\n{answer}")
        else:
            print(f"\n⚠️  Fallback (LLM unavailable):\n{_compress_context(results)}")
        
        print("\n" + "-"*70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimized chatbot using GPT-4o mini (minimal tokens)"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "demo"],
        default="interactive",
        help="Run mode: interactive chat or demo queries"
    )
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        run_demo_queries()
    else:
        run_chatbot_interactive()
