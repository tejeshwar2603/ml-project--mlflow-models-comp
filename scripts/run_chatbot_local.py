"""
Lightweight chatbot runner using local LLM fallback (no API costs).
Demonstrates:
  - Vector search retrieval
  - Structured output generation
  - Multi-mode analysis without external API calls
  - Minimal token/memory footprint
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

# ============ LOCAL MINI LLM CHATBOT (NO API COSTS) ============

ANALYSIS_TEMPLATES = {
    "general": {
        "prefix": "Based on the retrieved context:",
        "postfix": "Next steps: Review the evidence and adjust forecasting strategy as needed.",
    },
    "capacity": {
        "prefix": "Capacity Analysis:",
        "postfix": "Action: Monitor utilization trends and plan scaling in advance.",
    },
    "incident": {
        "prefix": "Risk Assessment:",
        "postfix": "Mitigation: Implement the suggested preventative measures.",
    },
    "ticket": {
        "prefix": "Jira Ticket Draft:",
        "postfix": "Resolution: Engineer review required before implementation.",
    },
    "root_cause": {
        "prefix": "Likely Root Causes:",
        "postfix": "Investigation: Cross-reference metrics and logs for confirmation.",
    },
}

def _compress_context(results: list[dict[str, Any]], max_results: int = 3) -> str:
    """Return top results with minimal formatting."""
    lines = []
    for idx, item in enumerate(results[:max_results], 1):
        score = item['score']
        text_short = item['text'][:180].replace('\n', ' ').strip()
        lines.append(f"  [{idx}] (relevance:{score:.1%}) {text_short}...")
    return "\n".join(lines)

def _generate_local_answer(
    question: str,
    retrieved: list[dict[str, Any]],
    ml_output: dict[str, Any] | None,
    analysis_mode: str,
) -> str:
    """Generate answer using local templates (zero API cost)."""
    
    template = ANALYSIS_TEMPLATES.get(analysis_mode, ANALYSIS_TEMPLATES["general"])
    
    # Build answer
    lines = [template["prefix"]]
    lines.append("")
    
    # Add retrieved context
    if retrieved:
        lines.append("📚 Retrieved Evidence:")
        lines.append(_compress_context(retrieved, max_results=3))
        lines.append("")
    
    # Add ML forecast if available
    if ml_output:
        lines.append("🤖 ML Forecast Context:")
        lines.append(f"  • Server: {ml_output.get('server_id', 'N/A')}")
        lines.append(f"  • Horizon: {ml_output.get('horizon', 'N/A')} days")
        if ml_output.get('prediction') is not None:
            pred = ml_output.get('prediction')
            if pred >= 90:
                risk = "🔴 CRITICAL"
            elif pred >= 80:
                risk = "🟠 HIGH"
            elif pred >= 70:
                risk = "🟡 MEDIUM"
            else:
                risk = "🟢 LOW"
            lines.append(f"  • Predicted CPU: {pred}% - {risk}")
        lines.append("")
    
    # Mode-specific guidance
    if analysis_mode == "capacity" and ml_output and ml_output.get('prediction', 0) >= 80:
        lines.append("⚠️  High CPU predicted. Recommended actions:")
        lines.append("  1. Validate against current metrics")
        lines.append("  2. Review related Jira issues for similar incidents")
        lines.append("  3. Prepare scaling options for review")
    
    elif analysis_mode == "incident":
        lines.append("🛡️  Preventative Actions:")
        lines.append("  1. Monitor metric trends weekly")
        lines.append("  2. Maintain runbooks for known incident patterns")
        lines.append("  3. Automate scaling policies for predicted peaks")
    
    elif analysis_mode == "ticket" and retrieved:
        lines.append("🎫 Ticket Summary:")
        lines.append(f"  Title: CPU Utilization Forecast Alert")
        lines.append(f"  Status: Review Required")
        lines.append(f"  Priority: High" if ml_output and ml_output.get('prediction', 0) >= 80 else "  Priority: Medium")
    
    lines.append("")
    lines.append(template["postfix"])
    
    return "\n".join(lines)

def run_interactive_demo():
    """Interactive demonstration mode."""
    # Load documents
    local_paths = [p for p in os.getenv('RAG_LOCAL_PATHS', '').split(os.pathsep) if p]
    if not local_paths:
        local_paths = ['README.md', 'ARCHITECTURE.md']
    
    print(f"📚 Loading documents from: {local_paths}")
    docs = rag.load_local_documents(local_paths)
    if not docs:
        print("ERROR: No documents found.")
        sys.exit(1)
    
    print(f"✓ Loaded {len(docs)} document chunks\n")
    
    # Build vector store
    vs = rag.VectorStore()
    try:
        vs.build(docs)
        print("✓ Vector store ready")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Interactive loop
    print("\n" + "="*70)
    print("🤖 AIOps Chatbot - Local Mode (Zero API Cost)")
    print("="*70)
    print("\nCommands:")
    print("  • Type your question and press Enter")
    print("  • 'mode:general|capacity|incident|ticket|root_cause' - Change analysis mode")
    print("  • 'ml:{...}' - Set ML forecast data as JSON")
    print("  • 'examples' - Show demo queries")
    print("  • 'exit' - Quit\n")
    
    analysis_mode = "general"
    ml_output = {
        "server_id": "App-101",
        "horizon": 7,
        "prediction": 85.0,
    }
    
    def show_examples():
        print("\n" + "-"*70)
        print("📋 Example Queries:")
        print("-"*70)
        examples = [
            "What are the main forecasting models?",
            "How do we handle high CPU predictions?",
            "What's the project architecture?",
            "How is evaluation done?",
        ]
        for i, ex in enumerate(examples, 1):
            print(f"  {i}. {ex}")
        print("-"*70 + "\n")
    
    while True:
        try:
            user_input = input("\n💬 Q: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'examples':
                show_examples()
                continue
            
            # Parse commands
            if user_input.startswith('mode:'):
                analysis_mode = user_input.split(':', 1)[1].strip()
                if analysis_mode not in ANALYSIS_TEMPLATES:
                    print(f"Available modes: {', '.join(ANALYSIS_TEMPLATES.keys())}")
                    continue
                print(f"✓ Mode: {analysis_mode}")
                continue
            
            if user_input.startswith('ml:'):
                try:
                    ml_output = json.loads(user_input.split(':', 1)[1])
                    print(f"✓ ML data set: {ml_output}")
                    continue
                except json.JSONDecodeError:
                    print("ERROR: Invalid JSON")
                    continue
            
            # Search
            print("🔍 Searching...", end='', flush=True)
            results = vs.search(user_input, top_k=5)
            print(f" ✓ Found {len(results)} relevant chunks")
            
            # Generate answer
            print("💭 Generating answer (local)...", end='', flush=True)
            answer = _generate_local_answer(user_input, results, ml_output, analysis_mode)
            print(" ✓")
            
            print(f"\n{answer}")
            
            if results:
                print("\n📌 Sources:")
                for r in results[:3]:
                    print(f"   • {r['source']} (relevance: {r['score']:.1%})")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def run_batch_demo():
    """Demo mode with pre-defined queries."""
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
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Demo queries
    queries = [
        ("What are the main forecasting models?", "general", None),
        ("How should we handle high CPU predictions?", "capacity", {"server_id": "App-101", "horizon": 7, "prediction": 92.0}),
        ("What prevents forecasting errors?", "incident", None),
        ("Create a Jira ticket for CPU risk", "ticket", {"server_id": "App-102", "horizon": 3, "prediction": 88.0}),
    ]
    
    print("="*70)
    print("🚀 DEMO: AIOps Chatbot - Local Mode (Zero API Cost)")
    print("="*70)
    
    for idx, (question, mode, ml_data) in enumerate(queries, 1):
        print(f"\n[Query {idx}/{len(queries)}]")
        print(f"❓ Q: {question}")
        print(f"📊 Mode: {mode}")
        
        results = vs.search(question, top_k=5)
        print(f"✓ Retrieved {len(results)} documents")
        
        answer = _generate_local_answer(question, results, ml_data, mode)
        print(f"\n{answer}")
        
        if results:
            print("\n📌 Top sources:")
            for r in results[:2]:
                print(f"   • {r['source']}")
        
        print("\n" + "-"*70)
    
    print("\n✅ Demo complete!")
    print("\n💡 To run interactive mode: python scripts/run_chatbot_local.py --mode interactive")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Local AIOps chatbot (zero API cost)")
    parser.add_argument("--mode", choices=["interactive", "demo"], default="demo",
                       help="Run mode")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        run_interactive_demo()
    else:
        run_batch_demo()
