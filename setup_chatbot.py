#!/usr/bin/env python
"""
Setup script for AIOps Chatbot - Configure LLM and environment
"""
import os
import sys
from pathlib import Path

def setup_env():
    """Interactive setup for chatbot environment"""
    print("\n" + "=" * 70)
    print("🤖 AIOps Chatbot - Environment Setup")
    print("=" * 70)
    
    env_file = Path(".env")
    
    print("\n📝 This script will help you configure the chatbot LLM provider.")
    print("\nChoose an LLM provider:\n")
    print("1. Grok (xAI) - Recommended")
    print("2. OpenAI (GPT-4)")
    print("3. Skip LLM setup (fallback to RAG-only)")
    
    choice = input("\nSelect (1-3): ").strip()
    
    env_vars = {}
    
    if choice == "1":
        print("\n🦌 Grok (xAI) Setup")
        print("Get your API key from: https://console.x.ai")
        key = input("Enter your Grok API key: ").strip()
        if key:
            env_vars["llma-key"] = key
            env_vars["LLAMA_API_URL"] = "https://api.x.ai/v1/chat/completions"
            env_vars["AIOPS_LLM_MODEL"] = "grok-4.3"
            print("✅ Grok configured")
        else:
            print("⚠️  No key provided")
    
    elif choice == "2":
        print("\n🔑 OpenAI Setup")
        print("Get your API key from: https://platform.openai.com/api-keys")
        key = input("Enter your OpenAI API key: ").strip()
        if key:
            env_vars["OPENAI_API_KEY"] = key
            env_vars["AIOPS_LLM_MODEL"] = "gpt-4-turbo"
            print("✅ OpenAI configured")
        else:
            print("⚠️  No key provided")
    
    elif choice == "3":
        print("\n⚠️  Skipping LLM setup - using RAG-only mode")
        print("Responses will be based on retrieved context only")
    
    # Always set these defaults
    env_vars["API_HOST"] = "0.0.0.0"
    env_vars["API_PORT"] = "8001"
    env_vars["UI_PORT"] = "8002"
    env_vars["RAG_VECTOR_STORE_PATH"] = "artifacts/vector_store.pkl"
    env_vars["FORECAST_MODEL_URI"] = "models:/cpu_forecast/Production"
    
    # Write to environment
    if env_file.exists():
        print(f"\n⚠️  {env_file} already exists - updating...")
    
    with open(env_file, "a" if env_file.exists() else "w") as f:
        for key, value in env_vars.items():
            f.write(f"\n{key}={value}")
    
    print(f"\n✅ Configuration saved to {env_file}")
    
    print("\n" + "=" * 70)
    print("📊 Next steps:")
    print("=" * 70)
    print("\n1. Start both servers:")
    print("   python run_chatbot_full.py")
    print("\n2. Or start individually:")
    print("   API Server:  python start_chatbot.py")
    print("   UI Server:   python chatbot_ui_server.py")
    print("\n3. Open in browser: http://localhost:8002")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    setup_env()
