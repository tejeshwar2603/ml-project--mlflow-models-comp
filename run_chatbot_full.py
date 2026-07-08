#!/usr/bin/env python
"""
Start both API and UI servers for the chatbot
"""
import subprocess
import time
import sys
import os

def main():
    print("=" * 60)
    print("🤖 AIOps Chatbot - Full Stack Server")
    print("=" * 60)
    
    # Start API server
    print("\n📡 Starting API Server on port 8001...")
    api_process = subprocess.Popen(
        [sys.executable, "start_chatbot.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Wait for API server to start
    time.sleep(3)
    
    # Start UI server
    print("🌐 Starting UI Server on port 8002...\n")
    ui_process = subprocess.Popen(
        [sys.executable, "chatbot_ui_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    print("=" * 60)
    print("✅ Servers started successfully!")
    print("=" * 60)
    print("\n📊 API Server:  http://localhost:8001")
    print("🎨 UI Server:   http://localhost:8002")
    print("\n👉 Open http://localhost:8002 in your browser")
    print("\nPress Ctrl+C to stop all servers\n")
    
    try:
        # Keep both processes running
        while True:
            time.sleep(1)
            if api_process.poll() is not None:
                print("❌ API server stopped unexpectedly")
                ui_process.terminate()
                sys.exit(1)
            if ui_process.poll() is not None:
                print("❌ UI server stopped unexpectedly")
                api_process.terminate()
                sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        api_process.terminate()
        ui_process.terminate()
        api_process.wait(timeout=5)
        ui_process.wait(timeout=5)
        print("✅ All servers stopped")
        sys.exit(0)

if __name__ == "__main__":
    main()
