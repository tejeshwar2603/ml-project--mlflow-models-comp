#!/usr/bin/env python
"""
Standalone Chatbot UI Server
Serves a modern web interface for chatbot interaction
"""
import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AIOps Chatbot UI Server")

# Add CORS middleware to allow requests to the API server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
API_SERVER = os.getenv("API_SERVER_URL", "http://localhost:8001")
UI_PORT = int(os.getenv("UI_PORT", "8002"))
UI_HOST = os.getenv("UI_HOST", "0.0.0.0")

logger.info(f"Chatbot UI Server configured to connect to API at: {API_SERVER}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main chatbot UI"""
    return get_chatbot_html()


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """Serve the chatbot UI page"""
    return get_chatbot_html()


@app.post("/api/chat")
async def proxy_chat(request: dict):
    """Proxy chat requests to the backend API"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{API_SERVER}/chat", json=request)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to API server at {API_SERVER}. Make sure it's running.",
        )
    except Exception as e:
        logger.error(f"Error proxying chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Check health of both UI and API servers"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            api_response = await client.get(f"{API_SERVER}/health")
            if api_response.status_code == 200:
                api_payload = api_response.json()
                api_status = api_payload.get("status", "ok")
                llm_configured = bool(api_payload.get("llm_configured"))
            else:
                api_status = "error"
                llm_configured = False
    except Exception as e:
        logger.warning(f"API server health check failed: {e}")
        api_status = "unreachable"
        llm_configured = False
    
    return {
        "ui_status": "ok",
        "api_status": api_status,
        "llm_configured": llm_configured,
        "api_server": API_SERVER,
    }


def get_chatbot_html():
    """Return the standalone chatbot UI HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIOps Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            width: 100%;
            max-width: 900px;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 36px;
            margin-bottom: 8px;
            font-weight: 700;
        }
        
        .header p {
            font-size: 16px;
            opacity: 0.9;
        }
        
        .info-banner {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
            color: white;
            font-size: 14px;
            line-height: 1.6;
        }
        
        .info-banner.warning {
            background: rgba(251, 146, 60, 0.1);
            border-color: rgba(251, 146, 60, 0.5);
        }
        
        .info-banner strong {
            display: block;
            margin-bottom: 8px;
        }
        
        .chat-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            height: 600px;
            overflow: hidden;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        
        .message {
            display: flex;
            margin-bottom: 12px;
            animation: slideIn 0.3s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message.bot {
            justify-content: flex-start;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 12px;
            word-wrap: break-word;
            line-height: 1.5;
        }
        
        .user .message-content {
            background: #667eea;
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .bot .message-content {
            background: #f0f0f0;
            color: #333;
            border-bottom-left-radius: 4px;
        }
        
        .message.error .message-content {
            background: #fee;
            color: #c33;
            border-bottom-left-radius: 4px;
        }
        
        .loading {
            display: flex;
            gap: 4px;
            align-items: center;
        }
        
        .loading span {
            width: 8px;
            height: 8px;
            background: #667eea;
            border-radius: 50%;
            animation: bounce 1.4s infinite;
        }
        
        .loading span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .loading span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes bounce {
            0%, 80%, 100% {
                transform: scaleY(1);
            }
            40% {
                transform: scaleY(1.5);
            }
        }
        
        .input-area {
            padding: 16px 24px;
            border-top: 1px solid #eee;
            display: flex;
            gap: 12px;
            background: #fafafa;
        }
        
        .input-wrapper {
            flex: 1;
            display: flex;
            gap: 8px;
        }
        
        input[type="text"] {
            flex: 1;
            border: 1px solid #ddd;
            border-radius: 24px;
            padding: 12px 16px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        
        input[type="text"]:focus {
            border-color: #667eea;
        }
        
        input[type="text"]::placeholder {
            color: #999;
        }
        
        button {
            background: #667eea;
            color: white;
            border: none;
            border-radius: 24px;
            padding: 12px 24px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
            font-size: 14px;
        }
        
        button:hover {
            background: #5568d3;
        }
        
        button:active {
            transform: scale(0.98);
        }
        
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .controls {
            display: flex;
            gap: 8px;
        }
        
        select {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 12px;
            font-size: 13px;
            cursor: pointer;
            background: white;
        }
        
        .status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px 16px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        .status-indicator.connected {
            background: #4caf50;
        }
        
        .status-indicator.disconnected {
            background: #f44336;
            animation: none;
        }
        
        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.5;
            }
        }
        
        .response-details {
            font-size: 12px;
            color: #666;
            margin-top: 8px;
            padding: 8px;
            background: #f5f5f5;
            border-radius: 6px;
            max-width: 70%;
        }
        
        .response-details strong {
            color: #333;
        }
        
        .scrollbar-hide::-webkit-scrollbar {
            display: none;
        }
        
        .chat-messages {
            scrollbar-width: thin;
            scrollbar-color: #ddd #f5f5f5;
        }
        
        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-messages::-webkit-scrollbar-track {
            background: #f5f5f5;
            border-radius: 10px;
        }
        
        .chat-messages::-webkit-scrollbar-thumb {
            background: #ddd;
            border-radius: 10px;
        }
        
        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: #ccc;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AIOps Chatbot</h1>
            <p>Ask about forecasts, capacity planning, and operational insights</p>
        </div>
        
        <div id="configStatus" class="info-banner"></div>
        
        <div class="chat-container">
            <div class="chat-messages" id="messages">
                <div class="message bot">
                    <div class="message-content">
                        Hi! I'm your AIOps assistant. Ask me about CPU forecasts, capacity planning, incident prevention, or any operational insights. 👋
                    </div>
                </div>
            </div>
            
            <div class="input-area">
                <div class="input-wrapper">
                    <input 
                        type="text" 
                        id="input" 
                        placeholder="Ask about forecasts, capacity planning, root causes..." 
                        autocomplete="off"
                    />
                    <select id="analysisMode">
                        <option value="general">General</option>
                        <option value="capacity_planning">Capacity</option>
                        <option value="incident_prevention">Prevention</option>
                        <option value="root_cause">Root Cause</option>
                        <option value="jira_ticket">Jira Ticket</option>
                        <option value="executive_report">Executive</option>
                    </select>
                </div>
                <button id="sendBtn" onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <div style="margin-top: 20px; color: white; font-size: 13px;">
            <details style="cursor: pointer;">
                <summary style="margin-bottom: 10px;">💡 Try these example queries</summary>
                <div style="background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px; margin-top: 8px;">
                    <div style="margin: 8px 0; cursor: pointer; hover: opacity: 0.8;" onclick="document.getElementById('input').value='Analyze CPU forecast for server App-101'; sendMessage()">
                        → "Analyze CPU forecast for server App-101"
                    </div>
                    <div style="margin: 8px 0; cursor: pointer;" onclick="document.getElementById('input').value='What servers will exceed 90% CPU next week?'; sendMessage()">
                        → "What servers will exceed 90% CPU next week?"
                    </div>
                    <div style="margin: 8px 0; cursor: pointer;" onclick="document.getElementById('analysisMode').value='capacity_planning'; document.getElementById('input').value='Plan capacity for the next 7 days'; sendMessage()">
                        → "Plan capacity for the next 7 days"
                    </div>
                    <div style="margin: 8px 0; cursor: pointer;" onclick="document.getElementById('analysisMode').value='root_cause'; document.getElementById('input').value='What causes high CPU usage?'; sendMessage()">
                        → "What causes high CPU usage?"
                    </div>
                </div>
            </details>
        </div>
    </div>
    
    <div class="status" id="status">
        <div class="status-indicator connected" id="statusIndicator"></div>
        <span id="statusText">Connected</span>
    </div>

    <script>
        const messagesDiv = document.getElementById('messages');
        const input = document.getElementById('input');
        const sendBtn = document.getElementById('sendBtn');
        const analysisMode = document.getElementById('analysisMode');
        const configStatus = document.getElementById('configStatus');
        
        let isConnected = true;
        let hasLLM = false;
        
        // Check API health and LLM status on load
        async function checkHealth() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                isConnected = data.api_status === 'ok';
                hasLLM = data.api_status === 'ok' && data.llm_configured;
                updateStatus();
            } catch (e) {
                isConnected = false;
                updateStatus();
            }
        }
        
        function updateStatus() {
            let banner = '<strong>⚠️ Configuration Status:</strong>';
            
            if (!isConnected) {
                banner += '<div>API Server: <span style="color: #ff6b6b;">❌ Disconnected</span></div>';
                configStatus.className = 'info-banner warning';
            } else {
                banner += '<div>API Server: <span style="color: #51cf66;">✅ Connected</span></div>';
                
                if (hasLLM) {
                    banner += '<div>LLM Provider: <span style="color: #51cf66;">✅ Configured</span> - Full AI responses enabled</div>';
                    configStatus.className = 'info-banner';
                } else {
                    banner += '<div>LLM Provider: <span style="color: #ffa94d;">⚠️ Not configured</span></div>';
                    banner += '<div style="margin-top: 8px; font-size: 12px;">Run <code style="background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 3px;">python setup_chatbot.py</code> to configure an LLM provider (Grok or OpenAI) for full AI responses. Without it, answers will be based on retrieved context only.</div>';
                    configStatus.className = 'info-banner warning';
                }
            }
            
            configStatus.innerHTML = banner;
        }
        
        function addMessage(content, isUser = false, isError = false, details = null) {
            const msg = document.createElement('div');
            msg.className = `message ${isUser ? 'user' : isError ? 'error' : 'bot'}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            msg.appendChild(contentDiv);
            
            if (details && !isUser) {
                const detailsDiv = document.createElement('div');
                detailsDiv.className = 'response-details';
                detailsDiv.innerHTML = details;
                msg.appendChild(detailsDiv);
            }
            
            messagesDiv.appendChild(msg);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function addLoadingMessage() {
            const msg = document.createElement('div');
            msg.className = 'message bot';
            msg.id = 'loading-msg';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.innerHTML = '<div class="loading"><span></span><span></span><span></span></div>';
            msg.appendChild(contentDiv);
            
            messagesDiv.appendChild(msg);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function removeLoadingMessage() {
            const loading = document.getElementById('loading-msg');
            if (loading) loading.remove();
        }
        
        async function sendMessage() {
            const question = input.value.trim();
            if (!question) return;
            
            if (!isConnected) {
                addMessage('❌ Error: Cannot connect to API server. Please ensure the API is running.', false, true);
                return;
            }
            
            addMessage(question, true);
            input.value = '';
            sendBtn.disabled = true;
            addLoadingMessage();
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question,
                        analysis_mode: analysisMode.value,
                        top_k: 5,
                    }),
                });
                
                removeLoadingMessage();
                
                if (!response.ok) {
                    const error = await response.json().catch(() => ({ detail: response.statusText }));
                    addMessage(`Error: ${error.detail || 'Request failed'}`, false, true);
                } else {
                    const data = await response.json();
                    const answer = data.answer || 'No answer received.';
                    
                    let details = `<strong>Mode:</strong> ${data.analysis_mode || 'general'}`;
                    if (data.risk) {
                        details += ` | <strong>Risk:</strong> ${data.risk.level || 'unknown'}`;
                    }
                    if (data.sources && data.sources.length > 0) {
                        details += ` | <strong>Sources:</strong> ${data.sources.length} retrieved`;
                    }
                    
                    addMessage(answer, false, false, details);
                }
            } catch (err) {
                removeLoadingMessage();
                isConnected = false;
                updateStatus();
                addMessage(`Connection error: ${err.message}`, false, true);
            } finally {
                sendBtn.disabled = false;
                input.focus();
            }
        }
        
        // Event listeners
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        
        // Check health on load and periodically
        checkHealth();
        setInterval(checkHealth, 30000);
    </script>
</body>
</html>
"""


def main():
    logger.info(f"🚀 Starting AIOps Chatbot UI Server on {UI_HOST}:{UI_PORT}")
    logger.info(f"📡 Backend API Server: {API_SERVER}")
    logger.info(f"🌐 Open browser at: http://localhost:{UI_PORT}")
    
    uvicorn.run(app, host=UI_HOST, port=UI_PORT, log_level="info")


if __name__ == "__main__":
    main()
