"""
Zepto Data & AI Platform - Module 3: Support Assistant
FastAPI Application serving the LangGraph RAG Agent
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from config import MOCK_LLM
from schemas import QueryRequest, QueryResponse
from vector_store import ingest_corpus, get_chroma_client, COLLECTION_NAME
from graph import query_assistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup initialization: Pre-indexes policy corpus into local ChromaDB."""
    logger.info("Initializing Zepto Support Assistant API...")
    try:
        ingest_corpus()
        logger.info("Policy documents indexed successfully on startup.")
    except Exception as e:
        logger.error(f"Error during startup document ingestion: {e}")
    yield
    logger.info("Shutting down Zepto Support Assistant API...")


app = FastAPI(
    title="Zepto Policy Support Assistant",
    description="Customer policy answering assistant using LangGraph and ChromaDB.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def index():
    """Interactive Customer Support Web UI and quick launcher."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Zepto Support Assistant</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }
    .container { max-width: 800px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 28px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    h1 { color: #38bdf8; margin-top: 0; display: flex; align-items: center; justify-content: space-between; font-size: 22px; }
    .badge { font-size: 13px; background: #0284c7; color: white; padding: 6px 14px; border-radius: 999px; text-decoration: none; font-weight: 500; }
    .badge:hover { background: #0369a1; }
    .chat-box { background: #0f172a; border-radius: 8px; padding: 16px; min-height: 280px; max-height: 420px; overflow-y: auto; margin-bottom: 16px; border: 1px solid #334155; }
    .msg { margin-bottom: 14px; padding: 12px 16px; border-radius: 8px; line-height: 1.5; font-size: 14px; }
    .user-msg { background: #2563eb; color: white; margin-left: 20%; }
    .bot-msg { background: #334155; color: #e2e8f0; margin-right: 15%; }
    .sources { font-size: 12px; color: #94a3b8; margin-top: 8px; border-top: 1px solid #475569; padding-top: 6px; }
    .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
    .chip { background: #334155; color: #38bdf8; border: 1px solid #475569; padding: 6px 12px; border-radius: 999px; cursor: pointer; font-size: 13px; transition: all 0.2s; }
    .chip:hover { background: #475569; color: #7dd3fc; }
    .input-row { display: flex; gap: 10px; }
    input[type="text"] { flex: 1; padding: 12px 16px; border-radius: 8px; border: 1px solid #475569; background: #0f172a; color: white; font-size: 14px; }
    input[type="text"]:focus { outline: none; border-color: #38bdf8; }
    button { background: #38bdf8; color: #0f172a; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 14px; }
    button:hover { background: #7dd3fc; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
  </style>
</head>
<body>
  <div class="container">
    <h1>
      <span>⚡ Zepto Policy Support Assistant</span>
      <a class="badge" href="/docs" target="_blank">Open Swagger API Docs &rarr;</a>
    </h1>
    <p style="color: #94a3b8; margin-top: -6px; font-size: 14px;">Powered by LangGraph RAG Agent & ChromaDB vector search.</p>
    
    <div class="chips">
      <span class="chip" onclick="askPreset('What is the delivery fee for orders below 149?')">🚚 Delivery Fee & Time</span>
      <span class="chip" onclick="askPreset('Can I return damaged or spoiled grocery items?')">📦 Returns & Refunds</span>
      <span class="chip" onclick="askPreset('How much does Zepto Pass cost?')">⭐ Zepto Pass Membership</span>
      <span class="chip" onclick="askPreset('What is the capital of France?')">❓ Non-Policy Question</span>
    </div>

    <div class="chat-box" id="chatBox">
      <div class="msg bot-msg">
        <strong>Zepto Assistant:</strong> Hello! I am your Zepto Policy Support Assistant. Click any example question above or type your own question below!
      </div>
    </div>

    <form class="input-row" onsubmit="handleSend(event)">
      <input type="text" id="queryInput" placeholder="Ask a question about delivery, refunds, membership, etc..." autocomplete="off" />
      <button type="submit" id="sendBtn">Send</button>
    </form>
  </div>

  <script>
    async function askQuestion(text) {
      const chatBox = document.getElementById('chatBox');
      const input = document.getElementById('queryInput');
      const btn = document.getElementById('sendBtn');
      
      chatBox.innerHTML += `<div class="msg user-msg"><strong>Customer:</strong> ${escapeHtml(text)}</div>`;
      chatBox.scrollTop = chatBox.scrollHeight;
      input.value = '';
      btn.disabled = true;

      try {
        const res = await fetch('/ask', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({query: text})
        });
        const data = await res.json();
        const sourcesHtml = data.sources && data.sources.length ? `<div class="sources">📚 Cited Sources: ${data.sources.join(', ')}</div>` : '';
        chatBox.innerHTML += `<div class="msg bot-msg"><strong>Zepto Assistant:</strong> ${escapeHtml(data.answer)}${sourcesHtml}</div>`;
      } catch (err) {
        chatBox.innerHTML += `<div class="msg bot-msg" style="color:#f87171;">Error connecting to API: ${err}</div>`;
      } finally {
        btn.disabled = false;
        chatBox.scrollTop = chatBox.scrollHeight;
      }
    }

    function handleSend(e) {
      e.preventDefault();
      const val = document.getElementById('queryInput').value.trim();
      if (val) askQuestion(val);
    }

    function askPreset(text) {
      askQuestion(text);
    }

    function escapeHtml(str) {
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
  </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


@app.get("/health")
def health_check():
    """Health check endpoint reporting vector store status and mock mode toggle."""
    client = get_chroma_client()
    try:
        col = client.get_collection(COLLECTION_NAME)
        doc_count = col.count()
    except Exception:
        doc_count = 0

    return {
        "status": "healthy",
        "mock_llm": MOCK_LLM,
        "indexed_policy_documents": doc_count,
        "service": "Zepto Support Assistant"
    }


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """
    Primary RAG inquiry endpoint:
    - Routes intent (policy_question vs general_question) via LangGraph
    - Performs cosine similarity retrieval on ChromaDB
    - Produces validated Pydantic JSON response schema
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        response = query_assistant(request.query)
        return response
    except Exception as e:
        logger.error(f"Error processing inquiry '{request.query}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal assistant error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
