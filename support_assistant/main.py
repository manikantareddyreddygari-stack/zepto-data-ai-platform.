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
