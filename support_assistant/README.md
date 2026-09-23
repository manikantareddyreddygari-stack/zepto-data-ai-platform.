# Module 3: Support Assistant (`/support_assistant`)

## Overview
This module implements a grounded, production-ready GenAI customer support service for Zepto's quick-commerce platform. It features:
- A curated document corpus of 8 official Zepto operational policies (`docs/doc_01.txt` ... `docs/doc_08.txt`).
- Local vector embeddings generated via `sentence-transformers/all-MiniLM-L6-v2` and indexed in a persistent local `ChromaDB` collection.
- An intelligent LangGraph `StateGraph` workflow that routes customer queries through intent classification, semantic retrieval, and structured response synthesis.
- A deterministic, offline **Mock Mode (`MOCK_LLM=1`)** as the graded baseline—requiring zero external API keys, zero paid services, and zero network calls to any LLM provider.
- Full validation against a strict Pydantic JSON schema (`answer`, `sources`, `confidence`) with retry handling for real-LLM extensions.
- A containerized FastAPI service with `POST /ask` and `GET /health` endpoints, fully deployable via Docker.

---

## The `MOCK_LLM` Environment Toggle
The support assistant is engineered with a strict gating toggle:
- **`MOCK_LLM=1` (Default / Graded Baseline)**:
  - The service operates in a completely deterministic, offline mock mode.
  - **No signup, no API key, and no external LLM network calls are made.**
  - Retrieval from ChromaDB using `all-MiniLM-L6-v2` still executes for real on local embeddings.
  - Generates answers using the exact required mock template: `f"Based on the retrieved context: {top_chunk_snippet}"`.
  - Confidence is deterministically set to `1.0`, and cited document IDs are populated from actual ChromaDB vector matches.
- **`MOCK_LLM=0` (Optional Real-LLM Extension)**:
  - Connects to an external LLM provider (e.g. Groq free tier or local endpoint).
  - Uses the structured prompt template below with few-shot guidance and negative constraints.
  - Validates JSON output against the Pydantic schema with an automated 2-retry corrective loop on failure.

---

## Structured Prompt Template
The system prompt is engineered according to the **Role–Context–Task–Format–Length** framework, incorporating explicit negative constraints and few-shot examples:

```text
[ROLE]
You are Zepto's official AI Support Assistant, providing accurate, grounded information about company policies.

[CONTEXT]
Retrieved Policy Documents:
{context}

[TASK]
Answer the user's inquiry: "{query}"

[NEGATIVE CONSTRAINTS]
- Do not answer using information not present in the provided context.
- If the answer cannot be determined strictly from the context, state: "I do not have sufficient information in Zepto policy documents to answer this."
- Do not speculate, invent promotional codes, or mention external competitor services.

[FORMAT]
Respond strictly with a JSON object adhering to:
{
  "answer": "Clear, grounded response string",
  "sources": ["doc_id1", "doc_id2"],
  "confidence": 0.95
}

[LENGTH]
Keep the answer concise and direct (under 150 words).

[FEW-SHOT EXAMPLE]
User Query: "How much does Zepto Pass cost?"
Context:
[doc_03] Zepto offers three account tiers: Basic (free, default tier), Zepto Pass (INR 49 per month, free standard delivery on all orders), and Zepto Pass+ (INR 99 per month).
Output JSON:
{
  "answer": "Zepto Pass costs INR 49 per month and includes free standard delivery on all orders as well as 5% off select categories.",
  "sources": ["doc_03"],
  "confidence": 1.0
}
```

---

## RAG Pipeline Architecture & Data Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION & EMBEDDING STAGE                                         │
│    docs/doc_01.txt ... doc_08.txt                                      │
│         │                                                              │
│         ▼                                                              │
│    sentence-transformers (all-MiniLM-L6-v2) ──► ChromaDB Collection    │
│    (vector_store.py: ingest_corpus)             (zepto_policy_corpus) │
└────────────────────────────────────────────────────────────────────────┘
                                                         │
┌────────────────────────────────────────────────────────┼───────────────┐
│ 2. INQUIRY & LANGGRAPH ROUTING STAGE                   │               │
│                                                        ▼               │
│    POST /ask {"query": "..."} ──► Node 1: classify_intent              │
│    (main.py)                      (graph.py: keyword heuristic / LLM)   │
│                                          │                             │
│                    ┌─────────────────────┴─────────────────────┐       │
│                    ▼ (policy_question)                         ▼       │
│    Node 2: retrieve_and_answer                 Node 3: direct_answer   │
│    (ChromaDB Cosine Retrieval)                 (Canned general string) │
│    - Mock: Canned snippet template             - Mock: No retrieval    │
│    - Real: Grounded LLM generation             - Real: Direct LLM      │
└────────────────────┬───────────────────────────────────────────┬───────┘
                     │                                           │
┌────────────────────┴───────────────────────────────────────────┴───────┐
│ 3. SCHEMA ENFORCEMENT & SERVING                                        │
│    Validated Pydantic Response Schema (schemas.py: QueryResponse)      │
│    { "answer": "...", "sources": [...], "confidence": 1.0 }            │
└────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibility Breakdown:
1. **Ingestion & Embedding (`vector_store.py:ingest_corpus`)**:
   - Reads 8 raw policy documents from `docs/`.
   - Embeds each document using `all-MiniLM-L6-v2`.
   - Stores dense vectors, text chunks, and metadata in local ChromaDB directory `chroma_db/`.
2. **Intent Classification Node (`graph.py:classify_intent_node`)**:
   - Evaluates incoming query against key policy terms: `"delivery"`, `"return"`, `"refund"`, `"membership"`, `"tracking"`, `"cancel"`, `"gift card"`, `"support hours"`.
   - Routes to `retrieve_and_answer` if matched; else to `direct_answer`.
3. **Retrieval & Answering Node (`graph.py:retrieve_and_answer_node`)**:
   - Calls `vector_store.py:retrieve_similar_chunks()` to run real cosine similarity retrieval on ChromaDB.
   - Extracts top-3 chunks.
   - Mock branch: Returns `f"Based on the retrieved context: {top_chunk_snippet}"` and populates `sources = ["doc_XX", ...]`.
4. **Direct Answering Node (`graph.py:direct_answer_node`)**:
   - Mock branch: Returns `"I can only answer questions about Zepto policies right now."` with `sources = []`.
5. **Schema Validation (`schemas.py:QueryResponse`)**:
   - Ensures response contains `answer: str`, `sources: List[str]`, and `confidence: float`.

---

## Example Call Transcripts (Recorded with Default `MOCK_LLM=1`)

### Call 1: Policy Question (Delivery Policies)
- **Request**:
  ```bash
  curl -X POST "http://localhost:7860/ask" \
       -H "Content-Type: application/json" \
       -d '{"query": "What are your delivery fees and delivery timeframes?"}'
  ```
- **Raw JSON Response**:
  ```json
  {
    "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard deliv...",
    "sources": [
      "doc_01",
      "doc_04",
      "doc_05"
    ],
    "confidence": 1.0
  }
  ```

### Call 2: Policy Question (Returns & Refunds)
- **Request**:
  ```bash
  curl -X POST "http://localhost:7860/ask" \
       -H "Content-Type: application/json" \
       -d '{"query": "Can I return damaged or spoiled grocery items for a refund?"}'
  ```
- **Raw JSON Response**:
  ```json
  {
    "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened...",
    "sources": [
      "doc_02",
      "doc_06",
      "doc_05"
    ],
    "confidence": 1.0
  }
  ```

### Call 3: General Question (Non-Policy Query)
- **Request**:
  ```bash
  curl -X POST "http://localhost:7860/ask" \
       -H "Content-Type: application/json" \
       -d '{"query": "What is the capital city of Australia?"}'
  ```
- **Raw JSON Response**:
  ```json
  {
    "answer": "I can only answer questions about Zepto policies right now.",
    "sources": [],
    "confidence": 1.0
  }
  ```

---

## How to Run & Verify Locally

### 1. Run Automated Test Suite
```bash
python support_assistant/test_assistant.py
```

### 2. Run FastAPI Application
```bash
uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```
- Interactive Swagger UI: `http://localhost:7860/docs`
- Health Check: `http://localhost:7860/health`

### 3. Containerization via Docker
Build and run the Docker container locally:
```bash
# Build image
docker build -t zepto-support-assistant -f support_assistant/Dockerfile support_assistant

# Run container
docker run -p 7860:7860 zepto-support-assistant
```
Test the containerized endpoint:
```bash
curl -X POST "http://localhost:7860/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "How do I cancel my Zepto order?"}'
```
