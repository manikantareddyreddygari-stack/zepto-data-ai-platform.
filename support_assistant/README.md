# Module 3: Support Assistant (`/support_assistant`)

**Author**: Manikanta Reddy Reddygari  
**Service**: Grounded Customer Policy Support Assistant (LangGraph + ChromaDB + FastAPI)

---

## Overview

In this module, I developed a grounded GenAI support assistant for Zepto's customer policies. It indexes 8 official operational policies, routes customer inquiries through a stateful LangGraph graph, retrieves grounded context from ChromaDB using local sentence embeddings, and serves responses through a containerized FastAPI endpoint.

To ensure deterministic and zero-cost grading, the entire service runs in an offline **Mock Mode (`MOCK_LLM=1`)** by default without requiring any API keys or paid accounts.

---

## The `MOCK_LLM` Toggle

- **`MOCK_LLM=1` (Default / Graded Baseline)**:
  - Runs 100% locally and offline. No API key, no signup, and no external network calls to LLMs.
  - ChromaDB cosine similarity retrieval still runs for real on local embeddings.
  - Returns grounded answers using the required template format: `Based on the retrieved context: <top_chunk_snippet>`.
  - Sets confidence to `1.0` and cites actual retrieved document IDs (`doc_01` to `doc_08`).
- **`MOCK_LLM=0` (Optional Real-LLM Mode)**:
  - Connects to an external LLM (e.g. Groq free tier).
  - Uses the structured prompt template below with few-shot guidance.
  - Automatically retries up to 2 times if the raw LLM output fails the Pydantic schema.

---

## Structured Prompt Template

I formatted the system prompt following the **Role–Context–Task–Format–Length** framework with explicit negative constraints and few-shot examples:

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

## RAG Pipeline Architecture & Workflow

```text
  1. Ingestion Stage:
     docs/doc_01.txt ... doc_08.txt 
          │
          ▼
     sentence-transformers (all-MiniLM-L6-v2) ──► ChromaDB (zepto_policy_corpus)
     (vector_store.py)

  2. Query & Intent Routing Stage:
     User Query ──► Node 1: classify_intent (graph.py)
                         │
             ┌───────────┴───────────┐
             ▼ (policy_question)     ▼ (general_question)
     Node 2: retrieve_and_answer     Node 3: direct_answer
     - Real ChromaDB Cosine Retrieval - Canned non-policy answer
     - Top-3 chunks extracted        - Sources = []
     - Template snippet answer       - Confidence = 1.0
             │                       │
             └───────────┬───────────┘
                         ▼
  3. Validation & Serving Stage:
     Pydantic Model (schemas.py: QueryResponse)
     FastAPI POST /ask (main.py)
```

### Component Breakdown:
1. **Ingestion (`vector_store.py:ingest_corpus`)**:
   - Reads 8 policy documents from `docs/`.
   - Embeds each document with `all-MiniLM-L6-v2`.
   - Stores dense vectors and metadata in a persistent local directory (`chroma_db/`).
2. **Intent Classification (`graph.py:classify_intent_node`)**:
   - Checks the incoming query for policy keywords (`"delivery"`, `"return"`, `"refund"`, `"membership"`, `"tracking"`, `"cancel"`, `"gift card"`, `"support hours"`).
   - Routes policy questions to `retrieve_and_answer` and general questions to `direct_answer`.
3. **Retrieval & Answer (`graph.py:retrieve_and_answer_node`)**:
   - Executes cosine similarity retrieval against ChromaDB for top-3 chunks.
   - Extracts the first 200 characters of the top chunk and formats the response.
4. **Direct Answer (`graph.py:direct_answer_node`)**:
   - Returns `"I can only answer questions about Zepto policies right now."` with an empty sources list.
5. **Schema Validation (`schemas.py:QueryResponse`)**:
   - Guarantees valid JSON containing `answer: str`, `sources: List[str]`, and `confidence: float`.

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
      "doc_02",
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

## How to Run & Verify

### 1. Run Automated Test Suite
```bash
python support_assistant/test_assistant.py
```

### 2. Run Local FastAPI Server
```bash
uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```
- Swagger UI: [http://localhost:7860/docs](http://localhost:7860/docs)
- Health check: [http://localhost:7860/health](http://localhost:7860/health)

### 3. Build & Run with Docker
```bash
docker build -t zepto-support-assistant -f support_assistant/Dockerfile support_assistant
docker run -p 7860:7860 zepto-support-assistant
```
