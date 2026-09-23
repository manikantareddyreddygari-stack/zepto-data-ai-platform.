"""
Zepto Data & AI Platform - Module 3: Support Assistant
Test Runner: Tests LangGraph execution and FastAPI endpoints with mock and live calls.
Outputs exact JSON responses for evaluation and documentation.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import logging
from fastapi.testclient import TestClient
from main import app
from vector_store import ingest_corpus
from config import MOCK_LLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_tests():
    print("=" * 80)
    print("ZEPTO SUPPORT ASSISTANT: TEST SUITE & ENDPOINT VERIFICATION")
    print(f"Active Mock Mode (MOCK_LLM): {MOCK_LLM}")
    print("=" * 80)

    # 1. Pre-index corpus
    print("\n[Step 1] Ingesting & Indexing 8 Policy Documents...")
    ingest_corpus()
    print("-> Corpus indexed in ChromaDB successfully.\n")

    # 2. Initialize FastAPI TestClient
    client = TestClient(app)

    # Test /health
    print("[Step 2] Testing GET /health...")
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    print(f"Health Status: {health_resp.json()}\n")

    # 3. Test Cases
    test_queries = [
        {
            "category": "Policy Question 1 (Delivery)",
            "query": "What are your delivery fees and delivery timeframes?",
            "expected_intent": "policy_question",
            "expected_top_source": "doc_01"
        },
        {
            "category": "Policy Question 2 (Returns & Refunds)",
            "query": "Can I return damaged or spoiled grocery items for a refund?",
            "expected_intent": "policy_question",
            "expected_top_source": "doc_02"
        },
        {
            "category": "Policy Question 3 (Membership Tiers)",
            "query": "How much does a Zepto Pass membership cost per month?",
            "expected_intent": "policy_question",
            "expected_top_source": "doc_03"
        },
        {
            "category": "General Question (Non-Policy)",
            "query": "What is the capital city of Australia?",
            "expected_intent": "general_question",
            "expected_top_source": None
        }
    ]

    results = []

    print("[Step 3] Executing POST /ask Test Cases...\n")
    for tc in test_queries:
        print("-" * 80)
        print(f"Test Case: {tc['category']}")
        print(f"Inquiry:   \"{tc['query']}\"")

        response = client.post("/ask", json={"query": tc["query"]})
        assert response.status_code == 200, f"Request failed with status {response.status_code}: {response.text}"

        data = response.json()
        print("\nRaw JSON Response:")
        print(json.dumps(data, indent=2))

        # Verification checks
        if tc["expected_intent"] == "policy_question":
            assert len(data["sources"]) > 0, "Expected non-empty sources for policy question!"
            assert tc["expected_top_source"] in data["sources"], f"Expected {tc['expected_top_source']} in sources!"
            assert data["answer"].startswith("Based on the retrieved context:"), "Expected mock policy prefix!"
            assert data["confidence"] == 1.0, "Expected mock confidence 1.0!"
        else:
            assert data["sources"] == [], "Expected empty sources for general question!"
            assert data["answer"] == "I can only answer questions about Zepto policies right now."
            assert data["confidence"] == 1.0

        results.append({
            "test_case": tc["category"],
            "query": tc["query"],
            "response": data
        })
        print()

    print("=" * 80)
    print("ALL SUPPORT ASSISTANT TEST CASES PASSED SUCCESSFULLY")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_tests()
