"""
Zepto Data & AI Platform - Module 3: Support Assistant
Configuration and Structured Prompt Templates
"""

import os

# Gating toggle: MOCK_LLM defaults to 1 (graded baseline) unless explicitly set to 0
MOCK_LLM = os.getenv("MOCK_LLM", "1").strip().lower() in ("1", "true", "yes")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
COLLECTION_NAME = "zepto_policy_corpus"

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours"
]

STRUCTURED_PROMPT_TEMPLATE = """
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
{{
  "answer": "Clear, grounded response string",
  "sources": ["doc_id1", "doc_id2"],
  "confidence": 0.95
}}

[LENGTH]
Keep the answer concise and direct (under 150 words).

[FEW-SHOT EXAMPLE]
User Query: "How much does Zepto Pass cost?"
Context:
[doc_03] Zepto offers three account tiers: Basic (free, default tier), Zepto Pass (INR 49 per month, free standard delivery on all orders), and Zepto Pass+ (INR 99 per month).
Output JSON:
{{
  "answer": "Zepto Pass costs INR 49 per month and includes free standard delivery on all orders as well as 5% off select categories.",
  "sources": ["doc_03"],
  "confidence": 1.0
}}
"""
