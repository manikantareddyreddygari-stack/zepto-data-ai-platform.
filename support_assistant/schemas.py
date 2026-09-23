"""
Zepto Data & AI Platform - Module 3: Support Assistant
Pydantic Request and Response Schemas
"""

from typing import List
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="Customer inquiry string", example="What is the delivery fee for orders below 149?")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Grounded response to customer inquiry")
    sources: List[str] = Field(default_factory=list, description="List of document IDs cited for policy answers, empty for general questions")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
