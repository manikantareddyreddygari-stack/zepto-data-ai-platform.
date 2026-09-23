"""
Zepto Data & AI Platform - Module 3: Support Assistant
Vector Store: Document Ingestion, Embedding Generation (all-MiniLM-L6-v2), and ChromaDB Indexing
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glob
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import (
    DOCS_DIR,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_model_instance = None


def get_embedding_model() -> SentenceTransformer:
    """Singleton loader for local sentence-transformers model."""
    global _model_instance
    if _model_instance is None:
        logger.info(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}...")
        _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model_instance


def get_chroma_client() -> chromadb.ClientAPI:
    """Returns local persistent ChromaDB client."""
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def ingest_corpus() -> chromadb.Collection:
    """
    Loads all 8 documents from docs/, chunks them, embeds with all-MiniLM-L6-v2,
    and stores vectors in a persistent ChromaDB collection.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    doc_files = sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt")))
    logger.info(f"Found {len(doc_files)} corpus files in {DOCS_DIR}.")

    ids = []
    documents = []
    metadatas = []

    for file_path in doc_files:
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        ids.append(doc_id)
        documents.append(content)
        metadatas.append({"doc_id": doc_id, "source_file": os.path.basename(file_path)})

    model = get_embedding_model()
    embeddings = model.encode(documents).tolist()

    # Upsert to prevent duplicate entry errors
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    logger.info(f"Successfully ingested and indexed {len(ids)} documents in ChromaDB collection '{COLLECTION_NAME}'.")
    return collection


def retrieve_similar_chunks(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieves top_k most similar document chunks from ChromaDB using cosine similarity.
    Runs for real in both mock and real-LLM modes.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:
        logger.info("Collection is empty; running ingestion first...")
        ingest_corpus()

    model = get_embedding_model()
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    if results and "ids" in results and results["ids"]:
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            text = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            # Convert cosine distance to cosine similarity (1 - distance)
            similarity = round(max(0.0, min(1.0, 1.0 - dist)), 4)
            chunks.append({
                "id": doc_id,
                "text": text,
                "metadata": meta,
                "distance": dist,
                "similarity": similarity
            })

    return chunks


if __name__ == "__main__":
    ingest_corpus()
    test_results = retrieve_similar_chunks("delivery charges and pin codes", top_k=2)
    print("Test Retrieval Output:")
    for res in test_results:
        print(f"ID: {res['id']}, Similarity: {res['similarity']}\nSnippet: {res['text'][:120]}...\n")
