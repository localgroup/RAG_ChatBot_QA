"""
Advanced retrieval options for the RAG Document Q&A application.

This module provides enhanced retrieval capabilities beyond simple semantic search:
- Hybrid search: Combine semantic (embedding-based) with BM25 (keyword-based) search
- Metadata filtering: Filter documents by source, date, or other metadata
- Result reranking: Re-score retrieved results for better relevance

Functions:
    hybrid_search(): Perform hybrid semantic + BM25 search.
    filter_by_metadata(): Filter results using document metadata.
    rerank_results(): Re-score results using relevance models.
"""


from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


# Imports - Configuration
from config import (
    ENABLE_HYBRID_SEARCH,
    SEMANTIC_WEIGHT,
    BM25_WEIGHT,
    ENABLE_METADATA_FILTERING,
    ENABLE_RERANKING,
    RETRIEVAL_K,
)


# Imports - Utilities
from src.logging_config import get_logger


# Hybrid Search (Semantic + BM25)
def create_hybrid_retriever(
    vector_store: Any,
    documents: list[Document],
) -> Any:
    """
    Create a hybrid retriever combining semantic and BM25 keyword search.

    Hybrid search improves retrieval quality by combining two complementary
    search paradigms:

    1. Semantic Search (70% by default):
       - Uses embeddings to understand query meaning
       - Good for conceptual queries: "What is attention?"
       - Finds semantically similar content

    2. BM25 Keyword Search (30% by default):
       - Traditional TF-IDF based keyword matching
       - Good for specific terms: "transformer architecture"
       - Finds exact keyword matches

    Real-World Example:
        Query: "Attention mechanism in NLP"
        - Semantic: Finds papers discussing attention conceptually
        - BM25: Finds papers with exact words "attention", "mechanism", "NLP"
        - Combined: Best of both worlds

    Args:
        vector_store (Any): FAISS or Chroma vector store with embeddings.
        documents (list[Document]): List of documents for BM25 indexing.

    Returns:
        Any: A hybrid retriever that combines both search methods.
            Use with retriever.get_relevant_documents(query)

    Note:
        - BM25 weights configured in config.py
        - Requires both embedding vectors and raw document text
        - Slightly slower than single-method retrieval but higher quality
        - Weights can be tuned based on use case

    """

    # Validate Configuration
    if not ENABLE_HYBRID_SEARCH:
        # If hybrid search is disabled, return just semantic retriever
        return vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    logger = get_logger()

    # Create Semantic Retriever
    # Use vector store embeddings for semantic search
    semantic_retriever = vector_store.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )

    # Create BM25 Retriever
    # Initialize BM25 retriever from raw documents
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = RETRIEVAL_K

    # Create Ensemble Retriever (Hybrid)
    # Combine both retrievers with configurable weights
    try:
        from langchain.retrievers import EnsembleRetriever

        hybrid_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[SEMANTIC_WEIGHT, BM25_WEIGHT],
        )

        logger.info(
            f"Hybrid retriever created: "
            f"Semantic {SEMANTIC_WEIGHT*100}% + BM25 {BM25_WEIGHT*100}%"
        )

        return hybrid_retriever

    except ImportError:
        # Fallback if EnsembleRetriever not available
        logger.warning(
            "EnsembleRetriever not available; using semantic-only retriever"
        )
        return semantic_retriever



# Metadata Filtering
def filter_by_metadata(
    documents: list[Document],
    metadata_filter: dict[str, Any] | None = None,
) -> list[Document]:
    """
    Filter retrieved documents by metadata criteria.

    Args:
        documents (list[Document]): List of documents to filter.
        metadata_filter (dict | None): Filtering criteria.
            Format: {"field": "value"} for exact match
                   {"field": {"$op": value}} for operators ($gt, $lt, $gte, $lte, etc.)
            If None, returns all documents unchanged.

    Returns:
        list[Document]: Filtered subset of documents matching criteria.
            Returns original list if filter is None or empty.

    Note:
        - Metadata must exist in document metadata dicts
        - Missing metadata fields are treated as False for filters
        - Useful as pre-filtering before vector search
        - Can significantly reduce search space for large corpora

    """

    # Validate Configuration
    if not ENABLE_METADATA_FILTERING or metadata_filter is None:
        # Filtering disabled or no filter specified; return all
        return documents

    logger = get_logger()

    # Apply Metadata Filters
    filtered_documents = []

    for document in documents:
        # Get document metadata
        doc_metadata = document.metadata or {}

        # Check if document matches all filter criteria
        matches = True

        for field, filter_value in metadata_filter.items():
            # Handle simple value match
            if isinstance(filter_value, str) or isinstance(filter_value, int):
                if doc_metadata.get(field) != filter_value:
                    matches = False
                    break

            # Handle operator-based filters ($gt, $lt, etc.)
            elif isinstance(filter_value, dict):
                doc_value = doc_metadata.get(field)

                # $gt: greater than
                if "$gt" in filter_value:
                    if not (doc_value > filter_value["$gt"]):
                        matches = False
                        break

                # $gte: greater than or equal
                if "$gte" in filter_value:
                    if not (doc_value >= filter_value["$gte"]):
                        matches = False
                        break

                # $lt: less than
                if "$lt" in filter_value:
                    if not (doc_value < filter_value["$lt"]):
                        matches = False
                        break

                # $lte: less than or equal
                if "$lte" in filter_value:
                    if not (doc_value <= filter_value["$lte"]):
                        matches = False
                        break

        # Include document if all criteria matched
        if matches:
            filtered_documents.append(document)

    logger.info(
        f"Metadata filtering: {len(documents)} documents "
        f"→ {len(filtered_documents)} results"
    )

    return filtered_documents


# ============================================================================
# Result Reranking
# ============================================================================


def rerank_results(
    query: str,
    documents: list[Document],
    top_k: int | None = None,
) -> list[Document]:
    """
    Re-score and rerank retrieved documents for better relevance.

    Reranking is a two-stage retrieval process:

    1. **First stage (Retrieval)**: Get broad set of candidates (e.g., top-50)
       Uses fast retrieval: embeddings, BM25, etc.

    2. **Second stage (Reranking)**: Re-score candidates more accurately
       Uses more accurate but slower method

    Benefits:
    - Better relevance: Can catch mistakes from initial retrieval
    - Handles edge cases: Query ambiguity, domain-specific relevance
    - Quality improvement: 10-30% relevance boost typical
    - Latency tradeoff: Slower (milliseconds) but accuracy matters

    This function implements reranking using semantic similarity with
    the original query as a ranking signal. More advanced methods
    could use cross-encoders or specialized reranking models.

    Args:
        query (str): The original query string.
        documents (list[Document]): Retrieved documents to rerank.
        top_k (int | None): Return top-k reranked results.
            If None, uses RERANK_TOP_K from config.py
            Returns all documents if None and ENABLE_RERANKING is False.

    Returns:
        list[Document]: Reranked documents, optionally limited to top-k.
            Sorted by relevance (highest first).

    Note:
        - Current implementation uses embedding similarity
        - Could be enhanced with cross-encoders for better relevance
        - Reranking disabled if ENABLE_RERANKING is False in config
        - Adds latency; use only when quality is critical

    Examples:
        >>> from src.retrieval import rerank_results
        >>> retrieved_docs = retriever.get_relevant_documents(query)
        >>> reranked_docs = rerank_results(query, retrieved_docs, top_k=5)
        >>> print(f"Top doc score improved by X%")
    """

    # ========================================================================
    # Validate Configuration
    # ========================================================================
    if not ENABLE_RERANKING or not documents:
        # Reranking disabled or no documents to rerank
        return documents

    if top_k is None:
        from config import RERANK_TOP_K
        top_k = RERANK_TOP_K

    logger = get_logger()

    # ========================================================================
    # Calculate Relevance Scores
    # ========================================================================
    # Score each document based on semantic similarity to query
    # Using embedding-based scoring; could be replaced with cross-encoders

    try:
        from src.embeddings import create_huggingface_embeddings

        embeddings = create_huggingface_embeddings()

        # Get query embedding
        query_embedding = embeddings.embed_query(query)

        # Score each document
        scored_documents = []

        for document in documents:
            # Get document embedding
            doc_embedding = embeddings.embed_query(document.page_content)

            # Calculate cosine similarity
            import numpy as np

            query_vec = np.array(query_embedding)
            doc_vec = np.array(doc_embedding)

            similarity = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )

            scored_documents.append((document, similarity))

    except Exception as e:
        logger.warning(f"Reranking failed: {str(e)}; returning original order")
        return documents[:top_k]

    # ========================================================================
    # Sort by Relevance Score and Limit to Top-K
    # ========================================================================
    # Sort descending by similarity score
    sorted_docs = sorted(scored_documents, key=lambda x: x[1], reverse=True)

    # Extract just the documents (not scores)
    reranked_documents = [doc for doc, _ in sorted_docs[:top_k]]

    logger.info(
        f"Reranked {len(documents)} documents → top {len(reranked_documents)}"
    )

    return reranked_documents
