"""
Embedding model creation and management for the RAG application.

This module provides functions to create and cache HuggingFace embedding models
used for converting text into vector representations for semantic similarity search.

Functions:
    create_huggingface_embeddings(): Create a HuggingFace embedding model.
"""


from langchain_huggingface import HuggingFaceEmbeddings

# Import configuration constants for embedding model name
from config import HF_EMBEDDING_MODEL


def create_huggingface_embeddings() -> HuggingFaceEmbeddings:
    """
    Create and return a HuggingFace embedding model for semantic similarity search.

    This function instantiates the embedding model specified in config.HF_EMBEDDING_MODEL.
    The model is used to convert text chunks into vector representations for
    similarity-based retrieval in the FAISS vector store.

    Returns:
        HuggingFaceEmbeddings: An initialized HuggingFace embedding model
            ready to encode text into vectors.

    """

    # Create and return the embedding model with the configured model name
    return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
