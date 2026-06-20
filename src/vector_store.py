"""
Vector store creation and management for the RAG application.

This module handles the creation and management of FAISS vector stores.
It splits documents into chunks and creates vector embeddings for
efficient semantic similarity search during retrieval.

Functions:
    build_vector_store(): Create a FAISS vector store from PDF documents.
"""

# ============================================================================
# Imports
# ============================================================================

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import configuration for chunk processing parameters
from config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_SOURCE_DOCUMENTS

# Import the embedding and document loading functions
from src.embeddings import create_huggingface_embeddings
from src.document_loader import load_pdf_documents


# ============================================================================
# Vector Store Creation
# ============================================================================


def build_vector_store() -> tuple[FAISS, int, int]:
    """
    Load documents, split them into chunks, and build a FAISS vector store.

    This function performs the following steps:
    1. Load all documents from the documents directory (PDF, TXT, DOCX)
    2. Select a subset of documents (limited by MAX_SOURCE_DOCUMENTS)
    3. Split documents into overlapping chunks using RecursiveCharacterTextSplitter
    4. Create vector embeddings for each chunk using HuggingFace embeddings
    5. Build a FAISS vector store for efficient semantic similarity search

    The FAISS vector store allows fast retrieval of the most similar chunks
    to a query, which are then passed to the LLM for answer generation.

    Vector Store Details:
    - Uses FAISS (Facebook AI Similarity Search) for efficient indexing
    - Stores embeddings and chunk metadata (source, file_type)
    - Supports similarity search with .similarity_search() method
    - Can be converted to a retriever for use in RAG chains

    Returns:
        tuple[FAISS, int, int]: A tuple containing:
            - FAISS: The constructed vector store ready for retrieval
            - int: Number of documents loaded (up to MAX_SOURCE_DOCUMENTS)
            - int: Total number of chunks created after splitting

    Raises:
        ValueError: If no documents are found in the documents folder.

    Note:
        - Chunks have size CHUNK_SIZE with overlap of CHUNK_OVERLAP
        - Overlap helps preserve context at chunk boundaries
        - Vector store is created fresh each time (not cached/persisted)
        - For production, consider persisting the vector store to disk

    Examples:
        >>> vector_store, doc_count, chunk_count = build_vector_store()
        >>> print(f"Created index with {chunk_count} chunks from {doc_count} documents")
        >>> retriever = vector_store.as_retriever(search_kwargs={'k': 4})
    """

    # ========================================================================
    # Step 1: Load documents (PDF, TXT, DOCX) from the documents directory
    # ========================================================================
    documents = load_pdf_documents()

    # ========================================================================
    # Step 2: Limit documents to MAX_SOURCE_DOCUMENTS for efficiency
    # ========================================================================
    # This prevents processing too many documents which could exhaust memory
    # or take excessive time. Useful for large document collections.
    selected_documents: list[Document] = documents[:MAX_SOURCE_DOCUMENTS]

    # ========================================================================
    # Step 3: Split documents into overlapping chunks
    # ========================================================================
    # Use RecursiveCharacterTextSplitter to create semantically meaningful chunks
    # - It tries to split on common boundaries (paragraphs, sentences, words)
    # - CHUNK_SIZE: target size for each chunk in characters
    # - CHUNK_OVERLAP: number of overlapping characters between consecutive chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # Split all selected documents into chunks
    split_documents: list[Document] = text_splitter.split_documents(selected_documents)

    # ========================================================================
    # Step 4: Create embeddings and build the FAISS vector store
    # ========================================================================
    # FAISS.from_documents automatically handles:
    # - Converting each chunk to a vector embedding using HuggingFace embeddings
    # - Building the FAISS index for efficient similarity search
    # - Storing chunk metadata (source file, page number)
    vector_store = FAISS.from_documents(
        split_documents,
        create_huggingface_embeddings(),
    )

    # ========================================================================
    # Return Results
    # ========================================================================
    # Return the vector store and metadata about the indexing process
    return vector_store, len(selected_documents), len(split_documents)


# ============================================================================
# Vector Store Persistence (Save/Load)
# ============================================================================


def save_vector_store(
    vector_store: FAISS,
    store_name: str = "default_index",
) -> None:
    """
    Save a FAISS vector store to disk for persistence across sessions.

    This function persists the vector store index to the filesystem, allowing
    the expensive computation of vectorizing and indexing documents to be
    reused across multiple application runs. Without persistence, the vector
    store must be rebuilt every time the app starts.

    Persistence Benefits:
    - Skip expensive embedding computation on app restart
    - Multiple vector stores can be saved with different names
    - Easy to version document collections
    - Enables sharing vector stores between team members

    Storage Structure:
        vector_stores/
        ├── default_index/
        │   ├── index.pkl
        │   ├── index.faiss
        │   ├── docstore.pkl
        │   └── index_to_docstore_id.pkl
        └── research_v2_index/
            └── ...

    Args:
        vector_store (FAISS): The FAISS vector store to persist.
        store_name (str): Name identifier for this vector store.
            Default: "default_index"
            Used as the subdirectory name within vector_stores/

    Returns:
        None

    Raises:
        OSError: If unable to create vector_stores directory or write files.
        Exception: If FAISS save operation fails.

    Side Effects:
        - Creates VECTOR_STORE_DIRECTORY if it doesn't exist
        - Writes multiple files to disk (index.pkl, index.faiss, etc.)
        - May overwrite existing vector store with same name

    Note:
        - Non-blocking: saves immediately
        - Safe to call after building new vector store
        - Store name should be unique for different document collections
        - Saved indices are not portable between FAISS versions

    Examples:
        >>> from src.vector_store import build_vector_store, save_vector_store
        >>> vector_store, doc_count, chunk_count = build_vector_store()
        >>> save_vector_store(vector_store, "research_papers_v1")
        >>> # Vector store is now persisted to vector_stores/research_papers_v1/
    """

    # ========================================================================
    # Import Configuration and Logging
    # ========================================================================
    from config import VECTOR_STORE_DIRECTORY
    from src.logging_config import get_logger

    logger = get_logger()

    # ========================================================================
    # Create Vector Store Directory
    # ========================================================================
    # Ensure the vector_stores directory exists
    VECTOR_STORE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Determine Save Path
    # ========================================================================
    # Create a subdirectory for this specific vector store
    store_path = VECTOR_STORE_DIRECTORY / store_name

    # ========================================================================
    # Save Vector Store to Disk
    # ========================================================================
    try:
        # FAISS.save_local() saves the index and metadata to the directory
        vector_store.save_local(str(store_path))

        logger.info(
            f"Vector store saved successfully to {store_path} "
            f"({store_name} collection)"
        )

    except Exception as e:
        logger.error(f"Failed to save vector store to {store_path}: {str(e)}")
        raise


def load_vector_store(
    store_name: str = "default_index",
) -> FAISS | None:
    """
    Load a previously persisted FAISS vector store from disk.

    This function loads a saved vector store index from the filesystem,
    restoring the pre-computed embeddings and document indices. This enables
    fast startup and avoids re-embedding documents on each app run.

    When to Use:
    - App startup: Try loading saved index before rebuilding
    - Switching between collections: Load different indices on demand
    - Team collaboration: Share pre-computed indices

    Returns None if:
    - The vector store doesn't exist (normal on first run)
    - Load fails due to corruption or version mismatch

    Args:
        store_name (str): Name identifier of the vector store to load.
            Default: "default_index"
            Must match the name used in save_vector_store()

    Returns:
        FAISS | None: The loaded vector store, or None if not found or load failed.
            If not None, can be used immediately with as_retriever().

    Raises:
        None directly. Logs errors instead so app can gracefully fall back
        to rebuilding the vector store.

    Side Effects:
        - Reads files from disk
        - Logs success/warning messages
        - Does not modify any files

    Note:
        - Graceful degradation: app continues if load fails
        - Returns None if store doesn't exist (not an error on first run)
        - Loaded stores may be version-specific to FAISS library version
        - Fast operation compared to rebuild_vector_store()

    Examples:
        >>> from src.vector_store import load_vector_store
        >>> vector_store = load_vector_store("default_index")
        >>> if vector_store is not None:
        ...     retriever = vector_store.as_retriever()
        ... else:
        ...     print("Vector store not found, will rebuild")
    """

    # ========================================================================
    # Import Configuration and Logging
    # ========================================================================
    from config import VECTOR_STORE_DIRECTORY
    from src.logging_config import get_logger

    logger = get_logger()

    # ========================================================================
    # Determine Load Path
    # ========================================================================
    store_path = VECTOR_STORE_DIRECTORY / store_name

    # ========================================================================
    # Check if Vector Store Exists
    # ========================================================================
    if not store_path.exists():
        logger.info(
            f"Vector store not found at {store_path}. "
            f"Will need to rebuild with build_vector_store()."
        )
        return None

    # ========================================================================
    # Load Vector Store from Disk
    # ========================================================================
    try:
        # Load the vector store from the saved directory
        vector_store = FAISS.load_local(
            str(store_path),
            embeddings=create_huggingface_embeddings(),
        )

        logger.info(
            f"Vector store loaded successfully from {store_path} "
            f"({store_name} collection)"
        )

        return vector_store

    except Exception as e:
        logger.warning(
            f"Failed to load vector store from {store_path}: {str(e)}. "
            f"Will rebuild vector store instead."
        )
        return None
