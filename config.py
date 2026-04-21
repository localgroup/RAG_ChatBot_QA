"""
Configuration and constants for the RAG Document Q&A application.

This module centralizes all configuration parameters, model names, and
directory paths used throughout the application.

Attributes:
    PROJECT_ROOT (Path): The root directory of this project, resolved from this file's location.
    DOCUMENT_DIRECTORY (Path): Directory containing the documents to index.
    GROQ_MODEL (str): The Groq LLM model identifier for generating answers.
    HF_EMBEDDING_MODEL (str): The HuggingFace embedding model used for document indexing.
    MAX_SOURCE_DOCUMENTS (int): Maximum number of documents to process in each build.
    CHUNK_SIZE (int): Size of text chunks for splitting documents (in characters).
    CHUNK_OVERLAP (int): Overlap between consecutive chunks to preserve context.
    RETRIEVAL_K (int): Number of top similar chunks to retrieve for each query.
"""

from pathlib import Path


# Directory Configuration
# Resolve the root directory of this project (directory containing this file)
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Path to the directory containing documents to be indexed
DOCUMENT_DIRECTORY: Path = PROJECT_ROOT / "documents"


# Language Model Configuration
# Groq LLM model identifier used for generating answers
GROQ_MODEL: str = "llama-3.1-8b-instant"

# Temperature parameter for Groq LLM (0 = deterministic, higher = more creative)
GROQ_TEMPERATURE: float = 0.7


# Embedding Model Configuration
# HuggingFace embedding model used for content-based similarity search
# all-MiniLM-L6-v2 is lightweight and works well for semantic search
HF_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"


# Document Processing Configuration
# Maximum number of documents to process during vector store creation
# Limits processing time and memory usage for large document collections
MAX_SOURCE_DOCUMENTS: int = 50

# Size of text chunks (in characters) when splitting documents
# Larger chunks = more context per retrieval but fewer chunks total
CHUNK_SIZE: int = 2000

# Overlap between consecutive chunks (in characters)
# Overlap preserves context at chunk boundaries and improves retrieval quality
CHUNK_OVERLAP: int = 200


# Retrieval Configuration
# Number of most similar chunks (k) to retrieve for each query
# Higher k = more context but potentially more noise in the answer
RETRIEVAL_K: int = 4


# Chat Configuration
# Application title displayed in the Streamlit UI
APP_TITLE: str = "RAG Document Q&A With Groq and Hugging Face Embeddings"

# Application subtitle/description shown under the title
APP_CAPTION: str = (
    "Chat-style application for querying documents."
    "Successive turns stay visible and previous turns are reused as context."
)


# Streaming Configuration
# Enable streaming responses for real-time LLM output
ENABLE_STREAMING: bool = True

# Maximum number of tokens to stream at once (higher = faster but less chunked)
STREAMING_CHUNK_SIZE: int = 50


# Vector Store Persistence Configuration
# Directory where vector store indices are persisted to disk
VECTOR_STORE_DIRECTORY: Path = PROJECT_ROOT / "vector_stores"

# Filename for the default vector store index
DEFAULT_VECTOR_STORE_NAME: str = "default_index"

# Enable automatic persistence of vector store to disk
AUTO_PERSIST_VECTOR_STORE: bool = True


# Logging Configuration
# Directory where log files are stored
LOGS_DIRECTORY: Path = PROJECT_ROOT / "logs"

# Log file name
LOG_FILE_NAME: str = "rag_app.log"

# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL: str = "INFO"

# Maximum log file size in bytes (10MB)
MAX_LOG_FILE_SIZE: int = 10 * 1024 * 1024

# Number of backup log files to keep
LOG_BACKUP_COUNT: int = 5

# Enable console logging in addition to file logging
ENABLE_CONSOLE_LOGGING: bool = True


# Caching Configuration
# Enable result caching for frequently asked questions
ENABLE_RESULT_CACHING: bool = True

# Cache time-to-live in seconds (3600 = 1 hour)
CACHE_TTL_SECONDS: int = 3600

# Maximum number of cached results to keep
MAX_CACHED_RESULTS: int = 100


# Advanced Retrieval Configuration
# Enable hybrid search (semantic + BM25 keyword search)
ENABLE_HYBRID_SEARCH: bool = True

# Weight for semantic search in hybrid search (0-1, remainder goes to BM25)
SEMANTIC_WEIGHT: float = 0.7

# BM25 weight in hybrid search (automatically 1 - SEMANTIC_WEIGHT)
BM25_WEIGHT: float = 1.0 - SEMANTIC_WEIGHT

# Enable metadata filtering in retrieval
ENABLE_METADATA_FILTERING: bool = True

# Enable result reranking for better relevance
ENABLE_RERANKING: bool = True

# Number of results to rerank (retrieve more initially, rerank top N)
RERANK_TOP_K: int = 10


# Document Upload Configuration
# Enable document upload feature for dynamic file processing
ENABLE_DOCUMENT_UPLOAD: bool = True

# Maximum file size for uploads in MB (25 MB default)
MAX_UPLOAD_FILE_SIZE_MB: int = 25

# Maximum total size for all uploaded files in a session (50 MB)
MAX_TOTAL_UPLOAD_SIZE_MB: int = 50

# Allowed file extensions for upload (comma-separated)
ALLOWED_FILE_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt", ".docx")

# Directory for temporarily storing uploaded files
UPLOAD_TEMP_DIRECTORY: Path = PROJECT_ROOT / "uploads_temp"

# Automatic cleanup: delete upload temp files on app restart
AUTO_CLEANUP_UPLOADS: bool = True

# Maximum pages to extract from uploaded documents (prevents extremely large files)
MAX_PAGES_PER_UPLOAD: int = 100

# Use separate vector store for uploaded documents (keep original indexed)
SEPARATE_UPLOAD_VECTOR_STORE: bool = True

# Vector store name suffix for uploaded documents
UPLOAD_VECTOR_STORE_SUFFIX: str = "_user_uploads"
