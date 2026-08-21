"""
Utility functions for the RAG Document Q&A application.

This module provides helper functions for:
- Session state management
- Chat history conversion between formats
- Source metadata extraction and formatting
- Error handling and validation

Functions:
    validate_configuration(): Validate required environment and files.
    initialize_session_state(): Initialize Streamlit session keys.
    build_chat_history_messages(): Convert session messages to LangChain format.
    build_source_payload(): Extract document metadata for UI display.
"""

# Imports

import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

# Import configuration for directory path
from config import DOCUMENT_DIRECTORY


# Environment and Configuration Validation

def validate_configuration() -> None:
    """
    Validate required environment variables and project files.

    This function checks that:
    1. GROQ_API_KEY environment variable is set (required for LLM)
    2. DOCUMENT_DIRECTORY exists and contains documents

    If validation fails, an error message is displayed in Streamlit and
    execution is halted with st.stop().

    Raises:
        None directly, but calls st.error() and st.stop() on validation failure.

    Side Effects:
        - May call st.error() to display error messages
        - May call st.stop() to halt Streamlit execution

    Note:
        - GROQ_API_KEY must be set before calling LLM functions
        - Typically called at the start of helper functions that need these resources
        - Environment variables can be set via .env file or export commands
    """

    # Check for required environment variables
    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing required environment variable: GROQ_API_KEY")
        st.stop()

    # Check that DOCUMENT_DIRECTORY exists
    if not DOCUMENT_DIRECTORY.exists():
        st.error(f"Documents directory not found: {DOCUMENT_DIRECTORY}")
        st.stop()


# Streamlit Session State Management

def initialize_session_state() -> None:
    """
    Initialize Streamlit session state keys used by the threaded chat app.

    Streamlit re-runs the entire script on each widget interaction. To persist
    data across interactions, we use st.session_state. This function initializes
    all keys with sensible defaults.

    Initialized Keys:
        - vector_store (FAISS | None): The built vector store, or None if not created
        - loaded_document_count (int): Number of PDFs loaded into the vector store
        - chunk_count (int): Number of text chunks created from PDFs
        - messages (list): Chat history as list of dicts with 'role' and 'content'
        - uploaded_vector_store (FAISS | None): Vector store for user uploads
        - uploaded_files (list): Metadata about uploaded files
        - using_uploads (bool): Whether to use uploaded or base vector store
        - upload_chunk_count (int): Number of chunks from uploaded documents
        - upload_session_size (float): Total MB of uploads in current session

    Since st.session_state.setdefault() only sets if the key doesn't exist,
    calling this multiple times is safe—it won't overwrite existing data.

    Side Effects:
        - Modifies st.session_state directly
        - Called once per Streamlit session at app startup

    Note:
        - messages stores dicts (serializable) rather than LangChain objects
        - Messages have format: {"role": "user"|"assistant", "content": str, ...}
        - Vector store is cached in session to avoid rebuilding on every interaction
        - Upload vector store is kept separate for flexible switching
    """

    # Initialize default values for session keys if they don't exist
    defaults = {
        # Base documents
        "vector_store": None,
        "loaded_document_count": 0,
        "chunk_count": 0,
        # Chat history
        "messages": [],
        # Upload documents
        "uploaded_vector_store": None,
        "uploaded_files": [],
        "using_uploads": False,
        "upload_chunk_count": 0,
        "upload_session_size": 0.0,
    }
    
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# Chat History Conversion

def build_chat_history_messages() -> list[HumanMessage | AIMessage]:
    """
    Convert stored chat turns into LangChain chat history message objects.

    The session stores messages as dicts for serialization and UI rendering.
    The LLM requires LangChain Message objects (HumanMessage, AIMessage, etc.)
    for the chat_history input parameter. This function performs the conversion.

    Message Format Conversion:
        Dict format (session storage):
            {"role": "user", "content": "What is X?"}
            {"role": "assistant", "content": "X is ..."}

        LangChain format (LLM input):
            HumanMessage(content="What is X?")
            AIMessage(content="X is ...")

    Returns:
        list[HumanMessage | AIMessage]: A list of LangChain message objects
            representing the chat history in chronological order.
            Returns empty list if no messages are stored.

    Note:
        - Preserves message order from st.session_state.messages
        - Ignores message entries without 'role' or 'content' keys
        - Used by create_rag_chain() to provide history context for retrieval
    """

    # Initialize empty list for LangChain message objects
    history: list[HumanMessage | AIMessage] = []

    # Iterate through stored messages and convert to LangChain format
    for message in st.session_state.messages:
        # Extract the role (user or assistant) and content from the dict
        role = message.get("role")
        content = message.get("content", "")

        # Convert to appropriate LangChain message type based on role
        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))

    # Return the list of LangChain message objects
    return history


# Source Document Metadata Extraction

def build_source_payload(response_context: list[Document]) -> list[dict[str, Any]]:
    """
    Extract and format retrieval metadata from retrieved document chunks.

    The RAG chain returns retrieved documents with metadata. This function
    extracts source filename, page number, and chunk content, formatting them
    for display in the Streamlit UI (e.g., in an expandable "Sources" section).

    Input (from RAG chain):
        response_context: list of LangChain Document objects with:
            - page_content (str): The chunk text
            - metadata (dict): Including 'source' (file path) and 'page' (page num)

    Output:
        Serializable metadata dict suitable for st.write() display.

    Returns:
        list[dict[str, Any]]: A list of dicts with keys:
            - "name" (str): Just the filename (not full path)
            - "page" (str): Page number from metadata
            - "content" (str): The chunk text

    Note:
        - Handles missing metadata gracefully (uses "Unknown" or "N/A")
        - Extracts just the filename from the full path for cleaner UI
        - Preserves order of returned documents
    """

    # Initialize empty list for source data
    sources: list[dict[str, Any]] = []

    # Iterate through retrieved documents and extract metadata
    for document in response_context:
        # Extract source file path and get just the filename
        source_path = document.metadata.get("source", "Unknown")
        source_name = Path(source_path).name  # Extracts just the filename

        # Extract page number (or "N/A" if not available)
        page_number = document.metadata.get("page", "N/A")

        # Build the source metadata dict for UI display
        sources.append(
            {
                "name": source_name,
                "page": page_number,
                "content": document.page_content,
            }
        )

    # Return the formatted sources list
    return sources


# Rate Limit Error Formatting

def format_rate_limit_error(error_message: str) -> str:
    """
    Extract key info from Groq rate limit error and format for UI display.
    
    Extracts token usage limits and retry time from error message string,
    showing only helpful information without sensitive IDs or raw data.
    
    Args:
        error_message (str): The full error message from Groq API as a string
    
    Returns:
        str: Formatted markdown string with user-friendly error info
    """
    import re
    
    # Extract key numbers from error message (string)
    limit_match = re.search(r'Limit (\d+)', error_message)
    used_match = re.search(r'Used (\d+)', error_message)
    requested_match = re.search(r'Requested (\d+)', error_message)
    retry_match = re.search(r'Please try again in ([\d.]+)s', error_message)
    
    # Start building formatted message
    formatted = "**Rate Limit Exceeded**\n\n"
    formatted += "Code: `rate_limit_exceeded`\n\n"
    
    # Add token usage info if available
    if limit_match and requested_match:
        formatted += (
            f"📊 **Token Usage:**\n"
            f"- Limit: {limit_match.group(1)} tokens/min\n"
            f"- Requested: {requested_match.group(1)} tokens\n"
        )
        if used_match:
            formatted += f"- Used: {used_match.group(1)} tokens\n"
        formatted += "\n"
    
    # Add retry time if available
    if retry_match:
        formatted += f"⏳ **Wait {retry_match.group(1)} seconds** before retrying\n\n"
    
    # Add helpful tips
    formatted += (
        "💡 **What you can do:**\n"
        "- Ask shorter, more concise questions\n"
        "- Reduce your message size and try again\n"
        "- Try again after the wait time has passed\n"
    )
    
    return formatted
