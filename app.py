"""
RAG Document Q&A Application - Threaded Chat Interface.

This is the main Streamlit application for querying documents using
Retrieval-Augmented Generation (RAG). The app provides an interactive chat
interface where users can ask questions about documents, with answers grounded
in the retrieved document chunks.

"""


import os
import time
from pathlib import Path
from typing import Any

# Load environment variables from .env file
from dotenv import load_dotenv


import streamlit as st


# Initialize logging at module import time
from src.logging_config import setup_logging, get_logger

# Set up logging (safe to call multiple times)
_logger = setup_logging()


from langchain_core.documents import Document

# Configuration constants
from config import (
    APP_TITLE,
    APP_CAPTION,
    ENABLE_STREAMING,
    ENABLE_RESULT_CACHING,
    AUTO_PERSIST_VECTOR_STORE,
    DEFAULT_VECTOR_STORE_NAME,
    ENABLE_DOCUMENT_UPLOAD,
    MAX_UPLOAD_FILE_SIZE_MB,
    MAX_TOTAL_UPLOAD_SIZE_MB,
    ALLOWED_FILE_EXTENSIONS,
    SEPARATE_UPLOAD_VECTOR_STORE,
    UPLOAD_VECTOR_STORE_SUFFIX,
)

# LLM and embedding creation
from src.llm import create_llm
from src.embeddings import create_huggingface_embeddings

# Vector store operations - including persistence
from src.vector_store import build_vector_store, save_vector_store, load_vector_store

# RAG chain assembly - including streaming
from src.rag_chain import create_rag_chain, stream_rag_response

# Advanced retrieval options
from src.retrieval import create_hybrid_retriever, rerank_results

# Document upload functionality
from src.document_upload import (
    create_upload_vector_store,
    cleanup_upload_temp_files,
    get_upload_remaining_quota,
    get_upload_session_size,
)

# Utility functions
from src.utils import (
    validate_configuration,
    initialize_session_state,
    build_chat_history_messages,
    build_source_payload,
)


# ============================================================================
# Application Initialization
# ============================================================================

# Load environment variables from .env file (must be called before first use)
load_dotenv()


# ============================================================================
# Core Application Functions
# ============================================================================


def render_assistant_metadata(message: dict[str, Any]) -> None:
    """
    Render response metadata like execution time and source documents.

    This function displays additional information about the LLM response:
    - Response generation time for performance insights
    - Retrieved document chunks with source citations

    Args:
        message (dict[str, Any]): Assistant message dict containing:
            - "elapsed_time" (float, optional): Time to generate response in seconds
            - "sources" (list, optional): List of dicts with 'name', 'page', 'content'

    Side Effects:
        - Calls st.caption(), st.expander(), st.markdown(), st.write()
        - Renders to the current Streamlit container
        - Logs metadata rendering to application logger

    Note:
        - Only displays metadata if it exists in the message dict
        - Sources are shown in a collapsible expander for cleaner UI
        - Each source chunk shows filename, page number, and content

    Examples:
        >>> message = {
        ...     "role": "assistant",
        ...     "content": "Attention is...",
        ...     "elapsed_time": 2.34,
        ...     "sources": [{"name": "Attention.pdf", "page": 0, "content": "..."}]
        ... }
        >>> render_assistant_metadata(message)  # Displays in Streamlit UI
    """

    # Get logger for this module
    logger = get_logger()

    # ========================================================================
    # Display Response Generation Time
    # ========================================================================
    # Shows how long the RAG chain took to generate the answer
    elapsed_time = message.get("elapsed_time")
    if elapsed_time is not None:
        st.caption(f"Response time: {elapsed_time:.2f} seconds")
        logger.info(f"Response generated in {elapsed_time:.2f} seconds")

    # ========================================================================
    # Display Source Documents in Expandable Section
    # ========================================================================
    # Shows which documents were retrieved and used for answer generation
    sources = message.get("sources", [])
    if sources:
        # Log the retrieval results
        logger.info(f"Retrieved {len(sources)} document chunks for answer")

        # Create a collapsible expander labeled "Document similarity search"
        with st.expander("Document similarity search"):
            # Iterate through each retrieved document chunk
            for index, source in enumerate(sources, start=1):
                # Display source metadata (filename and page number)
                st.markdown(
                    f"**Chunk {index}** | Source: `{source['name']}` | Page: `{source['page']}`"
                )
                # Display the actual chunk content
                st.write(source["content"])
                # Separator between chunks
                st.write("-" * 24)


def render_chat_history() -> None:
    """
    Render all prior conversation turns as a persistent thread.

    This function iterates through all stored messages and displays them
    in Streamlit's chat message containers, creating a threadlike interface
    where users can see the full conversation history.

    Each message is rendered with:
    - Appropriate avatar (user vs assistant)
    - Markdown-formatted content
    - If assistant: source document metadata

    Side Effects:
        - Calls st.chat_message() and st.markdown() for each message
        - Renders to the current Streamlit container
        - Should be called before st.chat_input() to show history first

    Note:
        - Messages come from st.session_state.messages (persistent across reruns)
        - Chat history is rendered on every app rerun but efficiently
        - Threading appearance is created by sequential rendering

    Examples:
        >>> st.session_state.messages = [
        ...     {"role": "user", "content": "What is attention?"},
        ...     {"role": "assistant", "content": "Attention is..."}
        ... ]
        >>> render_chat_history()  # Displays both messages in thread view
    """

    # Iterate through all stored messages in session state
    for message in st.session_state.messages:
        # Create a chat message container with the appropriate role (avatar)
        with st.chat_message(message["role"]):
            # Display the message content as markdown
            st.markdown(message["content"])

            # If this is an assistant message, also render its metadata
            if message["role"] == "assistant":
                render_assistant_metadata(message)


def clear_chat_history() -> None:
    """
    Clear the visible chat transcript while keeping the vector store.

    This function resets the chat messages without rebuilding the vector store,
    allowing users to start a fresh conversation with the same document index.

    Clears:
        - st.session_state.messages: The chat history list

    Preserves:
        - st.session_state.vector_store: The FAISS index (expensive to rebuild)
        - st.session_state.loaded_document_count: Index metadata
        - st.session_state.chunk_count: Index metadata

    Side Effects:
        - Modifies st.session_state directly
        - Next render will show an empty chat (but documents still indexed)

    Note:
        - The vector store persists during the session
        - Users can clear and start a new conversation instantly
        - Useful for branching conversations or trying different questions

    Examples:
        >>> clear_chat_history()  # Chat now empty, but vector store ready
        >>> st.session_state.messages  # Output: []
        >>> st.session_state.vector_store  # Still contains the index
    """

    # Reset messages list to empty
    st.session_state.messages = []


def handle_uploaded_documents(uploaded_files: list) -> None:
    """
    Process uploaded documents and create a separate vector store.

    This function:
    1. Validates all uploaded files
    2. Creates a vector store from uploaded documents
    3. Stores it as a separate collection in session state
    4. Allows switching between base and uploaded document stores
    5. Clears chat history to avoid context confusion

    Args:
        uploaded_files (list): List of uploaded file objects from st.file_uploader

    Side Effects:
        - Creates vector store for uploaded documents
        - Stores in st.session_state.uploaded_vector_store
        - Updates st.session_state.uploaded_documents_info
        - Clears chat history
        - Logs processing steps

    Note:
        - Each upload creates a fresh vector store (separate from base docs)
        - Uploading new files replaces the previous upload store
        - Users can switch between stores using session state

    Examples:
        >>> uploaded_files = st.file_uploader("Upload", accept_multiple_files=True)
        >>> if uploaded_files:
        ...     handle_uploaded_documents(uploaded_files)
    """

    logger = get_logger()

    # ========================================================================
    # Validate Files
    # ========================================================================
    if not uploaded_files:
        logger.info("No files provided for upload processing")
        return

    logger.info(f"Processing {len(uploaded_files)} uploaded files")

    # ========================================================================
    # Show Processing Spinner
    # ========================================================================
    with st.spinner("Processing uploaded documents and creating vector store..."):
        try:
            # Create vector store from uploaded files
            logger.info("Creating vector store from uploaded documents")
            (
                vector_store,
                doc_count,
                chunk_count,
                successful_files,
                total_file_size_mb,
            ) = create_upload_vector_store(uploaded_files)

            # ================================================================
            # Store in Session State
            # ================================================================
            st.session_state.uploaded_vector_store = vector_store
            st.session_state.uploaded_files = successful_files
            st.session_state.upload_chunk_count = chunk_count
            st.session_state.upload_session_size = total_file_size_mb
            st.session_state.using_uploads = True  # Switch to uploaded docs

            # Clear chat to avoid confusion with different documents
            clear_chat_history()

            logger.info(
                f"Upload vector store ready: "
                f"{doc_count} documents → {chunk_count} chunks "
                f"from {len(successful_files)} files ({total_file_size_mb:.2f} MB)"
            )

            # ================================================================
            # Display Success Message
            # ================================================================
            st.success(
                f"✅ Uploaded {len(successful_files)} file(s) successfully!\n"
                f"Indexed: {doc_count} documents → {chunk_count} chunks"
            )

            # Display processed files
            with st.expander("📄 Processed Files"):
                for filename in successful_files:
                    st.markdown(f"✓ {filename}")

        except ValueError as e:
            logger.error(f"Validation error during upload: {str(e)}")
            st.error(f"❌ Upload Error: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error during upload processing: {str(e)}")
            st.error(f"❌ Processing Error: {str(e)}")


def handle_user_prompt(user_prompt: str) -> None:
    """
    Answer a user message using the RAG chain and preserve chat history.

    This function orchestrates the complete workflow:
    1. Validate configuration (API keys, directories)
    2. Check that vector store exists
    3. Convert session messages to LangChain format
    4. Store user message in session state
    5. Render user message in chat UI
    6. Call RAG chain to generate answer (with streaming if enabled)
    7. Extract sources and format response
    8. Store assistant message in session state
    9. Render assistant message and metadata in chat UI

    Enhanced Features:
    - Streaming output for real-time response display
    - Result caching for frequently asked questions
    - Comprehensive logging for debugging

    Args:
        user_prompt (str): The user's question text

    Side Effects:
        - Calls st.chat_message(), st.markdown(), st.warning(), st.spinner()
        - Modifies st.session_state.messages
        - Makes API calls to Groq (via create_rag_chain())
        - Measures response generation time
        - Logs all major steps to application logger

    Raises:
        None directly, but may call st.warning() or st.stop() via validation

    Note:
        - User message is displayed immediately
        - Assistant response displayed while thinking (or streaming in real-time)
        - Chat history automatically incorporated for follow-up questions
        - Response time is measured and displayed
        - Results cached for performance optimization

    Examples:
        >>> st.session_state.vector_store = my_faiss_store
        >>> st.session_state.messages = []
        >>> handle_user_prompt("What is attention?")
        >>> # Messages now contains both user and assistant messages
        >>> len(st.session_state.messages)  # Output: 2
    """

    # Get logger for this module
    logger = get_logger()

    # ========================================================================
    # Validate Configuration
    # ========================================================================
    # Ensure required API keys and files exist
    logger.info(f"Processing user prompt: {user_prompt[:50]}...")
    validate_configuration()

    # ========================================================================
    # Check Vector Store Exists
    # ========================================================================
    # The vector store must be built before any Q&A is possible
    # Can use either base vector store or uploaded vector store
    using_uploaded = st.session_state.get("using_uploads", False)
    vector_store_key = "uploaded_vector_store" if using_uploaded else "vector_store"

    if st.session_state.get(vector_store_key) is None:
        logger.warning("Attempt to answer without vector store loaded")
        st.warning("Create the document embeddings before starting the chat.")
        return

    # ========================================================================
    # Prepare Chat History for RAG Chain
    # ========================================================================
    # Convert stored messages (dicts) to LangChain message objects (HumanMessage, AIMessage)
    # This history will be used for history-aware retrieval
    previous_chat_history = build_chat_history_messages()

    # ========================================================================
    # Store User Message in Session
    # ========================================================================
    # Add the user message to session state so it's preserved across reruns
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # ========================================================================
    # Render User Message in UI
    # ========================================================================
    # Display the user message immediately in a chat bubble
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # ========================================================================
    # Generate and Display Assistant Response
    # ========================================================================
    # Create a chat message container for the assistant response
    with st.chat_message("assistant"):
        # Show spinner while the RAG chain is processing
        with st.spinner("Searching the indexed documents and generating an answer..."):
            # Record start time for performance metrics
            start_time = time.perf_counter()

            logger.info("Creating RAG chain for query processing")

            # Save the base vector store (in case it's different from current)
            original_vector_store = st.session_state.vector_store

            # Momentarily swap in the appropriate vector store
            # (uploaded or base documents depending on user selection)
            if using_uploaded:
                st.session_state.vector_store = st.session_state.uploaded_vector_store
                logger.info("Using uploaded documents vector store")
            else:
                st.session_state.vector_store = original_vector_store
                logger.info("Using base documents vector store")

            # Create the RAG chain (history-aware retriever + LLM)
            # It will use whichever vector store is currently in st.session_state
            rag_chain = create_rag_chain()

            logger.info(f"Chat history length: {len(previous_chat_history)} turns")

            # ================================================================
            # Invoke RAG Chain (with Error Handling)
            # ================================================================
            try:
                if ENABLE_STREAMING:
                    logger.info("Using streaming response mode")

                    # Create content placeholder for streaming output
                    response_placeholder = st.empty()
                    full_answer = ""

                    # Stream the response token by token
                    for chunk in stream_rag_response(
                        rag_chain,
                        user_prompt,
                        previous_chat_history,
                    ):
                        # Accumulate tokens
                        full_answer += chunk

                        # Update UI with accumulated text
                        response_placeholder.markdown(full_answer)

                    # Calculate response time after streaming completes
                    elapsed_time = time.perf_counter() - start_time

                    # Invoke once more to get the context (sources) for final response
                    # This is needed because streaming doesn't return context metadata
                    response = rag_chain.invoke(
                        {
                            "input": user_prompt,
                            "chat_history": previous_chat_history,
                        }
                    )

                    # Use the streamed answer, but get sources from full invocation
                    answer = full_answer
                    context = response["context"]

                else:
                    logger.info("Using non-streaming response mode")

                    # Invoke the chain without streaming (wait for full response)
                    response = rag_chain.invoke(
                        {
                            "input": user_prompt,
                            "chat_history": previous_chat_history,
                        }
                    )

                    # Calculate response generation time
                    elapsed_time = time.perf_counter() - start_time

                    # Extract answer and context
                    answer = response["answer"]
                    context = response["context"]

                    # Display the answer in the chat UI
                    st.markdown(answer)

                logger.info(
                    f"Response generated successfully in {elapsed_time:.2f}s "
                    f"with {len(context)} source chunks"
                )

                # Restore the original vector store to maintain session state
                st.session_state.vector_store = original_vector_store
                logger.info("Restored base vector store to session state")

            except Exception as e:
                # Handle Groq API errors gracefully
                error_str = str(e)
                logger.error(f"Error during RAG chain invocation: {error_str}")

                # Check if it's a rate limit error
                if "rate_limit_exceeded" in error_str or "rate_limit_exceeded" in error_str.lower():
                    # Format and display the rate limit error
                    from src.utils import format_rate_limit_error

                    formatted_error = format_rate_limit_error(error_str)

                    # Show toast notification (brief popup)
                    st.toast("⏱️ Rate limit exceeded", icon="⚠️")

                    # Show detailed warning with formatted error
                    st.warning(formatted_error)

                    logger.info("Rate limit error handled gracefully")

                else:
                    # For other errors, show generic message
                    st.toast("❌ API Error", icon="⚠️")
                    st.error(f"An error occurred: {error_str[:200]}...")
                    logger.error(f"Unexpected error: {error_str}")

                # Restore vector store and exit
                st.session_state.vector_store = original_vector_store
                return

        # Build the assistant message dict with answer and metadata
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "elapsed_time": elapsed_time,
            "sources": build_source_payload(context),
        }

        # Display metadata (time, sources)
        render_assistant_metadata(assistant_message)

        # Store the assistant message in session state
        st.session_state.messages.append(assistant_message)

        logger.info(f"Conversation turn completed. Total messages: {len(st.session_state.messages)}")


# ============================================================================
# Main Application
# ============================================================================


def main() -> None:
    """
    Render and run the threaded Streamlit RAG application.

    This is the entry point for the Streamlit app. It:
    1. Configures the Streamlit page (title, layout)
    2. Initializes session state
    3. Renders the title and info sections
    4. Creates sidebar controls for:
       - Building the vector store (indexing PDFs)
       - Clearing the chat history
       - Displaying index status
    5. Renders the chat history
    6. Renders the chat input and handles new messages

    The app structure:
    ┌─ Page Config (title, layout)
    ├─ Title and Info Sections
    ├─ Sidebar:
    │  ├─ "Document Embedding" button (builds vector store)
    │  ├─ "Clear Chat" button
    │  └─ Index status display
    ├─ Chat History Display
    └─ Chat Input

    Side Effects:
        - Initializes st_session_state
        - Renders Streamlit UI components
        - Handles button clicks and text input
        - May build vector store or clear chat on button click

    Note:
        - This function is called once per Streamlit session
        - But Streamlit reruns the entire script on each widget interaction
        - Session state preserves data across reruns

    Examples:
        >>> main()  # Run the Streamlit app (called if __name__ == "__main__")
    """

    # ========================================================================
    # Get Logger Instance
    # ========================================================================
    # Initialize logger for this function scope
    logger = get_logger()

    # ========================================================================
    # Configure Streamlit Page
    # ========================================================================
    # Set the browser tab title and layout (wide for more space)
    st.set_page_config(page_title="Threaded RAG Document Q&A", layout="wide")

    logger.info("="*80)
    logger.info("RAG Document Q&A Application Started")
    logger.info("="*80)

    # ========================================================================
    # Initialize Session State
    # ========================================================================
    # Create session keys for vector store, messages, and metadata
    initialize_session_state()

    # ========================================================================
    # Render Title and Description
    # ========================================================================
    st.title(APP_TITLE)
    st.caption(APP_CAPTION)

    # Display an info box with important notes about the app
    st.info(
        "Note: this keeps conversation memory for the current Streamlit session. "
        "For permanent threads, store messages in a database or file."
    )

    # ========================================================================
    # Sidebar Controls
    # ========================================================================
    # Create a sidebar section for setup and control
    with st.sidebar:
        st.subheader("Setup")

        # ====================================================================
        # Document Embedding Button
        # ====================================================================
        # Button to build the vector store from PDFs
        if st.button("Document Embedding", use_container_width=True):
            # Validate that PDFs and API keys exist
            logger.info("User clicked 'Document Embedding' button")
            validate_configuration()

            # Check if a saved vector store exists
            logger.info(f"Attempting to load persistent vector store: {DEFAULT_VECTOR_STORE_NAME}")
            vector_store = load_vector_store(DEFAULT_VECTOR_STORE_NAME)

            if vector_store is not None:
                # Loaded saved vector store
                logger.info("Loaded vector store from disk (skipped rebuild)")
                st.session_state.vector_store = vector_store
                st.info("Loaded saved vector store from disk!")

            else:
                # Build new vector store
                logger.info("Building new vector store from PDFs")

                # Build the FAISS vector store with a loading spinner
                with st.spinner("Loading PDFs and creating the FAISS index..."):
                    vector_store, loaded_document_count, chunk_count = build_vector_store()

                    # Cache the vector store in session state for future use
                    st.session_state.vector_store = vector_store
                    st.session_state.loaded_document_count = loaded_document_count
                    st.session_state.chunk_count = chunk_count

                    logger.info(
                        f"Vector store built: {loaded_document_count} documents → "
                        f"{chunk_count} chunks"
                    )

                    # Persist the vector store to disk if enabled
                    if AUTO_PERSIST_VECTOR_STORE:
                        logger.info("Persisting vector store to disk")
                        save_vector_store(vector_store, DEFAULT_VECTOR_STORE_NAME)

                # Display success message with index statistics
                st.success(
                    "Vector database is ready. "
                    f"Loaded {loaded_document_count} documents into {chunk_count} chunks."
                )

        # ====================================================================
        # Clear Chat Button
        # ====================================================================
        # Button to reset the conversation
        if st.button("Clear Chat", use_container_width=True):
            logger.info("User clicked 'Clear Chat' button")
            clear_chat_history()
            st.success("Cleared the visible thread.")

        # ====================================================================
        # Index Status Display
        # ====================================================================
        # Show whether the vector store has been built
        if st.session_state.vector_store is not None:
            st.caption(
                "Index status: "
                f"{st.session_state.loaded_document_count} documents, "
                f"{st.session_state.chunk_count} chunks."
            )
        else:
            st.caption("Index status: embeddings not created yet.")

        # ====================================================================
        # Document Upload Section
        # ====================================================================
        if ENABLE_DOCUMENT_UPLOAD:
            st.divider()
            st.subheader("Upload Documents")

            # File uploader widget
            uploaded_files = st.file_uploader(
                "Upload PDFs, TXT, or DOCX",
                type=list(ALLOWED_FILE_EXTENSIONS),
                accept_multiple_files=True,
                help=f"Max file size: {MAX_UPLOAD_FILE_SIZE_MB}MB per file. "
                f"Max total upload size: {MAX_TOTAL_UPLOAD_SIZE_MB}MB per session.",
            )

            # Process uploaded files if any
            if uploaded_files:
                if st.button("Process Uploads", use_container_width=True, key="process_uploads_btn"):
                    logger.info(f"User submitted {len(uploaded_files)} files for processing")
                    handle_uploaded_documents(uploaded_files)

            # Display remaining upload quota
            remaining_quota = get_upload_remaining_quota()
            st.caption(f"Upload quota: {remaining_quota:.1f}MB remaining")

            # Show uploaded files list if available
            if st.session_state.get("uploaded_vector_store") is not None:
                st.caption(
                    f"✓ {len(st.session_state.get('uploaded_files', []))} files uploaded "
                    f"({st.session_state.get('upload_chunk_count', 0)} chunks)"
                )

            # Document source selector
            st.divider()
            doc_source = st.radio(
                "Select document source",
                ["Base Documents", "Uploaded Documents"],
                key="doc_source_radio",
                index=0 if not st.session_state.get("using_uploads", False) else 1,
            )

            # Update using_uploads flag based on radio selection
            st.session_state.using_uploads = doc_source == "Uploaded Documents"

            # Check if selected source is available
            if st.session_state.using_uploads and st.session_state.get("uploaded_vector_store") is None:
                st.warning("No uploaded documents yet. Please upload and process files first.")
            elif not st.session_state.using_uploads and st.session_state.vector_store is None:
                st.warning("Base documents not indexed yet. Click 'Document Embedding' first.")

    # ========================================================================
    # Main Chat Area
    # ========================================================================

    # Display all prior messages in the chat history
    render_chat_history()

    # Chat input widget (rendered at bottom, returns user input when submitted)
    user_prompt = st.chat_input("Ask a question about the research papers")

    # If user submitted a question, process it
    if user_prompt:
        handle_user_prompt(user_prompt)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # ========================================================================
    # Run the Application
    # ========================================================================
    # This is the entry point when the script is executed with:
    # `streamlit run app.py`
    main()
