"""
RAG chain assembly and management for the Q&A application.

This module creates the Retrieval-Augmented Generation (RAG) chain that
combines document retrieval with language model generation. The chain
handles history-aware retrieval (follow-up questions) and synthesizes
answers grounded in retrieved document chunks.

The RAG pipeline:
1. History-aware retriever: Re-contextualizes follow-up questions using chat history
2. Document chain: Combines retrieved chunks with the LLM for answer generation
3. Retrieval chain: Orchestrates the full pipeline from question to answer

Functions:
    create_rag_chain(): Create the complete RAG chain for question answering.
"""


from typing import Any

import streamlit as st
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import configuration for retrieval parameters
from config import RETRIEVAL_K

# Import the LLM factory
from src.llm import create_llm


# Prompt template for re-contextualizing follow-up questions
# This prompt is used by the history-aware retriever to rephrase user questions
# in the context of previous conversation turns, making them standalone
# queries suitable for document retrieval
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Rewrite the latest user question as a standalone question using "
                "the chat history when needed. Do not answer the question."
            ),
        ),
        # MessagesPlaceholder: dynamically inserts the chat history (prior turns)
        MessagesPlaceholder("chat_history"),
        # The current user question to be re-contextualized
        ("human", "{input}"),
    ]
)


# Prompt template for generating answers
# This prompt is used by the document chain to generate final answers
# It combines the retrieved document context with the conversation history
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Answer the user's question using only the supplied context and "
                "the conversation history when it is relevant. If the context "
                "does not contain the answer, say you do not know.\n\n"
                "Context:\n{context}"
            ),
        ),
        # Include chat history so the LLM understands previous context
        MessagesPlaceholder("chat_history"),
        # The current user question (possibly reformulated by the retriever)
        ("human", "{input}"),
    ]
)


def create_rag_chain() -> Any:
    """
    Create a history-aware retrieval-augmented generation (RAG) chain.

    This function constructs the complete RAG pipeline that handles:
    1. Re-contextualization: Using chat history to rephrase follow-up questions
    2. Retrieval: Finding the most similar document chunks
    3. Answer Generation: Using the LLM to synthesize answers from chunks
 

    Requires:
        st.session_state.vector_store: FAISS vector store previously built
            by build_vector_store(). Must be initialized before calling this.

    Returns:
        Any: A LangChain Runnable chain that accepts:
            - "input" (str): The user question
            - "chat_history" (list): List of HumanMessage/AIMessage objects
            
            And returns a dict with:
            - "answer" (str): The generated answer from the LLM
            - "context" (list): List of retrieved Document chunks

    Raises:
        AttributeError: If st.session_state.vector_store is not initialized.

    Note:
        - RETRIEVAL_K controls how many chunks are retrieved (default: 4)
        - The "stuff" pattern concatenates all chunks into one context window
        - History-aware retrieval enables follow-up questions, like
          "Tell me more" or "What about X?" which reference prior turns
        - For very long conversations, may exceed LLM context windows

    """

    # Initialize the LLM
    llm = create_llm()

    # Get the retriever from the vector store
    # The retriever converts semantic similarity search into a LangChain interface
    # search_kwargs={'k': RETRIEVAL_K} specifies how many chunks to retrieve
    retriever = st.session_state.vector_store.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )


    # Create the history-aware retriever
    # This retriever uses the LLM to rephrase the user's question in light
    # of the chat history, making follow-up questions work correctly.
    # For example:
    # - Original: "Tell me more"
    # - Re-contextualized: "Tell me more about attention mechanisms in transformers"
    # (if the previous question was about attention in transformers)
    history_aware_retriever = create_history_aware_retriever(
        llm,                      # The LLM to use for re-contextualization
        retriever,                # The base retriever (vector store lookup)
        CONTEXTUALIZE_PROMPT,     # Prompt template for re-contextualization
    )


    # Create the document chain
    # This chain takes the retrieved documents (chunks) and uses the LLM
    # to generate an answer. The "stuff" pattern concatenates all chunks
    # into the prompt, suitable for moderate numbers of chunks.
    document_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)

    
    # Combine into the full retrieval chain
    # This orchestrates the full pipeline:
    # 1. Re-contextualize the question using chat history
    # 2. Retrieve similar chunks
    # 3. Generate answer from chunks and history
    # 4. Return both the answer and the context (for citations)
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    # Return the Assembled Chain
    return rag_chain


# Streaming RAG Chain (for Real-Time Output)


def stream_rag_response(
    rag_chain: Any,
    user_input: str,
    chat_history: list,
) -> Any:
    """
    Stream RAG responses token-by-token for real-time output.

    This generator function yields the LLM response incrementally as it's
    generated, enabling real-time streaming to the UI. Instead of waiting
    for the complete response, the LLM output appears character by character
    as it's computed, providing better user experience and perceived performance.


    Args:
        rag_chain (Any): The RAG chain created by create_rag_chain().
        user_input (str): The user's question to process.
        chat_history (list): List of HumanMessage/AIMessage for context.

    Yields:
        str: Successive chunks of the LLM response (token-by-token streaming).
            Caller should collect these streaming chunks and render to UI.

    Returns:
        None (this is a generator function)

    Note:
        - Streaming only works with LLMs that support streaming (Groq does)
        - Some overhead vs. batch processing but much better UX
        - Useful for long text generation
        - Chunks are typically small (a few tokens)

    """


    # Use stream() method instead of invoke() to get incremental output
    for event in rag_chain.stream(
        {
            "input": user_input,
            "chat_history": chat_history,
        }
    ):
        # Each event is a dict with keys like 'answer', 'context'
        # Extract and yield only the answer portion
        if "answer" in event:
            yield event["answer"]
