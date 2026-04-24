"""
Language Model (LLM) creation and initialization for the RAG application.

This module provides functions to create and configure the Groq chat model
used for generating answers in the RAG system. The LLM is responsible for
synthesizing answers from retrieved document chunks and chat history.

Functions:
    create_llm(): Create a configured Groq chat model instance.
"""


from langchain_groq import ChatGroq

# Import configuration constants for model name and temperature
from config import GROQ_MODEL, GROQ_TEMPERATURE




def create_llm() -> ChatGroq:
    """
    Create and return a Groq chat model for RAG answer generation.
    This function instantiates the ChatGroq model specified in config.GROQ_MODEL.

    Returns:
        ChatGroq: An initialized Groq chat model configured for RAG tasks.
            Ready to generate answers using invoke() or stream methods.

    Raises:
        KeyError: If GROQ_API_KEY environment variable is not set.

    Note:
        Requires GROQ_API_KEY to be set in environment variables.
        You can set this in a .env file or export as an environment variable.
        
    """

    # Create and return the Groq chat model with configured parameters
    return ChatGroq(model=GROQ_MODEL, temperature=GROQ_TEMPERATURE)
