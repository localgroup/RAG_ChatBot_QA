# RAG Document Q&A Application

A production-ready Retrieval-Augmented Generation (RAG) web application for querying documents using natural language. Built with **Streamlit**, **LangChain**, **HuggingFace embeddings**, and **Groq LLM**.

## Overview

This application enables interactive question-answering over documents through:

- **Multi-Format Support**: PDF, TXT, and DOCX documents
- **Semantic + Keyword Search**: Hybrid retrieval for best results
- **Real-Time Streaming**: Answers appear word-by-word as generated
- **History-Aware**: Follows context of previous questions
- **File Upload**: Add documents directly in UI (no folder access needed)
- **Intelligent Error Handling**: Graceful responses for rate limits and errors
- **Source Citations**: See exactly which documents informed each answer

## Features

✅ **Chat-Style Interface** - Threaded conversation with persistent history  
✅ **History-Aware Retrieval** - Follow-up questions use prior conversation context  
✅ **Multi-Format Support** - Upload PDF, TXT, and DOCX documents  
✅ **Streaming Responses** - Real-time token-by-token answer generation  
✅ **Advanced Retrieval** - Hybrid search (semantic + keyword-based BM25)  
✅ **Document Upload** - Upload files directly in the UI (50MB/session limit)  
✅ **Source Citation** - See which documents and pages informed each answer  
✅ **Performance Metrics** - Response generation time displayed for each answer  
✅ **Graceful Error Handling** - User-friendly messages for rate limits and errors  
✅ **Logging System** - All operations logged to `logs/` for debugging  
✅ **Vector Store Persistence** - Indexes cached to disk for fast reloads  
✅ **Easy Configuration** - All parameters centralized in `config.py`  

## Project Structure

```
.
├── main.py                   # Quick start guide (python main.py shows instructions)
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration & constants (all settings centralized)
├── requirements.txt          # Python dependencies
├── pyproject.toml            # UV project configuration (optional)
├── .env.example              # Environment variables template
├── .env                      # Your API keys (create from .env.example)
├── .gitignore                # Git ignore rules
├── README.md                 # This file
├── documents/                # Directory for documents (PDF, TXT, DOCX)
├── logs/                     # Application logs (auto-created)
├── vector_stores/            # Cached FAISS indexes (auto-created)
└── src/                      # Source modules (8 specialized components)
    ├── __init__.py          # Package initialization
    ├── embeddings.py        # HuggingFace embedding model creation
    ├── llm.py               # Groq LLM initialization
    ├── document_loader.py   # Multi-format document loading (PDF, TXT, DOCX)
    ├── document_upload.py   # File upload validation & processing
    ├── vector_store.py      # FAISS vector store creation & persistence
    ├── rag_chain.py         # RAG pipeline assembly with streaming
    ├── retrieval.py         # Advanced retrieval (hybrid + reranking)
    ├── logging_config.py    # Logging system setup
    └── utils.py             # Helper functions & session management
```

## Installation

### 1. Prerequisites

- Python 3.9 or higher
- UV package manager (recommended, faster) or pip
- Groq API key (free at https://console.groq.com/keys)

### 2. Clone or Download the Project

```bash
cd /path/to/rag-chatbot
```

### 3. Create Virtual Environment

**Using UV (recommended, faster):**
```bash
uv venv
uv sync
```

**Using pip:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Understanding Dependencies

**Core RAG Stack:**
- **streamlit**: Interactive web UI
- **langchain**: RAG orchestration framework
- **faiss-cpu**: Fast vector similarity search
- **langchain-groq**: Groq LLM integration
- **langchain-huggingface**: HuggingFace embeddings

**Document Processing:**
- **pypdf**: PDF text extraction
- **python-docx**: Word document parsing

**Advanced Features:**
- **sentence-transformers**: Embeddings model loader
- **rank-bm25**: Keyword-based retrieval (hybrid search)

**Note:** Only Groq API key is required. HuggingFace embeddings run locally (no key needed for public models).

### 5. Set Up Environment Variables

Copy `.env.example` to `.env` and add your Groq API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
```

**Get Groq API Key:** https://console.groq.com/keys (free tier available)

### 6. Add Documents

Place your documents in the `documents/` folder. Supported formats:

```
documents/
├── research_paper.pdf      # PDF documents
├── notes.txt               # Plain text files
└── report.docx             # Word documents
```

**Or upload directly in the app UI** (up to 50MB per session)

## Usage

### 1. Start the Application

**Quick Start:**
```bash
python main.py    # Shows instructions
```

**Recommended with UV:**
```bash
uv run streamlit run app.py
```


**Or run directly if Streamlit starts normally on your machine:**
```bash
streamlit run app.py
```

Browser opens at: `http://localhost:8501`

### 2. Workflow

#### **Option A: Use Documents from `documents/` Folder**
1. Click **"Document Embedding"** in the sidebar
2. App loads all PDFs, TXT, DOCX files and creates FAISS index
3. Status shows loaded documents and chunks
4. Start asking questions

#### **Option B: Upload Documents in UI**
1. Scroll to **"Upload Documents"** section in sidebar
2. Select PDF, TXT, or DOCX files (up to 50MB total per session)
3. Files processed automatically, separate vector store created
4. Switch between base and uploaded documents using radio button
5. Clear session to reset upload quota

#### **3. Ask Questions**
- Type question in chat input
- App streams response token-by-token (real-time)
- Response time and retrieval metrics displayed
- Source documents shown in expandable section

#### **4. Follow-Up Questions**
- Chat history automatically used for context
- Model understands references to previous answers
- Perfect for multi-turn conversations

#### **5. Error Handling**
- **Rate limit hit?** User-friendly message shows wait time
- **Question too large?** Tips provided to shorten query
- **File upload failed?** Clear feedback on what went wrong

#### **6. Manage Session**
- **Clear Chat** → Reset conversation, keep index
- **Change document source** → Switch between base/uploaded docs
- **New session** → Closes and refreshes (auto-cleanup)

## Configuration

All parameters centralized in `config.py`:

```python
# ============================================================================
# Models
# ============================================================================
GROQ_MODEL = "llama-3.1-8b-instant"        # LLM for generating answers
GROQ_TEMPERATURE = 0                       # 0 = deterministic, 1 = creative
HF_EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Embeddings for similarity search

# ============================================================================
# Document Processing
# ============================================================================
MAX_SOURCE_DOCUMENTS = 50    # Max documents to load
CHUNK_SIZE = 2000            # Characters per chunk
CHUNK_OVERLAP = 200          # Context overlap between chunks

# ============================================================================
# Retrieval Settings
# ============================================================================
RETRIEVAL_K = 4              # Number of chunks to retrieve per query
ENABLE_HYBRID_SEARCH = True  # Semantic + keyword-based search

# ============================================================================
# Features
# ============================================================================
ENABLE_STREAMING = True      # Real-time token display
ENABLE_RESULT_CACHING = True # Cache expensive queries
ENABLE_DOCUMENT_UPLOAD = True
AUTO_PERSIST_VECTOR_STORE = True

# ============================================================================
# Upload Settings
# ============================================================================
MAX_UPLOAD_FILE_SIZE_MB = 50     # Per file limit
MAX_TOTAL_UPLOAD_SIZE_MB = 50    # Per session limit
ALLOWED_FILE_EXTENSIONS = (".pdf", ".txt", ".docx")
```

### **Common Tunings:**

| Goal | Changes |
|------|---------|
| **More context** | Increase `CHUNK_SIZE` (2000→3000), increase `RETRIEVAL_K` (4→6) |
| **Faster responses** | Decrease `CHUNK_SIZE` (2000→1000), decrease `RETRIEVAL_K` (4→2) |
| **Better follow-ups** | Increase `CHUNK_OVERLAP` (200→400) |
| **More sources shown** | Increase `RETRIEVAL_K` (4→8) |
| **Disable streaming** | Set `ENABLE_STREAMING = False` |
| **Keyword search** | Set `ENABLE_HYBRID_SEARCH = True` (helps with exact terms) |

## Architecture

### Complete RAG Pipeline

```
┌─ User Input (Question + Chat History)
│
├→ History-Aware Retriever
│  └→ Re-contextualizes follow-up questions
│
├→ Hybrid Retrieval
│  ├→ Semantic search (FAISS)
│  └→ Keyword search (BM25)
│     └→ Combined & reranked
│
├→ Retrieved Chunks
│  └→ Top-K most relevant documents
│
├→ RAG Context Assembly
│  ├→ Combine chunks + history
│  └→ Build prompt for LLM
│
├→ Groq LLM Generation
│  ├→ Stream tokens in real-time (if enabled)
│  └→ Generate answer grounded in documents
│
└→ Post-Processing
   ├→ Extract sources
   ├→ Format response
   ├→ Measure timing
   └→ Log to file
```

### Data Flow

```
┌─ Documents (PDF/TXT/DOCX)
│  ↓
├─ Document Loader
│  ├─ PyPDFLoader (PDFs)
│  ├─ Text reader (TXT)
│  └─ python-docx (DOCX)
│  ↓
├─ Text Splitter
│  └─ Overlapping chunks (2000 chars)
│  ↓
├─ Embeddings (HuggingFace)
│  └─ Vector conversion (384-dim)
│  ↓
├─ FAISS Index
│  ├─ In-memory for session
│  └─ Saved to disk for reuse
│  ↓
└─ Ready for Retrieval
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | Central configuration & constants |
| `app.py` | Streamlit UI & orchestration |
| `src/embeddings.py` | Create HuggingFace embedding model |
| `src/llm.py` | Initialize Groq chat model |
| `src/document_loader.py` | Load multi-format documents |
| `src/document_upload.py` | Handle file uploads with validation |
| `src/vector_store.py` | Split docs, create & persist FAISS index |
| `src/rag_chain.py` | Assemble RAG pipeline with streaming |
| `src/retrieval.py` | Hybrid search & reranking |
| `src/logging_config.py` | Application logging setup |
| `src/utils.py` | Session state, history, error formatting |

## Performance Tips

1. **First Run is Slower**: Embeddings downloaded (~100MB) and vector store created. Subsequent runs cached.
2. **Streaming Enabled**: Provides real-time feedback (feels faster than waiting for full response)
3. **Adjust Chunk Size**: Larger chunks = fewer vectors but less precise; smaller = more precise but slower
4. **BM25 Hybrid**: Combines semantic search with keyword matching for better accuracy

## Troubleshooting

### "Missing required environment variable: GROQ_API_KEY"

**Solution**: 
- Ensure `.env` file exists in project root
- Check `.env` contains: `GROQ_API_KEY=your_key_here` (no extra spaces)
- Verify key is correct from https://console.groq.com/keys

### "Documents directory not found"

**Solution**: 
- Create `documents/` folder in project root
- Add PDF, TXT, or DOCX files to the folder
- Or use the upload feature in the UI

### "No documents were loaded"

**Solutions**:
- Ensure files are in `documents/` folder
- Check file types are supported (.pdf, .txt, .docx)
- Try a different PDF (some scanned images may not work)
- Check logs: `tail -f logs/rag_app.log`

### Rate Limit Error ("tokens per minute exceeded")

**Solutions**:
- Wait for the displayed retry time
- Ask shorter, more concise questions
- Consider upgrading Groq plan for higher limits
- Reduce `RETRIEVAL_K` to use fewer tokens

### Slow Response Times

**Solutions**:
- Reduce `CHUNK_SIZE` to fewer characters
- Reduce `RETRIEVAL_K` to retrieve fewer chunks
- Disable `ENABLE_HYBRID_SEARCH` for speed
- Check internet connection (Groq API latency)
- Use smaller documents

### Out of Memory Error

**Solutions**:
- Reduce `MAX_SOURCE_DOCUMENTS` (process fewer files)
- Reduce `CHUNK_SIZE` (smaller chunks use less memory)
- Close other applications
- Use a machine with more RAM

### Vector Store Not Building

**Solutions**:
- Check `documents/` folder has readable files
- Try uploading via UI instead
- Verify FAISS installation: `pip install --upgrade faiss-cpu`
- Check logs for specific error messages

## Advanced Usage

### Custom Prompts

Edit prompt templates in `src/rag_chain.py`:

```python
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([...])
ANSWER_PROMPT = ChatPromptTemplate.from_messages([...])
```

### Different Embedding Models

Edit `config.py`:

```python
HF_EMBEDDING_MODEL = "all-mpnet-base-v2"  # Higher quality but slower
```

### Disable Streaming

For production deployments, disable streaming in `config.py`:

```python
ENABLE_STREAMING = False
```

## Dependencies Summary

| Package | Purpose | Required |
|---------|---------|----------|
| streamlit | Web UI framework | ✅ |
| langchain | LLM orchestration | ✅ |
| langchain-groq | Groq API integration | ✅ |
| langchain-huggingface | Embeddings | ✅ |
| faiss-cpu | Vector search | ✅ |
| pypdf | PDF parsing | ✅ |
| python-docx | Word parsing | ✅ |
| sentence-transformers | Model loader | ✅ |
| rank-bm25 | Keyword search | ✅ |
| python-dotenv | Environment variables | ✅ |

## Limitations

- **Session-Based**: Chat history cleared when session ends (can be persisted with database)
- **Context Window**: Very long conversations may exceed LLM limits
- **Single User**: Designed for local use (not multi-user production)
- **Groq Rate Limits**: Free tier limited to 6,000 tokens/minute
- **File Size**: Individual files up to 50MB, 50MB per session total

## Future Enhancements

- [ ] Database persistence for chat history
- [ ] Support for more document formats (Excel, PowerPoint)  
- [ ] Advanced filtering (by date, category, tags)
- [ ] Query refinement suggestions
- [ ] Multi-language support
- [ ] Web deployment guide (Streamlit Cloud, Docker)
- [ ] Vector database alternatives (Pinecone, Weaviate)
- [ ] User authentication & multi-user support

## License

[Specify your license here]

## Contributing

[Contribution guidelines if open source]

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review logs in `logs/rag_app.log`
3. Check Groq API status at https://status.groq.com
