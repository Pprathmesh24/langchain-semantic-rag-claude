# Document Intelligence RAG System

A production-grade Retrieval-Augmented Generation (RAG) pipeline built with LangChain, ChromaDB, and Claude. Designed to answer questions grounded strictly in your own documents, with multi-turn conversation memory and advanced retrieval techniques.

---

## Architecture

```
                        ┌─────────────────────────────┐
                        │      INGESTION PIPELINE      │
                        │                              │
  Documents ──────────► │  Load → Chunk → Embed → Store│
  (PDF, TXT, DOCX,      │                              │
   CSV, HTML)           └──────────────┬──────────────┘
                                       │
                                  ChromaDB
                                 (persisted)
                                       │
                        ┌──────────────▼──────────────┐
                        │      RETRIEVAL PIPELINE      │
                        │                              │
  User Query ─────────► │  Embed Query → MMR Search    │
                        │                              │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │      GENERATION PIPELINE     │
                        │                              │
                        │  Context + History → Claude  │
                        │                              │
                        └──────────────┬──────────────┘
                                       │
                                   Answer
```

---

## Advanced Techniques Used

### 1. Semantic Chunking
Rather than splitting documents every N characters (fixed-size chunking), the pipeline uses `SemanticChunker` from LangChain Experimental. It embeds each sentence and measures cosine distance between consecutive sentences — splitting only when it detects a spike in distance, meaning the topic has shifted. This produces chunks that contain complete, coherent ideas instead of arbitrary text slices.

### 2. Hybrid Retrieval: MMR + Score Threshold Filtering
The retriever implements a custom two-stage retrieval strategy that combines diversity-aware ranking with minimum relevance enforcement:

**Stage 1 — Maximum Marginal Relevance (MMR):** Fetches a large candidate pool (`fetch_k = 20`) from ChromaDB, then iteratively selects `top_k = 5` chunks by maximising a combined objective: relevance to the query and dissimilarity to already-selected chunks. The `lambda_mult` parameter controls the relevance/diversity tradeoff (0.5 = balanced). This eliminates near-duplicate chunks that standard top-k similarity search would return.

**Stage 2 — Cosine Score Threshold Filtering:** Each MMR result's cosine distance is converted to a relevance score (`relevance = 1 - cosine_distance`) and evaluated against `SIMILARITY_THRESHOLD = 0.3`. Chunks below this threshold are discarded. This acts as a hard relevance gate — necessary because MMR's diversity objective can occasionally promote a low-relevance chunk into the final set, and to handle queries whose topic is absent from the corpus entirely.

This two-stage approach is not achievable through LangChain's standard `as_retriever()` API, which does not expose similarity scores in MMR mode. The pipeline instead calls `max_marginal_relevance_search_with_score()` directly on the Chroma collection and wraps the logic in a `RunnableLambda` to remain composable within LCEL chains.

### 4. Multi-File Type Ingestion
The ingestion pipeline supports mixed document types in a single directory via a file extension → loader mapping. Adding support for a new file type requires only one line in `FILE_LOADERS`. Supported types:

| Format | Loader |
|--------|--------|
| `.txt` | `TextLoader` |
| `.pdf` | `PyMuPDFLoader` |
| `.docx` | `Docx2txtLoader` |
| `.csv` | `CSVLoader` |
| `.html` | `UnstructuredHTMLLoader` |

### 5. Multi-Turn Conversation Memory
The generation pipeline maintains a `chat_history` list of `HumanMessage` and `AIMessage` objects. The full conversation history is injected into the prompt on every turn via `MessagesPlaceholder`, so Claude can reference earlier exchanges without external storage.

### 6. LCEL Chain (LangChain Expression Language)
The generation pipeline is built using the modern LCEL `|` pipe syntax instead of the legacy `RetrievalQA` chain. This makes each step explicit and composable — retriever, formatter, prompt, LLM, and output parser are all independently swappable.

### 7. Config-Driven Architecture
All tunable parameters live in a single `config.py` file — embedding model, chunk size, ChromaDB path, similarity threshold, MMR parameters, and LLM model. No magic numbers scattered across files.

---

## Project Structure

```
RAG/
├── src/
│   ├── config.py                # All tunable parameters
│   ├── injestion_pipeline.py    # Load → chunk → embed → store
│   ├── retrieval_pipeline.py    # Load vector store → MMR retriever
│   └── generation_pipeline.py  # RAG chain + conversation memory
├── rag_data/                    # Drop your documents here
├── chroma_db/                   # Auto-created, persisted vector store
└── requirements.txt
```

---

## Stack

| Component | Technology |
|-----------|-----------|
| LLM | Claude Haiku (Anthropic) |
| Embeddings | `all-mpnet-base-v2` (HuggingFace, local) |
| Vector Store | ChromaDB (cosine similarity) |
| Framework | LangChain + LangChain Experimental |
| Chunking | SemanticChunker |
| Retrieval | MMR + Cosine Score Threshold (two-step) |

---

## Setup

```bash
# Install dependencies
uv add -r requirements.txt

# Create the documents directory and add your files
mkdir rag_data
# Drop your PDF, TXT, DOCX, CSV, or HTML files into rag_data/
# (this directory is gitignored — it holds your personal documents)

# Set your Anthropic API key in .env
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Run ingestion (only needed once, or when documents change)
python src/injestion_pipeline.py

# Start the chatbot
python src/generation_pipeline.py
```

---

## Configuration

Edit `src/config.py` to tune the pipeline:

```python
EMBEDDING_MODEL = "all-mpnet-base-v2"    # swap for OpenAI embeddings in production
SIMILARITY_THRESHOLD = 0.3                # raise to be stricter about relevance
MMR_FETCH_K = 20                          # candidate pool size before MMR selection
MMR_LAMBDA = 0.5                          # 1.0 = pure relevance, 0.0 = pure diversity
AGENT_MODEL = "claude-haiku-4-5-20251001"
```
