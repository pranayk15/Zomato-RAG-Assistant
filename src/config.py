"""
Configuration Module for Zomato RAG Chatbot
Centralizes all paths, model identifiers, chunking parameters, and environment settings.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure console output supports Unicode on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load environment variables from .env file located at the project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


# -----------------------------------------------------------------------------
# Directory & File Paths
# -----------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_FILE = DATA_DIR / "documents.json"
EVALUATION_DATASET_FILE = DATA_DIR / "evaluation_dataset.json"

VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_index"
FAISS_INDEX_FILE = VECTORSTORE_DIR / "index.faiss"
METADATA_FILE = VECTORSTORE_DIR / "metadata.pkl"
BM25_INDEX_FILE = VECTORSTORE_DIR / "bm25.pkl"

# -----------------------------------------------------------------------------
# Chunking Configuration
# -----------------------------------------------------------------------------
# chunk_size: target character count for each text chunk (500–800 range)
# chunk_overlap: overlapping characters between adjacent chunks to maintain context
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# -----------------------------------------------------------------------------
# Embedding Model Configuration
# -----------------------------------------------------------------------------
# Hugging Face Sentence Transformers model running locally on CPU
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# -----------------------------------------------------------------------------
# LLM Configuration (Grok / OpenAI-Compatible / Groq)
# -----------------------------------------------------------------------------
GROK_API_KEY = os.getenv("GROK_API_KEY", os.getenv("XAI_API_KEY", os.getenv("GROQ_API_KEY", "")))
GROK_MODEL = os.getenv("GROK_MODEL", os.getenv("GROQ_MODEL", "grok-2-mini"))
GROK_BASE_URL = os.getenv("GROK_BASE_URL", "")

# Auto-configure base URL and default model if not explicitly specified
if not GROK_BASE_URL:
    if GROK_API_KEY.startswith("gsk_") or "llama" in GROK_MODEL.lower() or "mixtral" in GROK_MODEL.lower():
        GROK_BASE_URL = "https://api.groq.com/openai/v1"
        if GROK_MODEL == "grok-2-mini":
            GROK_MODEL = "groq/compound-mini"
    else:
        GROK_BASE_URL = "https://api.x.ai/v1"

# -----------------------------------------------------------------------------
# Retrieval & Advanced RAG Configuration
# -----------------------------------------------------------------------------
# top_k: final number of most relevant chunks passed to the LLM
TOP_K = int(os.getenv("TOP_K", "4"))

# Retrieval candidates pulled prior to cross-encoder reranking
RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "8"))

# Enable Hybrid Search (Dense FAISS + Sparse BM25 via Reciprocal Rank Fusion)
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "True").lower() == "true"

# Enable Cross-Encoder Re-Ranking (Two-stage retrieval)
USE_RERANKER = os.getenv("USE_RERANKER", "True").lower() == "true"
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Semantic Cache Configuration (instant sub-10ms response for repeat/paraphrased queries)
USE_SEMANTIC_CACHE = os.getenv("USE_SEMANTIC_CACHE", "True").lower() == "true"
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.90"))



def ensure_directories() -> None:
    """Create essential project directories if they do not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


# Automatically ensure storage directories exist when config is loaded
ensure_directories()
