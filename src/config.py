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

def get_secret(keys, default: str = "") -> str:
    """Retrieve configuration from os.environ or st.secrets (Streamlit Cloud)."""
    if isinstance(keys, str):
        keys = [keys]

    # 1. Environment variables
    for k in keys:
        val = os.getenv(k)
        if val and val.strip() and "your_api_key_here" not in val:
            return val.strip()

    # 2. Streamlit Cloud Secrets (handles flat keys, case-insensitive, and nested sections)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            # Direct match
            for k in keys:
                if k in st.secrets and isinstance(st.secrets[k], str) and st.secrets[k].strip():
                    return st.secrets[k].strip()

            # Case-insensitive match on top-level
            for sec_k, sec_v in st.secrets.items():
                for k in keys:
                    if sec_k.lower() == k.lower() and isinstance(sec_v, str) and sec_v.strip():
                        return sec_v.strip()

            # Search nested sections like [general] or [secrets]
            for sec_k, sec_v in st.secrets.items():
                if hasattr(sec_v, "items"):
                    for k in keys:
                        if k in sec_v and isinstance(sec_v[k], str) and sec_v[k].strip():
                            return sec_v[k].strip()
                        for inner_k, inner_v in sec_v.items():
                            if inner_k.lower() == k.lower() and isinstance(inner_v, str) and inner_v.strip():
                                return inner_v.strip()
    except Exception:
        pass

    return default


def get_grok_api_key() -> str:
    key = get_secret(["GROK_API_KEY", "GROQ_API_KEY", "XAI_API_KEY", "OPENAI_API_KEY"])
    if key:
        os.environ.setdefault("GROK_API_KEY", key)
        os.environ.setdefault("GROQ_API_KEY", key)
    return key


def get_grok_model() -> str:
    return get_secret(["GROK_MODEL", "GROQ_MODEL"], "groq/compound-mini")


def get_grok_base_url() -> str:
    explicit = get_secret(["GROK_BASE_URL", "GROQ_BASE_URL"])
    if explicit:
        return explicit
    key = get_grok_api_key()
    model = get_grok_model()
    if key.startswith("gsk_") or "llama" in model.lower() or "mixtral" in model.lower() or "compound" in model.lower():
        return "https://api.groq.com/openai/v1"
    return "https://api.x.ai/v1"


GROK_API_KEY = get_grok_api_key()
GROK_MODEL = get_grok_model()
GROK_BASE_URL = get_grok_base_url()

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
