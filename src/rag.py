"""
RAG (Retrieval-Augmented Generation) Module for Zomato Assistant
Features:
- Hybrid Search (Dense FAISS + Sparse BM25 via Reciprocal Rank Fusion)
- Two-Stage Cross-Encoder Re-Ranking (ms-marco-MiniLM-L-6-v2)
- In-Memory Semantic Caching (sub-10ms repeat responses)
- Real-Time Token Streaming with Gemini 2.5 Flash
- Multi-Turn Conversation Memory & Clickable Source Citations
"""

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np

# Ensure project root is in sys.path when script is executed directly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    FAISS_INDEX_FILE,
    METADATA_FILE,
    BM25_INDEX_FILE,
    TOP_K,
    RETRIEVAL_CANDIDATES,
    USE_HYBRID_SEARCH,
    USE_RERANKER,
    RERANKER_MODEL_NAME,
    USE_SEMANTIC_CACHE,
    SEMANTIC_CACHE_THRESHOLD
)
from src.ingest import generate_embeddings

# -----------------------------------------------------------------------------
# Module Caches
# -----------------------------------------------------------------------------
_CACHED_INDEX = None
_CACHED_METADATA = None
_CACHED_BM25 = None
_CACHED_BM25_CHUNKS = None
_CACHED_RERANKER = None


# -----------------------------------------------------------------------------
# 1. Vector Store & BM25 Loaders
# -----------------------------------------------------------------------------
def load_vectorstore(
    index_path: Path = FAISS_INDEX_FILE,
    metadata_path: Path = METADATA_FILE
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Load and cache the FAISS vector index and metadata store."""
    global _CACHED_INDEX, _CACHED_METADATA

    if _CACHED_INDEX is not None and _CACHED_METADATA is not None:
        return _CACHED_INDEX, _CACHED_METADATA

    if not index_path.exists() or not metadata_path.exists():
        try:
            from src.ingest import run_ingestion_pipeline
            run_ingestion_pipeline()
        except Exception as e:
            raise FileNotFoundError(
                f"FAISS index or metadata not found in '{index_path.parent}'.\n"
                f"Auto-ingestion attempted but failed: {e}\n"
                "Please run: python src/ingest.py"
            )

    try:
        import faiss
    except ImportError:
        raise ImportError("faiss is not installed. Please run: pip install faiss-cpu")

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    _CACHED_INDEX = index
    _CACHED_METADATA = metadata
    return _CACHED_INDEX, _CACHED_METADATA


def load_bm25(bm25_path: Path = BM25_INDEX_FILE):
    """Load and cache the BM25 sparse model and chunks."""
    global _CACHED_BM25, _CACHED_BM25_CHUNKS

    if _CACHED_BM25 is not None:
        return _CACHED_BM25, _CACHED_BM25_CHUNKS

    if not bm25_path.exists():
        return None, None

    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
        _CACHED_BM25 = data.get("bm25")
        _CACHED_BM25_CHUNKS = data.get("chunks")

    return _CACHED_BM25, _CACHED_BM25_CHUNKS


# -----------------------------------------------------------------------------
# 2. Semantic Cache (Sub-10ms Zero-Cost Responses)
# -----------------------------------------------------------------------------
class SemanticCache:
    """
    In-memory semantic vector cache.
    Matches queries by Cosine Similarity. If similarity >= threshold,
    returns the stored answer in ~5ms without invoking the LLM.
    """
    def __init__(self, threshold: float = SEMANTIC_CACHE_THRESHOLD):
        self.threshold = threshold
        self.entries: List[Dict[str, Any]] = []
        self.hits: int = 0
        self.misses: int = 0

    def lookup(self, query_vector: np.ndarray) -> Optional[Dict[str, Any]]:
        if not self.entries or not USE_SEMANTIC_CACHE:
            self.misses += 1
            return None

        best_score = -1.0
        best_entry = None

        for entry in self.entries:
            # Query vectors are L2-normalized so dot product is exact Cosine Similarity
            score = float(np.dot(query_vector[0], entry["query_vector"][0]))
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.threshold and best_entry:
            self.hits += 1
            cached_result = dict(best_entry["result"])
            cached_result["from_cache"] = True
            cached_result["cache_similarity"] = round(best_score, 4)
            cached_result["cached_query"] = best_entry["query"]
            return cached_result

        self.misses += 1
        return None

    def store(self, query_vector: np.ndarray, query: str, result: Dict[str, Any]) -> None:
        if not USE_SEMANTIC_CACHE:
            return
        # Store a clean copy of the result
        storable = dict(result)
        storable["from_cache"] = False
        self.entries.append({
            "query_vector": query_vector.copy(),
            "query": query,
            "result": storable
        })

    def clear(self) -> None:
        self.entries.clear()
        self.hits = 0
        self.misses = 0


# Global cache instance
_SEMANTIC_CACHE = SemanticCache()


# -----------------------------------------------------------------------------
# 3. Dense, Sparse & Hybrid Search with Reciprocal Rank Fusion (RRF)
# -----------------------------------------------------------------------------
def retrieve_dense_chunks(
    query: str,
    top_k: int = TOP_K,
    score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Perform dense semantic vector search via FAISS."""
    if not query or not query.strip():
        return []

    index, metadata = load_vectorstore()
    query_vector = generate_embeddings([query.strip()])

    actual_k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vector, actual_k)

    retrieved_chunks: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        similarity = float(score)
        if score_threshold is not None and similarity < score_threshold:
            continue
        chunk_data = dict(metadata[idx])
        chunk_data["similarity_score"] = round(similarity, 4)
        retrieved_chunks.append(chunk_data)

    return retrieved_chunks


def retrieve_sparse_bm25(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Perform sparse lexical keyword search via BM25."""
    bm25, chunks = load_bm25()
    if bm25 is None or not chunks:
        return []

    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        score = float(scores[idx])
        if score <= 0.0:
            continue
        chunk_data = dict(chunks[idx])
        chunk_data["bm25_score"] = round(score, 4)
        chunk_data["bm25_rank"] = rank
        results.append(chunk_data)

    return results


def hybrid_search(
    query: str,
    top_k: int = RETRIEVAL_CANDIDATES,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combine dense FAISS vector search and sparse BM25 keyword search using
    Reciprocal Rank Fusion (RRF):
        RRF_Score(d) = 1/(rrf_k + rank_dense) + 1/(rrf_k + rank_sparse)
    """
    dense_results = retrieve_dense_chunks(query, top_k=top_k)
    sparse_results = retrieve_sparse_bm25(query, top_k=top_k)

    if not sparse_results:
        return dense_results

    combined: Dict[str, Dict[str, Any]] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        cid = chunk["chunk_id"]
        combined[cid] = dict(chunk)
        combined[cid]["rrf_score"] = 1.0 / (rrf_k + rank)
        combined[cid]["dense_rank"] = rank

    for rank, chunk in enumerate(sparse_results, start=1):
        cid = chunk["chunk_id"]
        if cid in combined:
            combined[cid]["rrf_score"] += 1.0 / (rrf_k + rank)
            combined[cid]["bm25_rank"] = rank
            combined[cid]["bm25_score"] = chunk.get("bm25_score", 0.0)
        else:
            combined[cid] = dict(chunk)
            combined[cid]["rrf_score"] = 1.0 / (rrf_k + rank)
            combined[cid]["bm25_rank"] = rank
            combined[cid]["bm25_score"] = chunk.get("bm25_score", 0.0)
            combined[cid]["similarity_score"] = 0.0

    # Sort candidates by combined RRF score
    sorted_candidates = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_candidates[:top_k]


# -----------------------------------------------------------------------------
# 4. Cross-Encoder Re-Ranking (Two-Stage Retrieval)
# -----------------------------------------------------------------------------
def get_reranker(model_name: str = RERANKER_MODEL_NAME):
    """Load and cache the CrossEncoder reranker model."""
    global _CACHED_RERANKER
    if _CACHED_RERANKER is None:
        try:
            from sentence_transformers import CrossEncoder
            _CACHED_RERANKER = CrossEncoder(model_name)
        except Exception:
            return None
    return _CACHED_RERANKER


def rerank_chunks(
    query: str,
    candidate_chunks: List[Dict[str, Any]],
    top_k: int = TOP_K
) -> List[Dict[str, Any]]:
    """
    Re-rank candidate chunks using deep cross-attention between query and text.
    """
    if not candidate_chunks or not USE_RERANKER or len(candidate_chunks) <= 1:
        return candidate_chunks[:top_k]

    reranker = get_reranker()
    if reranker is None:
        return candidate_chunks[:top_k]

    pairs = [[query, c["text"]] for c in candidate_chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(candidate_chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)

    reranked = sorted(candidate_chunks, key=lambda x: x.get("rerank_score", -999.0), reverse=True)
    return reranked[:top_k]


def retrieve_relevant_chunks(
    query: str,
    top_k: int = TOP_K,
    score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Master retrieval pipeline:
    1. Dense or Hybrid Search (FAISS + BM25) to pull top candidates.
    2. Cross-Encoder Re-Ranking to select the top-k highest precision chunks.
    """
    if not query or not query.strip():
        return []

    # Step 1: Candidate retrieval (pull 8 candidates if reranker active)
    candidates_count = RETRIEVAL_CANDIDATES if USE_RERANKER else top_k

    if USE_HYBRID_SEARCH:
        candidates = hybrid_search(query, top_k=candidates_count)
    else:
        candidates = retrieve_dense_chunks(query, top_k=candidates_count, score_threshold=score_threshold)

    # Step 2: Cross-Encoder Re-Ranking
    if USE_RERANKER:
        final_chunks = rerank_chunks(query, candidates, top_k=top_k)
    else:
        final_chunks = candidates[:top_k]

    return final_chunks


# -----------------------------------------------------------------------------
# 5. Context Formatting & Prompt Construction
# -----------------------------------------------------------------------------
def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a structured context block for the LLM."""
    if not chunks:
        return "No relevant documents found in knowledge base."

    formatted_blocks = []
    for idx, c in enumerate(chunks, start=1):
        block = (
            f"[Source {idx}: {c.get('title', 'Unknown')} (ID: {c.get('document_id', 'Unknown')})]\n"
            f"{c.get('text', '').strip()}"
        )
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)


RAG_PROMPT_TEMPLATE = """You are a helpful, professional, and friendly Zomato customer support assistant.

Your goal is to assist customers clearly and efficiently using ONLY the factual context provided below.

Response Style & Length Rules:
1. Conciseness: Keep straightforward answers concise—typically 2 to 5 clear sentences, or a few short bullet points for lists or steps.
2. Professional Tone: Respond in a warm, polite, and natural conversational customer support voice.
3. No Fluff: Do NOT start with meta-phrases like "Based on the provided context" or "According to the documents". Jump straight into helping the customer.
4. Grounding: Rely strictly on the provided Context. Do NOT invent policies, numbers, or promises.
5. Conversation Memory: If the customer asks a follow-up question (such as "How long does it take?", "What if it's already cooked?"), use the Recent Conversation to resolve pronouns and context.
6. Out-of-Scope: If the question cannot be answered from the context or is not about Zomato, say politely:
   "I don't have enough information about that in my available Zomato knowledge base. I can only help with Zomato policies, orders, refunds, Gold membership, and platform terms."
7. Boundaries: Do not claim access to live user orders, GPS tracking, personal accounts, or internal restaurant systems.

Context:
{context}

Recent Conversation:
{chat_history}

Customer Question:
{question}

Customer Support Answer:"""


def format_chat_history(chat_history: Optional[List[Dict[str, str]]], max_turns: int = 3) -> str:
    """Format recent dialogue turns into a compact context block for the LLM."""
    if not chat_history:
        return "None (New conversation)"

    filtered = [m for m in chat_history if m.get("role") in ("user", "assistant") and m.get("content")]
    recent = filtered[-max_turns * 2:]

    if not recent:
        return "None (New conversation)"

    transcript_lines = []
    for msg in recent:
        role = "Customer" if msg["role"] == "user" else "Assistant"
        content_snippet = msg["content"][:220].replace("\n", " ").strip()
        transcript_lines.append(f"{role}: {content_snippet}")

    return "\n".join(transcript_lines)


def clean_support_response(text: str) -> str:
    """Post-process LLM response to eliminate common meta-language and prefixes."""
    if not text:
        return ""

    cleaned = text.strip()
    redundant_prefixes = [
        "customer support answer:",
        "support answer:",
        "answer:",
        "based on the provided context,",
        "based on the provided context:",
        "according to the provided context,",
        "according to the provided context:",
        "based on the available zomato documents,"
    ]

    for prefix in redundant_prefixes:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()

    return cleaned


def build_prompt(query: str, context: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Construct the final grounded customer support prompt with conversation memory."""
    history_str = format_chat_history(chat_history)
    return RAG_PROMPT_TEMPLATE.format(
        context=context,
        chat_history=history_str,
        question=query.strip()
    )


# -----------------------------------------------------------------------------
# 6. LLM Calling & Token Streaming (Grok / OpenAI-Compatible / Groq)
# -----------------------------------------------------------------------------
def get_llm_client():
    """Instantiate an OpenAI-compatible client configured for Grok or Groq."""
    from openai import OpenAI
    from src.config import get_grok_api_key, get_grok_base_url

    api_key = get_grok_api_key()
    if not api_key:
        raise ValueError(
            "Grok API key is missing. Please set GROK_API_KEY (or GROQ_API_KEY) in your Streamlit Cloud Secrets or .env file."
        )

    base_url = get_grok_base_url()
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )


def call_llm(prompt: str) -> str:
    """Send prompt to Grok LLM and return generated text."""
    from src.config import get_grok_model

    client = get_llm_client()
    model = get_grok_model()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful, professional Zomato customer support assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=600
    )
    if response.choices and response.choices[0].message and response.choices[0].message.content:
        return response.choices[0].message.content.strip()
    raise RuntimeError("Grok LLM returned an empty response.")


def stream_llm(prompt: str) -> Generator[str, None, None]:
    """Yield tokens from Grok LLM in real time as they are generated."""
    from src.config import get_grok_model

    client = get_llm_client()
    model = get_grok_model()
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful, professional Zomato customer support assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=600,
        stream=True
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# -----------------------------------------------------------------------------
# 7. Source Citations
# -----------------------------------------------------------------------------
def extract_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate unique source documents from retrieved chunks."""
    seen_doc_ids = set()
    unique_sources = []

    for chunk in chunks:
        doc_id = chunk.get("document_id")
        if not doc_id or doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        unique_sources.append({
            "document_id": doc_id,
            "title": chunk.get("title", "Official Zomato Documentation"),
            "category": chunk.get("category", "general"),
            "source_name": chunk.get("source_name", "Zomato"),
            "source_url": chunk.get("source_url", ""),
            "url": chunk.get("source_url", ""),
            "similarity_score": chunk.get("similarity_score", 0.0),
            "rerank_score": chunk.get("rerank_score", None)
        })

    return unique_sources


def format_sources_text(sources: List[Dict[str, Any]]) -> str:
    """Format sources into clean clickable markdown citations."""
    if not sources:
        return ""

    lines = ["**Sources:**"]
    for idx, s in enumerate(sources, start=1):
        title = s.get("title", "Zomato Policy")
        url = s.get("source_url", "")
        category = s.get("category", "").replace("_", " ").title()

        if url:
            line = f"{idx}. [{title}]({url}) — *{category}*"
        else:
            line = f"{idx}. {title} — *{category}*"
        lines.append(line)

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 8. Master RAG Query Execution (Synchronous & Streaming)
# -----------------------------------------------------------------------------
def query_rag(
    query: str,
    top_k: int = TOP_K,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Execute full RAG pipeline:
    Semantic Cache -> Hybrid Retrieval -> Re-ranking -> Grounded Prompt -> Gemini Generation.
    """
    if not query or not query.strip():
        return {
            "query": query,
            "retrieval_query": "",
            "answer": "Please ask a specific question regarding Zomato policies, orders, cancellations, refunds, or Gold membership.",
            "raw_answer": "",
            "sources": [],
            "sources_text": "",
            "context": "",
            "prompt": "",
            "retrieved_chunks": [],
            "from_cache": False
        }

    # Step 1: Query Embedding & Semantic Cache Lookup
    query_vector = generate_embeddings([query.strip()])
    cached_response = _SEMANTIC_CACHE.lookup(query_vector)
    if cached_response is not None:
        return cached_response

    # Step 2: Contextualize follow-up query
    retrieval_query = query.strip()
    if chat_history:
        recent_user_qs = [m["content"] for m in chat_history if m.get("role") == "user" and m.get("content")]
        if recent_user_qs:
            last_q = recent_user_qs[-1]
            words = retrieval_query.lower().split()
            follow_up_triggers = ["how long", "what about", "what if", "and", "why", "can i", "is it", "does it"]
            if len(words) <= 6 or any(retrieval_query.lower().startswith(trig) for trig in follow_up_triggers):
                retrieval_query = f"{last_q} {query.strip()}"

    # Step 3: Hybrid Search + Cross-Encoder Re-Ranking
    chunks = retrieve_relevant_chunks(retrieval_query, top_k=top_k)

    # Step 4: Citations & Context
    sources = extract_sources(chunks)
    sources_text = format_sources_text(sources)
    context = format_context(chunks)

    # Step 5: Prompt Formulation
    prompt = build_prompt(query, context, chat_history=chat_history)

    # Step 6: Generate Answer with Gemini
    raw_answer = call_llm(prompt)
    answer = clean_support_response(raw_answer)

    is_refusal = "don't have enough information" in answer.lower() or "only help with zomato" in answer.lower()
    active_sources = [] if is_refusal else sources
    active_sources_text = "" if is_refusal else sources_text

    result = {
        "query": query,
        "retrieval_query": retrieval_query,
        "answer": answer,
        "raw_answer": raw_answer,
        "sources": active_sources,
        "sources_text": active_sources_text,
        "context": context,
        "prompt": prompt,
        "retrieved_chunks": chunks,
        "from_cache": False
    }

    # Step 7: Store in Semantic Cache
    if not is_refusal:
        _SEMANTIC_CACHE.store(query_vector, query, result)

    return result
