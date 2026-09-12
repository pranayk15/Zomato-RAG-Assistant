"""
Document Ingestion Module for Zomato RAG Chatbot
Handles document loading, validation, text cleaning, and metadata preservation.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path when script is executed directly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import DOCUMENTS_FILE



def load_documents(file_path: Path = DOCUMENTS_FILE) -> List[Dict[str, Any]]:
    """
    Load raw knowledge base documents from a JSON file.

    Args:
        file_path: Path to the documents.json file.

    Returns:
        List of raw document dictionaries.

    Raises:
        FileNotFoundError: If the documents file does not exist.
        ValueError: If JSON is invalid or document structure is incorrect.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at '{file_path}'. "
            "Please ensure data/documents.json exists."
        )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
    except json.JSONDecodeError as err:
        raise ValueError(f"Failed to parse '{file_path}': Invalid JSON syntax ({err})")

    if not isinstance(documents, list) or len(documents) == 0:
        raise ValueError(f"'{file_path}' must contain a non-empty list of documents.")

    return documents


def clean_text(text: str) -> str:
    """
    Clean and normalize document text.

    - Replaces Windows \\r\\n newlines with \\n
    - Normalizes repeated blank lines (3+ newlines to 2)
    - Replaces multiple consecutive spaces or tabs with a single space
    - Strips leading and trailing whitespace

    Args:
        text: Raw text string from document content.

    Returns:
        Cleaned, normalized string.
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace horizontal whitespace (tabs, multiple spaces) with single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive newlines (max 2 consecutive newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def validate_and_prepare_documents(raw_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate document schemas, clean content, and structure metadata.

    Args:
        raw_docs: List of raw document dictionaries loaded from JSON.

    Returns:
        List of validated and cleaned document objects with unified metadata.
    """
    required_keys = {"document_id", "title", "category", "source_url", "content"}
    prepared_docs = []

    for idx, doc in enumerate(raw_docs, start=1):
        missing = required_keys - set(doc.keys())
        if missing:
            raise ValueError(
                f"Document at index {idx} (ID: {doc.get('document_id', 'unknown')}) "
                f"is missing required fields: {missing}"
            )

        raw_content = doc["content"]
        if not isinstance(raw_content, str) or not raw_content.strip():
            raise ValueError(
                f"Document '{doc['document_id']}' has empty or invalid content."
            )

        cleaned_content = clean_text(raw_content)

        prepared_doc = {
            "document_id": str(doc["document_id"]).strip(),
            "title": str(doc["title"]).strip(),
            "category": str(doc["category"]).strip(),
            "source_type": str(doc.get("source_type", "official")).strip(),
            "source_name": str(doc.get("source_name", "Zomato")).strip(),
            "source_url": str(doc["source_url"]).strip(),
            "content": cleaned_content,
            "metadata": doc.get("metadata", {})
        }
        prepared_docs.append(prepared_doc)

    return prepared_docs


def split_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    separators: List[str] = None
) -> List[str]:
    """
    Split text into overlapping chunks using hierarchical boundary separators.

    Separators are tried in priority order:
    1. Paragraphs (\\n\\n)
    2. Lines / Bullet points (\\n)
    3. Sentences (. )
    4. Words (" ")

    Args:
        text: Normalized input text string.
        chunk_size: Max characters per chunk (defaults to config.CHUNK_SIZE).
        chunk_overlap: Overlap characters between chunks (defaults to config.CHUNK_OVERLAP).
        separators: Priority list of string separators.

    Returns:
        List of text chunks.
    """
    from src.config import CHUNK_SIZE, CHUNK_OVERLAP

    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = CHUNK_OVERLAP
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    if not text:
        return []

    # If text already fits within chunk_size, no split is needed
    if len(text) <= chunk_size:
        return [text]

    # Find the highest-priority separator present in the text
    chosen_sep = ""
    for sep in separators:
        if sep in text:
            chosen_sep = sep
            break

    # Split into raw splits by the chosen separator
    if chosen_sep:
        splits = text.split(chosen_sep)
    else:
        # Fallback to character splitting
        splits = list(text)

    chunks: List[str] = []
    current_chunk = ""

    for split in splits:
        split_piece = (split + chosen_sep) if chosen_sep else split
        
        # If adding split_piece exceeds chunk_size and we already have accumulated text
        if len(current_chunk) + len(split_piece) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Keep the tail of current_chunk for overlap
            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                overlap_text = current_chunk[-chunk_overlap:]
                # Try not to start the overlap in the middle of a word if possible
                first_space = overlap_text.find(" ")
                if first_space != -1 and first_space < len(overlap_text) - 1:
                    overlap_text = overlap_text[first_space + 1:]
                current_chunk = overlap_text + split_piece
            else:
                current_chunk = split_piece
        else:
            current_chunk += split_piece

    if current_chunk and current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Handle rare cases where a single split segment itself was larger than chunk_size
    final_chunks: List[str] = []
    for c in chunks:
        if len(c) > chunk_size * 1.5 and chosen_sep in separators[:-1]:
            # Recursively split oversized chunks with the next finer separator
            next_seps = separators[separators.index(chosen_sep) + 1:]
            final_chunks.extend(split_text(c, chunk_size, chunk_overlap, next_seps))
        else:
            final_chunks.append(c)

    return final_chunks


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Dict[str, Any]]:
    """
    Split a list of validated documents into chunks with attached metadata.

    Args:
        documents: List of cleaned documents from validate_and_prepare_documents.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Number of overlapping characters.

    Returns:
        List of chunk dictionaries, each containing chunk_id, text, and source metadata.
    """
    all_chunks: List[Dict[str, Any]] = []

    for doc in documents:
        doc_id = doc["document_id"]
        text_chunks = split_text(doc["content"], chunk_size, chunk_overlap)

        for idx, chunk_text in enumerate(text_chunks, start=1):
            chunk_obj = {
                "chunk_id": f"{doc_id}_c{idx:02d}",
                "document_id": doc_id,
                "title": doc["title"],
                "category": doc["category"],
                "source_type": doc.get("source_type", "official"),
                "source_name": doc.get("source_name", "Zomato"),
                "source_url": doc["source_url"],
                "chunk_index": idx,
                "total_doc_chunks": len(text_chunks),
                "text": chunk_text,
                "char_count": len(chunk_text),
                "metadata": doc.get("metadata", {})
            }
            all_chunks.append(chunk_obj)

    return all_chunks


# Global singleton cache for embedding model to avoid reloading overhead
_EMBEDDING_MODEL = None


def get_embedding_model(model_name: str = None):
    """
    Load and cache the SentenceTransformer embedding model.

    Args:
        model_name: HuggingFace model identifier. Defaults to config.EMBEDDING_MODEL_NAME.

    Returns:
        SentenceTransformer instance.
    """
    global _EMBEDDING_MODEL
    from src.config import EMBEDDING_MODEL_NAME

    if model_name is None:
        model_name = EMBEDDING_MODEL_NAME

    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Please run: pip install sentence-transformers"
            )

        print(f"Loading embedding model '{model_name}' (runs locally on CPU)...")
        _EMBEDDING_MODEL = SentenceTransformer(model_name)

    return _EMBEDDING_MODEL


def generate_embeddings(texts: List[str], model=None):
    """
    Generate normalized dense vector embeddings for a list of text strings.

    Args:
        texts: List of text chunks or queries to embed.
        model: Optional pre-loaded SentenceTransformer instance.

    Returns:
        numpy.ndarray of shape (len(texts), embedding_dimension) with float32 dtype.
    """
    import numpy as np

    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    if model is None:
        model = get_embedding_model()

    # normalize_embeddings=True makes L2 norm = 1.0, enabling exact Cosine Similarity via dot product
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings.astype(np.float32)


def create_faiss_index(embeddings):
    """
    Build a FAISS vector index using Inner Product (Cosine Similarity on normalized vectors).

    Args:
        embeddings: numpy.ndarray of shape (N, dimension).

    Returns:
        faiss.IndexFlatIP instance populated with vectors.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("faiss is not installed. Please run: pip install faiss-cpu")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_faiss_index(
    index,
    metadata: List[Dict[str, Any]],
    index_path: Path = None,
    metadata_path: Path = None
) -> None:
    """
    Persist the FAISS index and chunk metadata to disk.

    Args:
        index: faiss.Index instance.
        metadata: List of chunk metadata dictionaries aligned with vector indices.
        index_path: Target path for index.faiss. Defaults to config.FAISS_INDEX_FILE.
        metadata_path: Target path for metadata.pkl. Defaults to config.METADATA_FILE.
    """
    import pickle
    try:
        import faiss
    except ImportError:
        raise ImportError("faiss is not installed. Please run: pip install faiss-cpu")

    from src.config import FAISS_INDEX_FILE, METADATA_FILE

    if index_path is None:
        index_path = FAISS_INDEX_FILE
    if metadata_path is None:
        metadata_path = METADATA_FILE

    index_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Write the FAISS binary index
    faiss.write_index(index, str(index_path))

    # 2. Write the companion metadata pickle
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)


def create_bm25_index(chunks: List[Dict[str, Any]], bm25_path: Path = None):
    """
    Tokenize chunks and construct a BM25 sparse lexical index.
    Saves the BM25 model and tokenized corpus to disk for fast startup.

    Args:
        chunks: List of chunk metadata dictionaries.
        bm25_path: Target path for bm25.pkl. Defaults to config.BM25_INDEX_FILE.

    Returns:
        BM25Okapi model instance.
    """
    import pickle
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("rank-bm25 is not installed. Please run: pip install rank-bm25")

    from src.config import BM25_INDEX_FILE

    if bm25_path is None:
        bm25_path = BM25_INDEX_FILE

    # Simple word tokenization for lexical matching
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    return bm25


def run_ingestion() -> None:
    """
    Execute the complete end-to-end ingestion pipeline:
    1. Load documents.json
    2. Validate and clean documents
    3. Split into overlapping chunks
    4. Generate SentenceTransformer embeddings
    5. Construct FAISS vector index
    6. Construct BM25 lexical index
    7. Persist index artifacts locally
    """
    from src.config import FAISS_INDEX_FILE, METADATA_FILE, BM25_INDEX_FILE

    print("=" * 65)
    print("🚀 Starting Zomato Knowledge Base Ingestion Pipeline")
    print("=" * 65)

    # Stage 1: Load documents
    print("[1/6] Loading documents from JSON...")
    raw_docs = load_documents()
    print(f"      Loaded {len(raw_docs)} documents.")

    # Stage 2: Validate and clean
    print("[2/6] Validating schemas and normalizing text...")
    prepared_docs = validate_and_prepare_documents(raw_docs)

    # Stage 3: Text chunking
    print("[3/6] Splitting documents into overlapping chunks...")
    chunks = chunk_documents(prepared_docs)
    print(f"      Generated {len(chunks)} chunks across {len(prepared_docs)} documents.")

    # Stage 4: Generate embeddings
    print("[4/6] Generating dense vector embeddings...")
    chunk_texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(chunk_texts)
    print(f"      Encoded {embeddings.shape[0]} vectors (dimension: {embeddings.shape[1]}).")

    # Stage 5: Build and save FAISS index
    print("[5/6] Building FAISS IndexFlatIP and serializing to disk...")
    index = create_faiss_index(embeddings)
    save_faiss_index(index, chunks, FAISS_INDEX_FILE, METADATA_FILE)

    # Stage 6: Build and save BM25 sparse index
    print("[6/6] Building BM25 lexical index for Hybrid Search...")
    create_bm25_index(chunks, BM25_INDEX_FILE)

    print("-" * 65)
    print("🎉 Ingestion Pipeline Completed Successfully!")
    print(f"  • FAISS Index:    {FAISS_INDEX_FILE} ({index.ntotal} vectors)")
    print(f"  • Metadata Store: {METADATA_FILE} ({len(chunks)} records)")
    print(f"  • BM25 Index:     {BM25_INDEX_FILE} (sparse lexical search)")
    print("=" * 65)


if __name__ == "__main__":
    run_ingestion()



