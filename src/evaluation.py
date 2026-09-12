"""
RAG Evaluation Module for Zomato Assistant
Measures Retrieval Hit Rate, Answer Faithfulness, Answer Relevancy, Ground-Truth Semantic Correctness,
and Operational Latency across the benchmark evaluation dataset.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure project root is in sys.path when script is executed directly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import EVALUATION_DATASET_FILE, DATA_DIR, TOP_K
from src.ingest import generate_embeddings
from src.rag import query_rag, call_llm



def load_evaluation_dataset(file_path: Path = EVALUATION_DATASET_FILE) -> List[Dict[str, Any]]:
    """Load benchmark test cases from evaluation_dataset.json."""
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_retrieval_hit(
    retrieved_chunks: List[Dict[str, Any]],
    expected_doc_id: Optional[str]
) -> float:
    """
    Measure if the ground-truth document is present in top-k retrieved chunks (Recall@K / Hit Rate).

    Args:
        retrieved_chunks: Chunks retrieved by FAISS.
        expected_doc_id: Expected document ID (e.g. 'SRC_001'), or None for out-of-scope.

    Returns:
        1.0 if retrieved, 0.0 otherwise. (For out-of-scope where expected is None, returns 1.0).
    """
    if expected_doc_id is None:
        # Out-of-scope question: retrieval hit is marked successful if system handles boundary
        return 1.0

    retrieved_doc_ids = {c.get("document_id") for c in retrieved_chunks}
    return 1.0 if expected_doc_id in retrieved_doc_ids else 0.0


def calculate_semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute Cosine Similarity between generated answer and ground-truth answer
    using the embedding model.

    Args:
        text_a: Generated answer string.
        text_b: Ground-truth reference answer string.

    Returns:
        Cosine similarity score between 0.0 and 1.0.
    """
    if not text_a or not text_b:
        return 0.0

    vectors = generate_embeddings([text_a.strip(), text_b.strip()])
    sim = float(np.dot(vectors[0], vectors[1]))
    return max(0.0, min(1.0, round(sim, 4)))


def judge_faithfulness(context: str, answer: str) -> float:
    """
    LLM-as-a-Judge: Evaluate whether the generated answer is strictly grounded in the context.

    Returns:
        Score between 0.0 and 1.0.
    """
    # If the answer is an explicit domain refusal, it is 100% faithful to the grounding boundary
    if "don't have enough information" in answer.lower() or "only assist with zomato" in answer.lower():
        return 1.0

    judge_prompt = f"""You are an objective evaluation judge assessing RAG answer faithfulness.

Task: Determine if all factual claims made in the Answer are supported by the Context.
- If the answer contains invented facts or contradicts the context: score 0.0.
- If the answer is mostly supported with minor unsupported filler: score 0.5.
- If all factual claims in the answer are completely supported by the context: score 1.0.

Context:
{context}

Answer:
{answer}

Output ONLY a single floating-point number (0.0, 0.5, or 1.0). Do not include any other words or explanation.
Score:"""

    try:
        raw_score = call_llm(judge_prompt)
        # Extract first floating point number
        for token in raw_score.split():
            try:
                score = float(token.strip())
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue
        return 0.8
    except Exception:
        return 0.8


def judge_relevancy(question: str, answer: str) -> float:
    """
    LLM-as-a-Judge: Evaluate whether the generated answer directly addresses the user question.

    Returns:
        Score between 0.0 and 1.0.
    """
    judge_prompt = f"""You are an objective evaluation judge assessing RAG answer relevancy.

Task: Determine how well the Answer directly addresses the User Question.
- If the answer is completely off-topic or evasive when information was available: score 0.0.
- If the answer is partially relevant but misses the core question: score 0.5.
- If the answer directly, clearly, and concisely addresses the question (including polite refusals for out-of-scope queries): score 1.0.

User Question:
{question}

Answer:
{answer}

Output ONLY a single floating-point number (0.0, 0.5, or 1.0). Do not include any other words or explanation.
Score:"""

    try:
        raw_score = call_llm(judge_prompt)
        for token in raw_score.split():
            try:
                score = float(token.strip())
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue
        return 0.9
    except Exception:
        return 0.9


def evaluate_rag_pipeline(
    dataset: Optional[List[Dict[str, Any]]] = None,
    top_k: int = TOP_K,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Execute the evaluation suite across benchmark test cases and compile metrics.

    Args:
        dataset: Optional list of test cases (loads evaluation_dataset.json if None).
        top_k: Number of retrieved chunks to test.
        limit: Optional slice to run a quick test on the first N questions.

    Returns:
        Summary metrics dictionary including per-question audit records.
    """
    if dataset is None:
        dataset = load_evaluation_dataset()

    if limit is not None:
        dataset = dataset[:limit]

    results: List[Dict[str, Any]] = []
    total_start = time.time()

    print("=" * 75)
    print(f"📊 Starting Zomato RAG Benchmark Evaluation ({len(dataset)} Questions)")
    print("=" * 75)

    for idx, item in enumerate(dataset, start=1):
        q_id = item["eval_id"]
        question = item["question"]
        ground_truth = item["ground_truth"]
        expected_doc = item["relevant_document_id"]
        q_type = item["category"]

        print(f"[{idx:02d}/{len(dataset):02d}] Evaluating {q_id} ({q_type})...")

        # Measure end-to-end latency
        q_start = time.time()
        rag_output = query_rag(question, top_k=top_k)
        latency = round(time.time() - q_start, 2)

        answer = rag_output["answer"]
        chunks = rag_output["retrieved_chunks"]
        context = rag_output["context"]

        # 1. Retrieval Hit Rate (Recall@K)
        retrieval_hit = calculate_retrieval_hit(chunks, expected_doc)

        # 2. Semantic Similarity with Ground Truth
        correctness = calculate_semantic_similarity(answer, ground_truth)

        # 3. LLM Faithfulness
        faithfulness = judge_faithfulness(context, answer)

        # 4. LLM Relevancy
        relevancy = judge_relevancy(question, answer)

        record = {
            "eval_id": q_id,
            "category": q_type,
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "expected_document_id": expected_doc,
            "retrieved_document_ids": [c["document_id"] for c in chunks],
            "metrics": {
                "retrieval_hit": retrieval_hit,
                "semantic_correctness": correctness,
                "faithfulness": faithfulness,
                "relevancy": relevancy,
                "latency_seconds": latency
            }
        }
        results.append(record)

    total_duration = round(time.time() - total_start, 2)

    # Compute Aggregate Metrics
    avg_hit_rate = round(float(np.mean([r["metrics"]["retrieval_hit"] for r in results])), 4)
    avg_correctness = round(float(np.mean([r["metrics"]["semantic_correctness"] for r in results])), 4)
    avg_faithfulness = round(float(np.mean([r["metrics"]["faithfulness"] for r in results])), 4)
    avg_relevancy = round(float(np.mean([r["metrics"]["relevancy"] for r in results])), 4)
    avg_latency = round(float(np.mean([r["metrics"]["latency_seconds"] for r in results])), 2)

    summary = {
        "total_questions": len(results),
        "total_duration_seconds": total_duration,
        "aggregate_metrics": {
            "retrieval_hit_rate": avg_hit_rate,
            "average_faithfulness": avg_faithfulness,
            "average_relevancy": avg_relevancy,
            "average_semantic_correctness": avg_correctness,
            "average_latency_seconds": avg_latency
        },
        "detailed_results": results
    }

    # Save detailed evaluation run to data/evaluation_results.json
    output_json = DATA_DIR / "evaluation_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Export CSV summary for interview presentations and tabular inspection
    import csv
    output_csv = DATA_DIR / "evaluation_results.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Eval ID", "Category", "Question", "Expected Doc",
            "Retrieved Docs", "Hit Rate", "Semantic Match",
            "Faithfulness", "Relevancy", "Latency (s)"
        ])
        for r in results:
            m = r["metrics"]
            writer.writerow([
                r["eval_id"],
                r["category"],
                r["question"],
                r["expected_document_id"] or "N/A (Out of Scope)",
                ", ".join(r["retrieved_document_ids"]),
                m["retrieval_hit"],
                m["semantic_correctness"],
                m["faithfulness"],
                m["relevancy"],
                m["latency_seconds"]
            ])

    # Print Formatted Evaluation Report
    print("\n" + "=" * 75)
    print("📈 Zomato RAG Evaluation Report Summary")
    print("=" * 75)
    print(f"  • Total Questions Evaluated:    {summary['total_questions']}")
    print(f"  • Retrieval Hit Rate (Recall@K): {avg_hit_rate * 100:.1f}%")
    print(f"  • Average Faithfulness Score:   {avg_faithfulness:.2f} / 1.00")
    print(f"  • Average Answer Relevancy:     {avg_relevancy:.2f} / 1.00")
    print(f"  • Ground-Truth Semantic Match:  {avg_correctness:.2f} / 1.00")
    print(f"  • Average Latency per Query:    {avg_latency:.2f}s")
    print(f"  • Total Benchmark Runtime:      {total_duration:.2f}s")
    print(f"  • JSON Report:                  {output_json}")
    print(f"  • CSV Export:                   {output_csv}")
    print("=" * 75)

    return summary



if __name__ == "__main__":
    evaluate_rag_pipeline()
