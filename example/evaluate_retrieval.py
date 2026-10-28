import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import json
import re
import time
from collections import defaultdict

import numpy as np

from chunk_pdf import load_chunks
from embedding import Embeddings
from retriever import Retriever
from reranker import Reranker
from vector_store import VectorStore

# ---------- Configuration ----------
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "test.pdf"
QUESTIONS_PATH = DATA_DIR / "evaluation_questions.json"
REPORT_PATH = BASE_DIR / "retrieval_evaluation_report.json"

TOP_K_RETRIEVE = 15
TOP_K_FINAL = 8  

# ---------- Helper functions ----------
def build_keyword_pattern(keyword):
    escaped = re.escape(str(keyword).strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(
        r"(?<![0-9A-Za-z])" + escaped + r"(?![0-9A-Za-z])",
        re.IGNORECASE,
    )

def find_matched_keywords(text, keyword_patterns):
    return [kw for kw, p in keyword_patterns.items() if p.search(text)]

def chunk_contains_all_keywords(text, keyword_patterns):
    return all(p.search(text) for p in keyword_patterns.values())

# ---------- Main evaluation ----------
def evaluate():
    if not QUESTIONS_PATH.exists() or not PDF_PATH.exists():
        print(f"Error: Data files not found in {DATA_DIR}")
        return

    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    print(f"Loading PDF and creating chunks from {PDF_PATH.name}...")
    
    embed_model = Embeddings()
    chunks = load_chunks(str(PDF_PATH), embed_model=embed_model)
    print(f"Total chunks created: {len(chunks)}")

    vector_store = VectorStore()
    vector_store.build(str(PDF_PATH))
    
    retriever = Retriever(store=vector_store)
    reranker = Reranker()

    report_results = []
    metrics_by_level = defaultdict(list)
    answerable_sims, unanswerable_sims = [], []

    print("\nEvaluating questions...")
    for question in questions:
        query = question["question"]
        expected_keywords = question.get("expected_keywords", [])
        answerable = question.get("answerable", True)
        level = question.get("level", "unspecified")

        keyword_patterns = {
            kw: build_keyword_pattern(kw) for kw in expected_keywords
        }

        candidates = retriever.retrieve_many([query], top_k=TOP_K_RETRIEVE)
        
        top_chunks = reranker.rerank(query, candidates, top_k=TOP_K_FINAL)

        keyword_details = []
        reciprocal_ranks = []

        for keyword, pattern in keyword_patterns.items():
            first_rank = None
            for rank, chunk in enumerate(top_chunks, start=1):
                if pattern.search(chunk["text"]):
                    first_rank = rank
                    break

            rr = 1.0 / first_rank if first_rank is not None else 0.0
            reciprocal_ranks.append(rr)
            keyword_details.append({
                "keyword": keyword,
                "first_rank": first_rank,
                "reciprocal_rank": round(rr, 4),
                "found": first_rank is not None,
            })

        keyword_mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None
        keyword_coverage = (
            sum(d["found"] for d in keyword_details) / len(keyword_details)
            if keyword_details else None
        )

        multi_hop = None
        if len(keyword_patterns) > 1:
            multi_hop = any(
                chunk_contains_all_keywords(c["text"], keyword_patterns)
                for c in top_chunks
            )

        top1_sim = top_chunks[0].get("rerank_score", 0.0) if top_chunks else None

        result = {
            "id": question["id"],
            "level": level,
            "question": query,
            "answerable": answerable,
            "keyword_mrr_at_k": round(keyword_mrr, 4) if keyword_mrr is not None else None,
            "keyword_coverage": round(keyword_coverage, 4) if keyword_coverage is not None else None,
            "multi_hop_success": multi_hop,
            "top1_similarity": round(top1_sim, 4) if top1_sim is not None else None,
            "keyword_details": keyword_details,
            "top_chunks": [],
        }

        for rank, chunk in enumerate(top_chunks, start=1):
            meta = chunk.get("metadata", {})
            result["top_chunks"].append({
                "rank": rank,
                "similarity": round(chunk.get("rerank_score", 0.0), 4),
                "page": meta.get("page"),
                "type": meta.get("type", "unknown"),
                "matched_keywords": find_matched_keywords(chunk["text"], keyword_patterns),
                "text_preview": chunk["text"][:200],
            })

        if keyword_mrr is not None:
            metrics_by_level[level].append({
                "mrr": keyword_mrr,
                "coverage": keyword_coverage,
                "multi_hop": multi_hop,
            })

        if top1_sim is not None:
            (answerable_sims if answerable else unanswerable_sims).append(top1_sim)

        report_results.append(result)

    all_mrrs = [it["mrr"] for items in metrics_by_level.values() for it in items]
    all_covs = [it["coverage"] for items in metrics_by_level.values() for it in items]
    all_mh = [it["multi_hop"] for items in metrics_by_level.values() for it in items if it["multi_hop"] is not None]

    mean_mrr = float(np.mean(all_mrrs)) if all_mrrs else 0.0
    mean_cov = float(np.mean(all_covs)) if all_covs else 0.0
    mh_rate = float(np.mean(all_mh)) if all_mh else None

    print("\n" + "=" * 60)
    print("ACTUAL PIPELINE (HYBRID + RERANK) EVALUATION")
    print("=" * 60)
    print(f"Overall MRR@{TOP_K_FINAL}:       {mean_mrr:.3f}")
    print(f"Overall coverage:        {mean_cov:.1%}")
    if mh_rate is not None:
        print(f"Multi-hop success rate:  {mh_rate:.1%}")


    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump({"config": {"mrr": mean_mrr, "coverage": mean_cov}, "questions": report_results}, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {REPORT_PATH}")

if __name__ == "__main__":
    evaluate()