import json
import re
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


# ---------- Configuration ----------

BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = BASE_DIR / "evaluation_questions_v2.json"
REPORT_PATH = BASE_DIR / "retrieval_evaluation_report.json"


MODEL_DIR = Path(r"E:\models\bekko-embedding-v1-a8m")
ONNX_PATH = MODEL_DIR / "onnx" / "model.onnx"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

MAX_LENGTH = 1024         
EMBED_BATCH_SIZE = 16    
TOP_K = 4


# ---------- ONNX model ----------

print("Loading ONNX model...")
t0 = time.time()

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4
sess_options.inter_op_num_threads = 1

session = ort.InferenceSession(
    str(ONNX_PATH),
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
tokenizer.enable_truncation(max_length=MAX_LENGTH)

input_names = [inp.name for inp in session.get_inputs()]
output_names = [out.name for out in session.get_outputs()]

print(f"  model loaded in {time.time() - t0:.1f}s")
print(f"  inputs:  {input_names}")
print(f"  outputs: {output_names}")


# ---------- Embedding function ----------

def embed_texts(texts):
    """
    Embed a list of texts with the local ONNX model.

    Returns a numpy array of shape (len(texts), hidden_dim),
    L2-normalized for cosine similarity.
    """
    all_vectors = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        encodings = tokenizer.encode_batch(batch)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array(
            [e.attention_mask for e in encodings], dtype=np.int64
        )

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = session.run(None, inputs)
        last_hidden = outputs[0]  # (batch, seq_len, hidden_dim)


        mask_expanded = attention_mask[..., np.newaxis].astype(np.float32)
        sum_embeddings = (last_hidden * mask_expanded).sum(axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), 1e-9, None)
        embeddings = sum_embeddings / sum_mask

        all_vectors.append(embeddings)

        print(f"  embedded {min(i + EMBED_BATCH_SIZE, len(texts))}/{len(texts)}")

    arr = np.concatenate(all_vectors, axis=0).astype(np.float32)

    # L2 normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-9, None)


# ---------- Keyword matching ----------

def build_keyword_pattern(keyword):
    """
    Regex that respects numeric boundaries but allows trailing punctuation
    when it's not followed by a digit (e.g. end of sentence).

    Examples:
        '£800'  matches 'value of £800.'  but NOT '£800,000'
        '£5'    does NOT match '£500'
        '2.4'   matches '2.4 kg'          but NOT '2.40' or '12.4'
    """
    escaped = re.escape(str(keyword).strip())
    return re.compile(
        r"(?<![0-9A-Za-z,.])"        # char before: not alnum/comma/period
        + escaped +
        r"(?![0-9A-Za-z])"           # char after: not alnum
        r"(?![,.](?=\d))",           # char after: not , or . followed by digit
        re.IGNORECASE,
    )

def find_matched_keywords(text, keyword_patterns):
    return [kw for kw, p in keyword_patterns.items() if p.search(text)]


def chunk_contains_all_keywords(text, keyword_patterns):
    return all(p.search(text) for p in keyword_patterns.values())


def get_top_chunks(chunks, similarities, top_k):
    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    return (
        ranked_indices,
        [chunks[int(i)] for i in ranked_indices],
        [float(similarities[int(i)]) for i in ranked_indices],
    )


# ---------- Cache ----------

CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

CHUNKS_CACHE = CACHE_DIR / "chunks.pkl"
EMBED_CACHE = CACHE_DIR / "chunk_embeddings_bekko.npz"


def load_or_build_chunks():
    if CHUNKS_CACHE.exists():
        print("Loading chunks from cache...")
        with CHUNKS_CACHE.open("rb") as f:
            return pickle.load(f)

    print("Building chunks from PDF...")
    from chunk_pdf import load_chunks
    chunks = load_chunks()
    with CHUNKS_CACHE.open("wb") as f:
        pickle.dump(chunks, f)
    return chunks


def load_or_build_chunk_embeddings(chunks):
    if EMBED_CACHE.exists():
        data = np.load(EMBED_CACHE)
        if int(data["count"]) == len(chunks):
            print("Loading chunk embeddings from cache...")
            return data["vectors"]

    print("Embedding all chunks...")
    vectors = embed_texts([c.page_content for c in chunks])
    np.savez(EMBED_CACHE, vectors=vectors, count=len(chunks))
    return vectors


# ---------- Main ----------

def evaluate():
    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    chunks = load_or_build_chunks()
    if not chunks:
        raise ValueError("No chunks produced from the PDF.")

    total_chunks = len(chunks)

    print(f"Questions: {len(questions)}")
    print(f"Chunks:    {total_chunks}")
    print(f"Top-K:     {TOP_K} ({TOP_K / total_chunks:.1%})")
    print(f"Model:     {ONNX_PATH}")

    t0 = time.time()
    chunk_vectors = load_or_build_chunk_embeddings(chunks)
    print(f"Chunk embedding time: {time.time() - t0:.1f}s")

    print("\nEmbedding queries...")
    t0 = time.time()
    query_texts = [q["question"] for q in questions]
    query_vectors = embed_texts(query_texts)
    print(f"Query embedding time: {time.time() - t0:.1f}s")

    report_results = []
    metrics_by_level = defaultdict(list)
    answerable_sims, unanswerable_sims = [], []

    for q_idx, question in enumerate(questions):
        query = question["question"]
        expected_keywords = question.get("expected_keywords", [])
        answerable = question.get("answerable", True)
        level = question.get("level", "unspecified")

        keyword_patterns = {
            kw: build_keyword_pattern(kw) for kw in expected_keywords
        }

        query_vector = query_vectors[q_idx]
        similarities = chunk_vectors @ query_vector

        top_indices, top_chunks, top_sims = get_top_chunks(
            chunks, similarities, TOP_K
        )

        keyword_details = []
        reciprocal_ranks = []

        for keyword, pattern in keyword_patterns.items():
            first_rank = None
            for rank, chunk in enumerate(top_chunks, start=1):
                if pattern.search(chunk.page_content):
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

        keyword_mrr = (
            float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None
        )
        keyword_coverage = (
            sum(d["found"] for d in keyword_details) / len(keyword_details)
            if keyword_details else None
        )

        multi_hop = None
        if len(keyword_patterns) > 1:
            multi_hop = any(
                chunk_contains_all_keywords(c.page_content, keyword_patterns)
                for c in top_chunks
            )

        tables_in_top_k = sum(
            1 for c in top_chunks if c.metadata.get("type") == "table"
        )
        top1_sim = top_sims[0] if top_sims else None

        result = {
            "id": question["id"],
            "level": level,
            "question": query,
            "answerable": answerable,
            "keyword_mrr_at_k": (
                round(keyword_mrr, 4) if keyword_mrr is not None else None
            ),
            "keyword_coverage": (
                round(keyword_coverage, 4) if keyword_coverage is not None else None
            ),
            "multi_hop_success": multi_hop,
            "top1_similarity": (
                round(top1_sim, 4) if top1_sim is not None else None
            ),
            "tables_in_top_k": tables_in_top_k,
            "keyword_details": keyword_details,
            "top_chunks": [],
        }

        for rank, (chunk, sim) in enumerate(zip(top_chunks, top_sims), start=1):
            result["top_chunks"].append({
                "rank": rank,
                "similarity": round(sim, 4),
                "page": chunk.metadata.get("page"),
                "type": chunk.metadata.get("type", "unknown"),
                "matched_keywords": find_matched_keywords(
                    chunk.page_content, keyword_patterns
                ),
                "text_preview": chunk.page_content[:200],
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

    all_mrrs, all_covs, all_mh = [], [], []
    for items in metrics_by_level.values():
        for it in items:
            all_mrrs.append(it["mrr"])
            all_covs.append(it["coverage"])
            if it["multi_hop"] is not None:
                all_mh.append(it["multi_hop"])

    mean_mrr = float(np.mean(all_mrrs)) if all_mrrs else 0.0
    mean_cov = float(np.mean(all_covs)) if all_covs else 0.0
    mh_rate = float(np.mean(all_mh)) if all_mh else None
    ans_sim = float(np.mean(answerable_sims)) if answerable_sims else None
    unans_sim = float(np.mean(unanswerable_sims)) if unanswerable_sims else None

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Overall MRR@{TOP_K}:       {mean_mrr:.3f}")
    print(f"Overall coverage:        {mean_cov:.1%}")
    if mh_rate is not None:
        print(f"Multi-hop success rate:  {mh_rate:.1%}")
    if ans_sim is not None:
        print(f"Top-1 sim (answerable):  {ans_sim:.3f}")
    if unans_sim is not None:
        print(f"Top-1 sim (unanswerable):{unans_sim:.3f} (lower is better)")

    print("\n--- By difficulty ---")
    for level in ["easy", "medium", "hard", "challenging", "unanswerable"]:
        if level not in metrics_by_level:
            continue
        items = metrics_by_level[level]
        lvl_mrrs = [i["mrr"] for i in items]
        lvl_covs = [i["coverage"] for i in items]
        lvl_mh = [i["multi_hop"] for i in items if i["multi_hop"] is not None]
        mh_text = f"  MultiHop={np.mean(lvl_mh):.1%}" if lvl_mh else ""
        print(
            f"{level:14s}: MRR={np.mean(lvl_mrrs):.3f}  "
            f"Coverage={np.mean(lvl_covs):.1%}  (n={len(items)}){mh_text}"
        )

    report = {
        "config": {
            "embedding_model": "bekko-embedding-v1-a8m",
            "provider": "local-onnx",
            "top_k": TOP_K,
            "total_chunks": total_chunks,
            "total_questions": len(questions),
            "keyword_mrr_at_k": mean_mrr,
            "mean_keyword_coverage": mean_cov,
            "multi_hop_success_rate": mh_rate,
            "mean_top1_similarity_answerable": ans_sim,
            "mean_top1_similarity_unanswerable": unans_sim,
        },
        "by_level": {
            level: {
                "count": len(items),
                "mean_mrr": float(np.mean([i["mrr"] for i in items])),
                "mean_coverage": float(np.mean([i["coverage"] for i in items])),
                "multi_hop_rate": (
                    float(np.mean([
                        i["multi_hop"] for i in items if i["multi_hop"] is not None
                    ]))
                    if any(i["multi_hop"] is not None for i in items)
                    else None
                ),
            }
            for level, items in metrics_by_level.items()
        },
        "questions": report_results,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    evaluate()