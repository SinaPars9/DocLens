import json
import os
import sys
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from diskcache import Cache

sys.path.append(str(Path(__file__).parent.parent))

from pipeline import RAGPipeline

load_dotenv()

SKIP_BUILD = False         
USE_LLM_CONTEXT = False    

JUDGE_API_KEY = os.getenv("1XAI_API_KEY")
JUDGE_CLIENT = OpenAI(
    api_key=JUDGE_API_KEY or "missing_key",
    base_url="https://1xai.ir/v1",
    max_retries=2,
    timeout=30.0,
)
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_CACHE = Cache("eval_judge_cache")  


def judge_answer(question: str, expected: str, generated: str, q_type: str) -> bool:
    """Use LLM to judge if the generated answer matches the expected answer."""
    
    if q_type == "abstention":
        ans_lower = generated.lower()
        abstention_keywords = [
            "not in the provided context", "doesn't include", "does not include",
            "does not contain", "cannot find", "no information",
            "not seem to contain", "don't know", "not mentioned", "not specified",
            "not available", "no relevant",
        ]
        return any(kw in ans_lower for kw in abstention_keywords)

    cache_key = hashlib.md5(f"{question}|{expected}|{generated}".encode()).hexdigest()
    if cache_key in JUDGE_CACHE:
        return JUDGE_CACHE[cache_key] #type:ignore

    prompt = f"""You are a strict answer evaluator for a RAG system.

Question: {question}
Expected Answer: {expected}
Generated Answer: {generated}

Does the Generated Answer convey the same factual meaning as the Expected Answer?
- Focus on numbers, entities, and direction (increase/decrease).
- Ignore minor wording differences.
- If the generated answer contains extra context but includes the expected fact, it's CORRECT.

Respond ONLY with a single word: CORRECT or INCORRECT.
"""

    try:
        response = JUDGE_CLIENT.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        verdict = response.choices[0].message.content.strip().upper() #type:ignore
        is_correct = "CORRECT" in verdict and "INCORRECT" not in verdict
    except Exception as e:
        print(f"  ⚠️ Judge failed: {e}")
        is_correct = False

    JUDGE_CACHE[cache_key] = is_correct
    return is_correct


def load_questions(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_content_hash(pdf_path: str) -> str:
    """Compute MD5 hash of the PDF content."""
    hasher = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


def run_evaluation():
    questions_file = Path(__file__).parent / "questions.json"
    data_dir = Path(__file__).parent / "data"
    questions = load_questions(questions_file)

    doc_groups = {}
    for q in questions:
        doc = q["document"]
        doc_groups.setdefault(doc, []).append(q)

    rag = RAGPipeline()

    
    metrics = {
        "retrieval_hits": 0,
        "verifier_passes": 0,
        "verifier_total": 0,  
        "answer_correct": 0,
        "total": 0,
        "by_category": {},  
    }

    def update_category(cat: str, key: str, is_hit: bool):
        if cat not in metrics["by_category"]:
            metrics["by_category"][cat] = {"total": 0, "hits": 0}
        metrics["by_category"][cat]["total"] += 1
        if is_hit:
            metrics["by_category"][cat]["hits"] += 1

    for doc_name, doc_questions in doc_groups.items():
        pdf_path = data_dir / doc_name
        if not pdf_path.exists():
            print(f"⚠️ File {doc_name} not found. Skipping...")
            continue

        if not SKIP_BUILD:
            print(f"\n{'='*50}")
            print(f"📄 Building Database for: {doc_name}")
            if USE_LLM_CONTEXT:
                print("⚠️ Using LLM for context generation (API cost applies)")
            else:
                print("⚡ Using Fast Mode (Free)")
            print(f"{'='*50}")
            rag.build(str(pdf_path), use_llm_context=USE_LLM_CONTEXT) #type:ignore
        else:
            print(f"\n⏩ Skipping Build for: {doc_name} (Using existing DB)")
            rag.document_id = compute_content_hash(str(pdf_path))
            rag.is_built = True

        print(f"\n📝 Running {len(doc_questions)} questions for {doc_name}...\n")

        for q in doc_questions:
            metrics["total"] += 1
            q_type = q.get("type", "simple")
            print(f"❓ Question [{q['id']}]: {q['question']}")

            chunks = []
            answer = ""
            verification = {}

            try:
                for event in rag.answer_stream(
                    q["question"],
                    selected_documents=[doc_name],  
                ):
                    if event["type"] == "chunks":
                        chunks = event["chunks"]
                    elif event["type"] == "text":
                        answer += event["content"]
                    elif event["type"] == "verification":
                        verification = event["data"]
            except Exception as e:
                print(f"  ❌ Pipeline error: {e}")
                continue

            retrieved_pages = [c["metadata"].get("page") for c in chunks]
            if q_type == "abstention":
               
                retrieval_hit = len(chunks) == 0 or len(retrieved_pages) <= 3
            else:
                retrieval_hit = any(p in retrieved_pages for p in q.get("expected_pages", []))

        
            verifier_pass = None
            if q_type != "abstention" and verification:
                metrics["verifier_total"] += 1
                verifier_pass = verification.get("is_fully_verified", False)

            
            expected_answer = q.get("expected_answer", "")
            answer_correct = judge_answer(
                q["question"], expected_answer, answer, q_type
            )

            
            if retrieval_hit:
                metrics["retrieval_hits"] += 1
            if verifier_pass:
                metrics["verifier_passes"] += 1
            if answer_correct:
                metrics["answer_correct"] += 1

            update_category(q_type, "hit", retrieval_hit)

            
            status_ret = "✅" if retrieval_hit else "❌"
            status_ans = "✅" if answer_correct else "❌"
            print(f"  {status_ret} Retrieval: {'PASS' if retrieval_hit else 'FAIL'} (Pages: {retrieved_pages[:5]})")
            if verifier_pass is not None:
                print(f"  {'✅' if verifier_pass else '❌'} Verifier: {'PASS' if verifier_pass else 'WARN'}")
            print(f"  {status_ans} Answer: {'CORRECT' if answer_correct else 'INCORRECT'}")
            print(f"  📄 Expected: {expected_answer[:80]}...")
            print(f"  📄 Got:      {answer[:80].replace(chr(10), ' ')}...\n")

    # 🌟 گزارش نهایی
    print(f"\n{'='*50}")
    print("🏁 EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total Questions: {metrics['total']}")
    if metrics["total"] > 0:
        print(f"Retrieval Hit Rate:   {metrics['retrieval_hits'] / metrics['total'] * 100:.1f}%")
        print(f"Answer Accuracy:      {metrics['answer_correct'] / metrics['total'] * 100:.1f}%")
        if metrics["verifier_total"] > 0:
            print(f"Verifier Pass Rate:   {metrics['verifier_passes'] / metrics['verifier_total'] * 100:.1f}% (on {metrics['verifier_total']} non-abstention Qs)")

    print(f"\n📊 Breakdown by Category:")
    for cat, data in metrics["by_category"].items():
        rate = data["hits"] / data["total"] * 100 if data["total"] else 0
        print(f"  - {cat}: {data['hits']}/{data['total']} ({rate:.1f}%)")

    
    report_path = Path(__file__).parent / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n💾 Report saved to: {report_path}")


if __name__ == "__main__":
    run_evaluation()