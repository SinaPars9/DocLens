import json
import os
import sys
import hashlib
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from pipeline import RAGPipeline

SKIP_BUILD = False


def load_questions(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation():
    questions_file = Path(__file__).parent / "questions.json"
    data_dir = Path(__file__).parent / "data"
    questions = load_questions(questions_file)

    doc_groups = {}
    for q in questions:
        doc = q["document"]
        if doc not in doc_groups:
            doc_groups[doc] = []
        doc_groups[doc].append(q)

    rag = RAGPipeline()

    total_retrieval_hits = 0
    total_verifier_passes = 0
    total_questions = 0

    for doc_name, doc_questions in doc_groups.items():
        pdf_path = data_dir / doc_name
        if not pdf_path.exists():
            print(f"⚠️ File {doc_name} not found in data folder. Skipping...")
            continue

        if not SKIP_BUILD:
            print(f"\n{'='*50}")
            print(f"📄 Building Database for: {doc_name} (This may take a while...)")
            print(f"{'='*50}")
            rag.build(str(pdf_path))
        else:
            print(f"\n⏩ Skipping Build for: {doc_name} (Using existing DB)")
            rag.document_id = hashlib.md5(str(pdf_path).encode()).hexdigest()[:8]

        print(f"\n📝 Running {len(doc_questions)} questions for {doc_name}...\n")

        for q in doc_questions:
            total_questions += 1
            print(f"❓ Question [{q['id']}]: {q['question']}")

            chunks = []
            answer = ""
            verification = {}

            for event in rag.answer_stream(q["question"]):
                if event["type"] == "chunks":
                    chunks = event["chunks"]
                elif event["type"] == "text":
                    answer += event["content"]
                elif event["type"] == "verification":
                    verification = event["data"]

            retrieved_pages = [c["metadata"].get("page") for c in chunks]

            if q["type"] == "abstention":
                ans_lower = answer.lower()
                abstention_keywords = [
                    "not in the provided context",
                    "doesn't include",
                    "does not include",
                    "does not contain",
                    "cannot find",
                    "no information",
                    "not seem to contain relevant information",
                    "don't know",
                ]
                if any(kw in ans_lower for kw in abstention_keywords):
                    hit = True
                else:
                    hit = False
            else:
                hit = any(page in retrieved_pages for page in q["expected_pages"])

            if hit:
                total_retrieval_hits += 1
                print(f"  ✅ Retrieval/Logic: PASS")
            else:
                print(
                    f"  ❌ Retrieval/Logic: FAIL (Expected Pages: {q['expected_pages']}, Got: {retrieved_pages})"
                )

            if verification:
                if verification.get("is_fully_verified", False):
                    total_verifier_passes += 1
                    print("  ✅ Verifier: PASS (All claims verified)")
                else:
                    print("  ❌ Verifier: WARNING (Some claims flagged)")

            clean_answer = answer[:150].replace("\n", " ")
            print(f"  📄 Answer snippet: {clean_answer}...\n")

    print(f"\n{'='*50}")
    print("🏁 EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total Questions: {total_questions}")
    if total_questions > 0:
        print(
            f"Retrieval/Logic Hit Rate: {(total_retrieval_hits / total_questions) * 100:.1f}%"
        )
        print(
            f"Verifier Pass Rate: {(total_verifier_passes / total_questions) * 100:.1f}%"
        )


if __name__ == "__main__":
    run_evaluation()
