import sys
import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")
from pipeline import RAGPipeline


API_KEY = os.getenv("1XAI_API_KEY")
BASE_URL = "https://1xai.ir/v1"
MODEL = "claude-haiku-5-5"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

DATA_DIR = BASE_DIR / "data"
QUESTIONS_PATH = DATA_DIR / "evaluation_questions.json"

def ask_judge(question, answer, contexts):

    
    context_text = "\n---\n".join(contexts)
    
    prompt = f"""
You are a STRICT Financial Auditor. Evaluate the RAG system's Answer based on the Context.

CRITICAL INSTRUCTIONS:
1. Identify the specific metric the question asks for (e.g., "Operating Expenses", "Revenue", "COGS").
2. Check if the Answer used the correct raw numbers for THAT SPECIFIC METRIC from the Context.
3. Check if the math in the Answer is correct.
4. If the Answer used the wrong metric (e.g., used Marketing Spend for Operating Expenses), Faithfulness MUST be 0.0.
5. If the math is correct and uses the right numbers, Faithfulness MUST be 1.0. (Do not penalize the model for doing math).

Return ONLY a valid JSON object with this exact structure:
{{
  "faithfulness_score": <float between 0 and 1>,
  "relevancy_score": <float between 0 and 1>,
  "reasoning": "<Explain which numbers were used and why you gave this score>"
}}

Question: {question}
Context:
{context_text}
Answer: 
{answer}
"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={ "type": "json_object" }
        )
        import json as json_lib
        result = json_lib.loads(response.choices[0].message.content) # type: ignore
        return (
            result.get("faithfulness_score", 0), 
            result.get("relevancy_score", 0),
            result.get("reasoning", "No reasoning provided") 
        )
    except Exception as e:
        print(f"Judge Error: {e}")
        return 0, 0, "Error"

def main():
    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        questions = json.load(f)
    
    hard_questions = [q for q in questions if q["level"] in ["hard", "unanswerable"]][:5]
    questions = hard_questions 

    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    if pipeline.vector_store.collection.count() == 0:
        pipeline.build(str(DATA_DIR / "test.pdf"))

    total_f = 0
    total_r = 0

    print("\nStarting LLM Judge Evaluation...")
    for idx, q in enumerate(questions):
        q_text = q["question"]
        print(f"\n[{idx+1}/5] Q: {q_text}")
        
        
        result = pipeline.answer_stream(q_text, chat_history=[])
        answer = ""
        contexts = []
        for event in result:
            if event["type"] == "text":
                answer += event["content"]
            elif event["type"] == "chunks":
                contexts = [c["text"] for c in event["chunks"]]
                print(f"  [Retrieved Chunks]")
                for c in event["chunks"]:
                    meta = c.get("metadata", {})
                    score = c.get("rerank_score", 0.0)
                    print(f"    - Type: {meta.get('type')} | Page: {meta.get('page')} | Score: {score:.2f}")
        print(f"  ANSWER: {answer}")

        f_score, r_score, reasoning = ask_judge(q_text, answer, contexts)
        total_f += f_score
        total_r += r_score
        
        print(f"  Faithfulness: {f_score:.2f} | Relevancy: {r_score:.2f}")
        print(f"  Judge Reasoning: {reasoning}") 
        print("-" * 50)

if __name__ == "__main__":
    main()