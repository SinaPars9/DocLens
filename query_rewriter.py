import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("AVALAI_API_KEY")
BASE_URL = "https://api.avalai.ir/v1"
MODEL = "gpt-4o-mini"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    max_retries=3,
    timeout=90.0
)

def resolve_and_expand(question, chat_history):
    if not API_KEY:
        raise ValueError("KILOCODE_API_KEY was not found in the .env file.")

    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in chat_history
    )

    system_prompt = """
You are a query analysis component for a conversational RAG system.
Given the conversation history and the latest user question, do TWO things simultaneously:

1. RESOLVE: Rewrite the question as a standalone query using the chat history.
2. EXPAND: Generate 3 alternative search queries with different wording to improve retrieval.

Rules for RESOLVE:
- Use chat history to resolve references ("it", "what about 2023").
- If the question is already standalone, return it unchanged.
- If it is genuinely ambiguous (multiple distinct interpretations), set status to "ambiguous" and list them.

Rules for EXPAND:
- Generate exactly 3 alternative phrasings that preserve the meaning.

Return ONLY a valid JSON object with this exact schema:
{
  "status": "resolved" | "ambiguous" | "unclear",
  "standalone": "<the resolved query, if status is resolved>",
  "ambiguous_queries": ["<q1>", "<q2>"],
  "expanded": ["<variant1>", "<variant2>", "<variant3>"],
  "needs_clarification": true | false
}
"""

    user_prompt = f"""
Conversation history:
{history_text or "(No previous conversation)"}

Latest question:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content

    if not content:
        return {
            "status": "resolved",
            "standalone": question,
            "ambiguous_queries": [],
            "expanded": [],
            "needs_clarification": False
        }

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return {
            "status": "resolved",
            "standalone": question,
            "ambiguous_queries": [],
            "expanded": [],
            "needs_clarification": False
        }

    return result


def create_rerank_query(original_question, queries, chunks):
    if not API_KEY:
        raise ValueError("KILOCODE_API_KEY was not found in the .env file.")
        
    system_prompt = f"""
You are a query rewriting assistant for a RAG system.

Create ONE standalone query for reranking retrieved document chunks.

Original user question:
{original_question}

Possible interpretations:
{queries}

Retrieved document chunks:
{chunks}

Return only the query text. No explanations.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0,
    )
    
    content = response.choices[0].message.content

    if not content:
        return original_question
        
    return content.strip()