import os
import json
import hashlib
from typing import Dict, Any, List, cast
from diskcache import Cache
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("HORMOZ_API_KEY")
BASE_URL = "https://api.hormouz.net/v1"
MODEL = "openai/gpt-4o-mini" 

client = OpenAI(
    api_key=API_KEY or "missing_key",
    base_url=BASE_URL,
    max_retries=3,
    timeout=90.0
)

cache = Cache("llm_cache")

def get_cache_key(prefix: str, *args: Any) -> str:
    data = prefix + str(args)
    return hashlib.md5(data.encode()).hexdigest()

def resolve_and_expand(question: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
    if not API_KEY:
        raise ValueError("API_KEY was not found in the .env file.")

    cache_key = get_cache_key("resolve_v6", question, str(chat_history))
    
    if cache_key in cache:
        print("[Cache Hit] Resolve + Expand")
        return cast(Dict[str, Any], cache[cache_key])

    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in chat_history)

    system_prompt = """
You are a query resolution assistant for a conversational RAG system.

CRITICAL TASK: Resolve the user's latest question using the chat history.
If the user asks a follow-up question, you MUST look at the PREVIOUS user question to find the missing context (like the Subject, Time Period, or Condition) and include it in the standalone query.

Example 1:
History: User asked "What is the value of X in 2023?"
Latest: "what about 2024?"
Standalone Output: "What is the value of X in 2024?"

Example 2:
History: User asked "Tell me about the Alpha component in System A."
Latest: "What about System B?"
Standalone Output: "Tell me about the Alpha component in System B."

Example 3:
History: User asked "What are the side effects of Drug Y?"
Latest: "What is the dosage?"
Standalone Output: "What is the dosage for Drug Y?"

Rule: ALWAYS output the resolved standalone query with the missing context included. 
Generate 3 alternative search queries based on the resolved query to improve retrieval.

Return ONLY a valid JSON object with this exact structure:
{
  "standalone": "<the resolved query>",
  "expanded": ["<var1>", "<var2>", "<var3>"]
}
"""

    user_prompt = f"Conversation history:\n{history_text or '(None)'}\n\nLatest question:\n{question}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content
    default_result = {"standalone": question, "expanded": []}

    if not content:
        result = default_result
    else:
        try:
            parsed = json.loads(content)
            result = parsed if "standalone" in parsed else default_result
        except json.JSONDecodeError:
            result = default_result

    cache[cache_key] = result
    return result