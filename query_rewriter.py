import os
import json
import hashlib
from typing import Dict, Any, List, cast
from diskcache import Cache
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("1XAI_API_KEY")
BASE_URL = "https://1xai.ir/v1"
MODEL = "gpt-4o-mini"

client = OpenAI(
    api_key=API_KEY or "missing_key", base_url=BASE_URL, max_retries=3, timeout=90.0
)

cache = Cache("llm_cache")


def get_cache_key(prefix: str, *args: Any) -> str:
    data = prefix + str(args)
    return hashlib.md5(data.encode()).hexdigest()


def resolve_and_expand(
    question: str, chat_history: List[Dict[str, str]]
) -> Dict[str, Any]:
    if not API_KEY:
        raise ValueError("API_KEY was not found in the .env file.")

    cache_key = get_cache_key("resolve_v6", question, str(chat_history))

    if cache_key in cache:
        print("[Cache Hit] Resolve + Expand")
        return cast(Dict[str, Any], cache[cache_key])

    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in chat_history)

    system_prompt = """
You are an expert query resolution assistant for a conversational RAG system.
Your task is to rewrite the user's latest question into a complete, standalone query by inferring the intended context from the chat history.

CRITICAL RULES:
1. Analyze the CONVERSATIONAL FLOW: Determine the "Active Topic" (e.g., the specific time period, entity, parameter, or condition currently being discussed). 
2. If the user asks a short follow-up question (e.g., "what about the other one?", "how about in [Year]?", "what is its [Attribute]?"), infer the missing context from the most logical Active Topic in the dialogue.
3. Do not blindly assume; if the user explicitly shifts context to a different entity, update the Active Topic.
4. Generate exactly 3 alternative search queries based on the resolved standalone query to improve document retrieval.

GENERAL EXAMPLES (Abstract structure, apply to any domain):
Example 1 (Borrowing the most recent condition):
History:
User: What is the [Subject A] for [Condition 1]?
Assistant: The [Subject A] for [Condition 1] is [Value].
User: What about [Condition 2]?
Assistant: The [Subject A] for [Condition 2] is [Value 2].
Latest: What about the [Subject B]?
Standalone: "What is the [Subject B] for [Condition 2]?" 
*(Reasoning: The model infers the active condition is [Condition 2] based on the immediate conversational flow, not [Condition 1]).*

Example 2 (Entity shift):
History:
User: Tell me about [Entity 1].
Assistant: [Entity 1] has [Attribute].
Latest: What about [Entity 2]?
Standalone: "Tell me about [Entity 2]."

Return ONLY a valid JSON object with this exact structure:
{
  "standalone": "<the fully resolved standalone query>",
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
