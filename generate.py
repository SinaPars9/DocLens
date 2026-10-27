import os
import hashlib
from typing import Optional, Generator, List, Dict, Any, cast
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


SYSTEM_PROMPT = """
You are an expert document analysis assistant.

Instructions:
- Answer the user's question using ONLY the provided document context and chat history.
- Use chat history to resolve pronouns and follow-up questions naturally.

Calculation & Reasoning Rules:
- If the user asks for a calculated metric and the context provides the raw numbers, YOU MUST perform the calculation yourself. Show your step-by-step reasoning.
- Distinguish clearly between facts stated in the text and results you calculated.

STRICT CONTEXT RULES:
- CRITICAL: You MUST answer the 'RESOLVED TARGET QUESTION'. If that target question specifies a particular condition (like a specific year, entity, or parameter), you must ONLY use data from that condition. 
- Do NOT use pre-calculated values from other contexts if they don't match the target question. If you have the raw numbers for the target condition, calculate it yourself.
- ONLY say "The information is not in the provided context" if the required raw data for the target condition is completely missing.

Output Style:
- Be direct, concise, and helpful. Format your answer for readability.
"""


def generate_answer_stream(
    question: str,
    standalone_query: str,
    chunks: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Generator[str, None, None]:
    if not API_KEY:
        raise ValueError("API_KEY was not found in the .env file.")

    context = "\n\n".join(chunk["text"] for chunk in chunks)
    cache_key = get_cache_key(
        "generate_v12", standalone_query, context, str(chat_history)
    )

    if cache_key in cache:
        print("[Cache Hit] Generate Answer (Stream)")
        yield cast(str, cache[cache_key])
        return

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)

    messages.append(
        {
            "role": "user",
            "content": f"""Original user question: {question}
RESOLVED TARGET QUESTION: {standalone_query}

Document context:
{context}

CRITICAL INSTRUCTION: You must answer the RESOLVED TARGET QUESTION. If it mentions a specific year or condition, calculate or extract the answer for that exact condition using the raw data provided in the context. Do not substitute data from other years.
""",
        }
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,  # type: ignore
        temperature=0.1,
        stream=True,
    )

    full_response = ""
    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            content = delta.content
            full_response += content
            yield content

    cache[cache_key] = full_response
