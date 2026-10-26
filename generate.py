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

CRITICAL RULES:
1. Pay strict attention to the chat history. If the user asks a follow-up question, the time period, subject, or condition from their previous question is STILL the active topic. DO NOT use data from different contexts.
2. If the user asks for a derived value or calculation (e.g., a ratio, difference, or percentage) and the context provides the raw data, YOU MUST perform the calculation yourself. Show your step-by-step reasoning.
3. Look at the provided text and tables carefully. Extract the specific values requested and calculate if needed.

Example:
If the history is about "Year 2022", and the user asks "what about the total?", you MUST calculate the total for 2022, NOT any other year.

Instructions:
- Answer using ONLY the provided context and chat history.
- Never invent facts or numbers.
- Be concise and show calculations when relevant.
"""


def generate_answer_stream(
    question: str,
    chunks: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Generator[str, None, None]:
    if not API_KEY:
        raise ValueError("API_KEY was not found in the .env file.")

    context = "\n\n".join(chunk["text"] for chunk in chunks)
    cache_key = get_cache_key("generate_v6", question, context, str(chat_history))

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
            "content": f"Document context:\n{context}\n\nQuestion:\n{question}",
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
