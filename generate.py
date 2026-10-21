import os
import hashlib
from typing import Optional, Generator, List, Dict, Any, cast
from diskcache import Cache
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("AVALAI_API_KEY")
BASE_URL = "https://api.avalai.ir/v1"
MODEL = "gpt-4o-mini"

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

SYSTEM_PROMPT = """
You are a document analysis assistant.

Instructions:
- Answer the user's latest question using the document context and chat history.
- Use chat history to resolve references to earlier topics, entities, and metrics.
- If the user asks a follow-up question, preserve the subject and metric from the previous relevant exchange unless the user changes them.
- Do not ask for clarification if the intended question can reasonably be inferred from the conversation.
- If the answer requires calculations, calculate them using the available data.
- Show the calculation when relevant.
- Distinguish explicitly stated facts from calculated results.
- Never invent facts or unsupported numbers.
- If the available information is insufficient to answer, clearly state what is missing.
- If the documents contain conflicting information, mention the conflict.
- Answer directly and concisely. Do not list unrelated information unless requested.
- CRITICAL: Pay strict attention to years and dates. If the user asks for a specific year (e.g., 2023), and that year is NOT explicitly stated in the context with the requested metric, you MUST say "The information for that year is not in the provided context." DO NOT use data from other years (like 2024 or 2025) as a substitute.
"""

def generate_answer(question: str, chunks: List[Dict[str, Any]], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    if not API_KEY:
        raise ValueError("AVALAI_API_KEY was not found in the .env file.")

    context = "\n\n".join(chunk["text"] for chunk in chunks)
    cache_key = get_cache_key("generate", question, context, str(chat_history))
    
    if cache_key in cache:
        print("[Cache Hit] Generate Answer")
        return cast(str, cache[cache_key])

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({
        "role": "user",
        "content": f"Document context:\n{context}\n\nQuestion:\n{question}",
    })

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages, # type: ignore
        temperature=0.2,
    )
    
    content = response.choices[0].message.content
    result = content if content else "No answer was produced."
    
    cache[cache_key] = result
    return result


def generate_answer_stream(
    question: str, 
    chunks: List[Dict[str, Any]], 
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Generator[str, None, None]:
    if not API_KEY:
        raise ValueError("AVALAI_API_KEY was not found in the .env file.")

    context = "\n\n".join(chunk["text"] for chunk in chunks)
    cache_key = get_cache_key("generate", question, context, str(chat_history))

    if cache_key in cache:
        print("[Cache Hit] Generate Answer (Stream)")
        yield cast(str, cache[cache_key])
        return

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({
        "role": "user",
        "content": f"Document context:\n{context}\n\nQuestion:\n{question}",
    })

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages, # type: ignore
        temperature=0.2,
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