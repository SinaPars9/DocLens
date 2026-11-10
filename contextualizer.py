import os
import hashlib
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from diskcache import Cache

load_dotenv()

API_KEY = os.getenv("1XAI_API_KEY")
BASE_URL = "https://1xai.ir/v1"
MODEL = "gpt-4o-mini"

client = OpenAI(
    api_key=API_KEY or "missing_key", base_url=BASE_URL, max_retries=2, timeout=30.0
)


context_cache = Cache("context_cache")


def _get_cache_key(doc_text: str, chunk_text: str) -> str:
    data = doc_text + chunk_text
    return hashlib.md5(data.encode()).hexdigest()


def generate_chunk_context(document_text: str, chunk_text: str) -> str:
    cache_key = _get_cache_key(document_text, chunk_text)

    if cache_key in context_cache:
        return context_cache[cache_key]  # type: ignore

    context_doc = document_text[:1500] if len(document_text) > 1500 else document_text

    prompt = f"""
    <document>
    {context_doc}
    </document>

    Here is a chunk from this document:
    <chunk>
    {chunk_text}
    </chunk>

    Please provide a brief, succinct context (2-3 sentences) to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. 
    Answer ONLY with the succinct context and nothing else. Do not use phrases like "This chunk" or "The document". Just state the facts about what the chunk is about and where it fits.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )

        context = response.choices[0].message.content
        final_context = context.strip() if context else ""

        context_cache[cache_key] = final_context
        return final_context

    except Exception as e:
        print(f"⚠️ Contextualization failed: {e}")
        return ""
