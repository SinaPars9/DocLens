import os
import hashlib
from typing import Optional, Generator, List, Dict, Any, cast
from diskcache import Cache
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("1XAI_API_KEY")
BASE_URL = "https://1xai.ir/v1"
MODEL = "claude-haiku-5-5"

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
- Answer the user's TARGET QUESTION using ONLY the provided document context and chat history.
- Use chat history to resolve pronouns and follow-up questions naturally.

Calculation & Reasoning Rules:
- If the user asks for a calculated metric and the context provides the raw numbers, YOU MUST perform the calculation yourself.
- STRICT MATH RULE: For any calculation, you MUST quote the exact raw numbers from the context first, show the formula, and then calculate. Example: "According to the context, Revenue is £2,400,000 and COGS is £1,416,000. Gross Profit = £2,400,000 - £1,416,000 = £984,000."
- Distinguish clearly between facts stated in the text and results you calculated.

STRICT CONTEXT RULES:
- If the TARGET QUESTION specifies a particular condition (like a specific year), you must ONLY use data from that condition. Do NOT use pre-calculated values from other contexts.
- ONLY say "The information is not in the provided context" if the required raw data is completely missing.
CRITICAL FINANCIAL RULE: 
- Do NOT confuse different financial metrics. "Operating Expenses" is a specific line item on the income statement (e.g., £708,000). It is NOT the same as "Marketing Spend" or "Cost of Goods Sold". Extract the exact metric requested.
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
        "generate_v13", standalone_query, context, str(chat_history)
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
            "content": f"""<original_question>
{question}
</original_question>

<target_question>
{standalone_query}
</target_question>

<context>
{context}
</context>
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