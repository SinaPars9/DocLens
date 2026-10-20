import os
import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("AVALAI_API_KEY")
BASE_URL = "https://api.avalai.ir/v1"
MODEL = "gpt-4o-mini"


def generate_answer(question, chunks, chat_history=None):
    if not API_KEY:
        raise ValueError("KILOCODE_API_KEY was not found in the .env file.")
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
    context = "\n\n".join(
        chunk["text"] for chunk in chunks
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    if chat_history:
        messages.extend(chat_history)

    messages.append({
        "role": "user",
        "content": f"""
Document context:
{context}

Question:
{question}
"""
    })
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content
