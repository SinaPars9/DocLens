
import json
import os
import httpx
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("KILOCODE_API_KEY")
BASE_URL = "https://api.kilo.ai/api/gateway"
MODEL = "kilo-auto/free"

def resolve_query(question, chat_history):
    if not API_KEY:
        raise ValueError("KILOCODE_API_KEY was not found in the .env file.")

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in chat_history
    )

    system_prompt = """
You are a query resolution component for a conversational AI system.

Your task is to interpret the user's latest question using the conversation history and produce standalone queries that preserve the user's intended meaning.

Rules:
1. Use the conversation history to resolve references such as "it", "that one", "the other option", "the previous one", or "what about 2023".
2. Preserve the user's original intent. Do not invent facts, preferences, entities, or intentions that are not sufficiently supported by the conversation.
3. Return "resolved" with exactly one standalone query when the intended meaning is sufficiently clear from the latest question and conversation history.
4. Return "ambiguous" with multiple standalone queries when two or more materially different interpretations are reasonably supported by the context. Include the plausible interpretations without arbitrarily choosing one.
5. Return "unclear" ONLY when the question contains unresolved references (e.g., "it", "that", "the other one") that CANNOT be resolved from the conversation history, OR when the question has no clear information need at all (e.g., "hmm", "what?", "ok so").
6. A self-contained question with a clear information need is ALWAYS "resolved", even if:
   - It does not mention a specific document, company, or dataset.
   - It does not include a year, date, or location that would normally provide context.
   - The retrieval system could handle the missing context on its own.
7. If the latest question can be understood WITHOUT the conversation history (i.e., it stands on its own), return "resolved" with that question as the only query. Do NOT mark such questions as "unclear".
8. Do not treat every short or context-dependent question as ambiguous. If the conversation clearly establishes its meaning, resolve it directly.
9. When returning multiple queries, each must represent a distinct plausible interpretation, not merely a rewording of the same interpretation.
10. Keep the original scope and intent of the question. Do not answer it, perform the requested task, or introduce unrelated topics.
11. Return only a valid JSON object using this schema:
{
  "status": "resolved" | "ambiguous" | "unclear",
  "queries": ["..."],
  "needs_clarification": true | false
}

Examples:

--- Example 1 (resolved, no history) ---
History: (none)
Question: "What was the revenue in 2023?"
Output:
{"status": "resolved", "queries": ["What was the revenue in 2023?"], "needs_clarification": false}

--- Example 2 (resolved, with history) ---
History:
  user: What was the revenue in 2023?
  assistant: Revenue in 2023 was £1,850,000.
Question: "What about 2024?"
Output:
{"status": "resolved", "queries": ["What was the revenue in 2024?"], "needs_clarification": false}

--- Example 3 (ambiguous) ---
History:
  user: Tell me about the East region.
  assistant: The East region contributed 15% of revenue.
Question: "What about it?"
Output:
{"status": "ambiguous", "queries": ["What about the East region?", "What about the previous topic?"], "needs_clarification": false}

--- Example 4 (unclear) ---
History: (none)
Question: "What about it?"
Output:
{"status": "unclear", "queries": [], "needs_clarification": true}
"""""
    user_prompt = f"""
Conversation history:
{history_text or "(No previous conversation)"}

Latest user question:
{question}
"""

    response = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        },
        timeout=httpx.Timeout(90.0, connect=15.0),
    )

    response.raise_for_status()
    data = response.json()

    choices = data.get("choices")
    if not choices:
        raise ValueError(f"Unexpected API response: {data}")

    content = choices[0].get("message", {}).get("content")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned an empty response.")

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"The model did not return valid JSON: {content}"
        ) from exc

    status = result.get("status")
    queries = result.get("queries")
    needs_clarification = result.get("needs_clarification")

    if status not in {"resolved", "ambiguous", "unclear"}:
        raise ValueError(f"Invalid status: {status}")

    if not isinstance(queries, list) or not all(
        isinstance(query, str) and query.strip()
        for query in queries
    ):
        raise ValueError("The queries field must contain non-empty strings.")

    if not isinstance(needs_clarification, bool):
        raise ValueError("needs_clarification must be a boolean.")

    if status == "resolved" and len(queries) != 1:
        raise ValueError("A resolved result must contain exactly one query.")

    if status == "ambiguous" and len(queries) < 2:
        raise ValueError("An ambiguous result must contain at least two queries.")

    if status == "unclear" and (
        queries or not needs_clarification
    ):
        raise ValueError("An unclear result must request clarification.")

    return {
        "status": status,
        "queries": [query.strip() for query in queries],
        "needs_clarification": needs_clarification,
    }



def expand_query(question):
    if not API_KEY:
        raise ValueError("KILOCODE_API_KEY was not found in the .env file.")
    prompt = f"""
    Generate 3 alternative search queries for the question below.
    The queries should:
    - Preserve the original meaning.
    - Use different wording where useful.
    - Be suitable for retrieving relevant passages from financial documents.
    - Return only a JSON object in this format:
      {{"queries": ["query 1", "query 2", "query 3"]}}

    Original question:
    {question}
    """
    response = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": prompt},
            ],
            "temperature": 0,
        },
        timeout=httpx.Timeout(90.0, connect=15.0),
    )

    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)

    queries = parsed["queries"]
    return queries

def create_rerank_query(original_question,queries,chunks):
    if not API_KEY:
        raise ValueError("KILOCODE_API_KEY was not found in the .env file.")
    system_prompt = f"""
You are a query rewriting assistant for a RAG system.

Your task is to create one standalone query for reranking retrieved document chunks.

You will receive:
- The original user question
- Multiple possible interpretations of that question
- Retrieved document chunks

Instructions:
1. Use the original question and interpretations to understand the user's information need.
2. Use the retrieved chunks to identify the relevant subject and terminology.
3. Generate one standalone query suitable for a cross-encoder reranker.
4. Preserve the original user's intent. Do not invent facts or assume an interpretation is correct without sufficient evidence.
5. If multiple interpretations remain plausible, create a query that covers their shared information need without arbitrarily choosing one.
6. Keep the query specific, concise, and understandable without chat history.
7. Return only the query text. Do not include explanations, alternatives, or JSON.
original user question: 
 {original_question}
possible interpretations of that question:
{queries}
Retrieved document chunks:
{chunks}
"""
    response = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
            ],
            "temperature": 0,
        },
        timeout=httpx.Timeout(90.0, connect=15.0),
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()
    return content


    


