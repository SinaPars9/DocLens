# DocLens — Advanced PDF RAG Assistant

DocLens is an end-to-end Retrieval-Augmented Generation (RAG) system that lets users upload a PDF, retrieve relevant passages, and ask questions about its content through a conversational interface. 

This project focuses on building a **experimental / learning project** document analysis pipeline. It goes beyond basic RAG by implementing Hybrid Search, Smart Table Reranking, Streaming, and local ONNX model inference to run efficiently even on low-spec hardware.

## Features

- **Hybrid Search (BM25 + Vector):** Combines semantic vector search (ChromaDB) with keyword-based BM25 (using Reciprocal Rank Fusion) to drastically improve retrieval accuracy for numeric data, specific entities, and tables.
- **Document-Isolated Layered Caching:** A dual-cache system (exact-match disk cache + ChromaDB semantic cache). Both caches are strictly isolated by a unique `document_id` hash (generated per PDF). This prevents cross-document cache pollution when a user switches between different files during a session.
- **Advanced Hybrid Parsing & Empty Table Filter:** Uses PyMuPDF for text, Camelot's `lattice` parser for bordered tables, and pdfplumber as a fallback. Includes a regex-based filter to automatically drop "ghost tables" (empty blocks or graphical boxes containing only whitespace/non-alphanumeric characters), keeping the LLM context clean.
- **Streaming Responses:** Built with Streamlit, generating answers token-by-token for a ChatGPT-like user experience with near-zero perceived latency.
- **Layered Semantic Caching:** Uses a dual-cache system. An exact-match disk cache for LLM calls, plus a ChromaDB-based Semantic Cache that recognizes meaningfully similar queries (e.g., "How much did we make in 2023?" vs "2023 revenue") and returns answers instantly with zero API cost.
- **Batched ONNX Inference:** Optimizes local CPU inference for both embedding (`bekko-embedding`) and reranking (`ms-marco-MiniLM`) models, preventing memory spikes.
- **Domain-Agnostic Prompts:** LLM prompts are generalized to handle financial reports, academic papers, or technical manuals with equal accuracy.
- **Transparent UI:** Displays retrieved chunks, page numbers, and rerank scores in a dedicated context panel for full explainability.
- **Transparent Query Inspector:** The Streamlit UI features a collapsible panel that shows the user exactly how their follow-up question was resolved into a standalone query before hitting the database (with cache-hit bypasses).
- **Hybrid LLM Strategy:** Uses `gpt-4o-mini` for cost-effective query rewriting, and `Claude Haiku 5.5` for final generation to ensure zero hallucination, accurate math reasoning, and strict adherence to context.
- **Enhanced Dynamic Table Boosting:** The Cross-Encoder reranker dynamically extracts numbers/keywords from the query and applies a heavy penalty/boost to ensure tables with exact numerical evidence outrank distractor text.
- **Custom LLM-as-a-Judge Evaluation:** Includes a zero-dependency evaluation script (`evaluate_llm_judge.py`) to measure Faithfulness and Relevancy, bypassing unstable third-party libraries like RAGAS.
- **Deterministic Answer Verification & HITL:** A zero-dependency Python verifier (`verifier.py`) that normalizes numbers, extracts operands for mathematical exceptions, and cross-checks LLM claims against source chunks. It filters out page numbers and headings to prevent false positives. The UI features a `🛡️ Verify Answer & Sources` panel with `👍/👎` feedback buttons that log to a local `feedback_log.json` Golden Dataset.
- **Developer Mode Toggle:** A UI toggle that hides the verification panel and exposes internal query-rewriting steps, giving developers and researchers deep visibility into the RAG pipeline without cluttering the user experience.

## 🏗️ Architecture Pipeline
0. **Document-Isolated Semantic Cache:** Checks if a semantically similar question (Cosine Similarity > 0.95) was already answered **for the current document (`document_id`)**. If yes, returns the cached answer and chunks instantly without hitting the LLM or Vector DB.
1. **Document Ingestion & Semantic Chunking:** Extracts text (via PyMuPDF) and tables (via Camelot, with pdfplumber fallback). Filters out false-positive tables (like 2-column text) and **automatically drops empty tables (non-alphanumeric blocks)**. Text undergoes Semantic Chunking while tables remain atomic.
2. **Query Processing:** Resolves follow-up questions using chat history and generates multiple expanded queries.
3. **Hybrid Retrieval:** Fetches candidates using both Vector Search (ChromaDB) and BM25.
4. **Fusion & Smart Reranking:** Combines results using RRF and scores them with a local Cross-Encoder, applying **Dynamic Table Boosting** for numeric queries.
5. **Generation:** Streams the final answer using Claude Haiku 5.5 based strictly on the retrieved context.
6. **Deterministic Verification:** Post-generation, a rule-based Python layer audits the LLM's answer sentence-by-sentence. It normalizes numbers, ignores page/table references, and validates calculated operands. The result is fed to the HITL UI.


## Project Structure

```text
DocLens/
├── app.py                 # Streamlit UI
├── chunk_pdf.py           # Text splitting logic
├── config.py              # local model example +chunk size+ chunk overlap
├── embedding.py           # ONNX Embedding model wrapper
├── generate.py            # LLM streaming generation
├── inspect_pdf.py         # PyMuPDF + pdfplumber extraction
├── pipeline.py            # Main RAG orchestrator
├── query_rewriter.py      # Query resolution and expansion
├── reranker.py            # ONNX Cross-Encoder wrapper with Smart Boost
├── semantic_cache.py      # ChromaDB-based semantic cache
├── retriever.py           # Hybrid Search (Vector + BM25) & RRF
├── vector_store.py        # ChromaDB management
├── verifier.py            # Pure-Python deterministic answer verification (NEW)
├── requirements.txt
└── example/               # Retrieval evaluation scripts and sample PDF
└── .streamlit             # for dark mode switch in streamlit
```
The `example/` directory contains sample PDF and retrieval-evaluation resources.

## Requirements

- Python 3.14 is recommended for the current development setup.
- The Python packages listed in `requirements.txt`.
- Access to the local embedding and reranker model files.
- An API endpoint and credentials compatible with the answer-generation code.

## Installation

Clone the repository:

```bash
git clone https://github.com/SinaPars9/DocLens.git
cd DocLens
```

Install the dependencies:

```bash
pip install -r requirements.txt
```
## Download Local Models:
Ensure the ONNX models are placed in your local directory (e.g., E:\models\). Update the paths in config.py if necessary
## Configuration

DocLens uses local model files for embedding and reranking. Configure their paths to match your environment.

If answer generation requires an API key, configure it using the environment-variable names expected by the code. Keep credentials in your local `.env` file or environment, and **never commit secrets to GitHub**.
Open `config.py` to update model paths, chunking thresholds (`SEMANTIC_THRESHOLD_PERCENTILE`), and retrieval limits (`TOP_K_RETRIEVE`).

Before running the project, verify that:

- The embedding model path is correct.
- The reranker model and tokenizer paths are correct.
- The API endpoint, model name, and credentials are configured.
- The configured model and API are accessible.

## Running the Application

Start the Streamlit interface:
*Note: The system automatically creates an `llm_cache/` folder to store previous responses. If you want to clear the cache and force the system to fetch fresh answers from the API, simply delete this folder.*
```bash
streamlit run app.py
```

Open the local URL printed in the terminal.

Upload a PDF, build the vector database, ask a question, and inspect the retrieved chunks alongside the generated answer.

## Retrieval Evaluation

The `example/` directory includes:

- `create_pdf.py` — creates a sample PDF for testing.
- `evaluate_retrieval.py` — evaluates retrieval against a set of questions.
- `evaluation_questions.json` — contains the evaluation questions.
- `evaluate_llm_judge.py` — evaluates retrieval against a set of questions with llm as a judge.
- `retrieval_evaluation_report.json` — contains a sample evaluation report.
- `test.pdf` — the sample PDF used in the example workflow.

To run the evaluation script:

```bash
python example/evaluate_retrieval.py
```

Check the script for any required arguments or configuration before running it.

## Limitations

**Complex PDF Layouts (Nested Tables):** While Camelot handles standard bordered tables perfectly, deeply nested tables or highly irregular multi-column layouts may still require vision-based extraction (e.g., GPT-4o Vision) in future iterations.

**Small Talk Handling:** The query rewriter (GPT-4o-mini) sometimes over-eagerly resolves greetings or gratitude (e.g., "thank you") back to the previous query, causing a duplicate response. A system prompt fix is planned.

**Multilingual Support:** The current pipeline is optimized for English. Future iterations will focus on Persian/Arabic RTL support, requiring different embedding models (e.g., BGE-M3) and updated regex patterns for the verifier.

## 🛠️ Tech Stack

- **UI:** Streamlit
- **PDF Processing:** PyMuPDF, pdfplumber, **Camelot** (for structured table extraction)
- **Data Transformation:** Pandas, Tabulate (for Markdown table conversion)
- **Vector Store:** ChromaDB
- **Keyword Search:** rank_bm25
- **Local Inference:** ONNX Runtime, Tokenizers
- **LLMs:** Claude Haiku 5.5 (Generation), GPT-4o-mini (Query Rewriting)
- **Verification & Evaluation:** Pure-Python (Deterministic Regex), Custom LLM Judge
- **Caching & Data Logging:** Diskcache, JSON (for HITL Golden Dataset feedback)

