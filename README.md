# DocLens — Advanced PDF RAG Assistant

DocLens is an end-to-end Retrieval-Augmented Generation (RAG) system that lets users upload a PDF, retrieve relevant passages, and ask questions about its content through a conversational interface. 

This project focuses on building a **experimental / learning project** document analysis pipeline. It goes beyond basic RAG by implementing Hybrid Search, Smart Table Reranking, Streaming, and local ONNX model inference to run efficiently even on low-spec hardware.

## Features

- **Hybrid Search (BM25 + Vector):** Combines semantic vector search (ChromaDB) with keyword-based BM25 (using Reciprocal Rank Fusion) to drastically improve retrieval accuracy for numeric data, specific entities, and tables.
- **Semantic Chunking & Robust Parsing:** Text is split based on semantic topic shifts (rather than arbitrary character limits), while a custom word-count filter prevents 2-column PDF layouts from being falsely detected as tables.
- **Streaming Responses:** Built with Streamlit, generating answers token-by-token for a ChatGPT-like user experience with near-zero perceived latency.
- **Layered Semantic Caching:** Uses a dual-cache system. An exact-match disk cache for LLM calls, plus a ChromaDB-based Semantic Cache that recognizes meaningfully similar queries (e.g., "How much did we make in 2023?" vs "2023 revenue") and returns answers instantly with zero API cost.
- **Batched ONNX Inference:** Optimizes local CPU inference for both embedding (`bekko-embedding`) and reranking (`ms-marco-MiniLM`) models, preventing memory spikes.
- **Domain-Agnostic Prompts:** LLM prompts are generalized to handle financial reports, academic papers, or technical manuals with equal accuracy.
- **Transparent UI:** Displays retrieved chunks, page numbers, and rerank scores in a dedicated context panel for full explainability.
- **Transparent Query Inspector:** The Streamlit UI features a collapsible panel that shows the user exactly how their follow-up question was resolved into a standalone query before hitting the database (with cache-hit bypasses).
- **Hybrid LLM Strategy:** Uses `gpt-4o-mini` for cost-effective query rewriting, and `Claude Haiku 5.5` for final generation to ensure zero hallucination, accurate math reasoning, and strict adherence to context.
- **Enhanced Dynamic Table Boosting:** The Cross-Encoder reranker dynamically extracts numbers/keywords from the query and applies a heavy penalty/boost to ensure tables with exact numerical evidence outrank distractor text.
- **Custom LLM-as-a-Judge Evaluation:** Includes a zero-dependency evaluation script (`evaluate_llm_judge.py`) to measure Faithfulness and Relevancy, bypassing unstable third-party libraries like RAGAS.
- **Deterministic Answer Verification:** A zero-dependency, pure-Python verifier (`verifier.py`) that acts as a strict auditor. It normalizes numbers, extracts operands for mathematical exceptions (e.g., sums/differences), and mathematically proves if the LLM's answer is grounded in the source text, verified, or hallucinated.
- **Human-in-the-Loop (HITL) Interface:** The Streamlit UI features a dedicated `🛡️ Verify Answer & Sources` panel. It highlights verified claims, exposes hallucinations, and provides `👍/👎` feedback buttons. User feedback (along with the source snippet and verdict) is automatically saved to a local JSON file (`feedback_log.json`) to build a "Golden Dataset" for future fine-tuning and regression testing.
- **Developer Mode Toggle:** A UI toggle that hides the verification panel and exposes internal query-rewriting steps, giving developers and researchers deep visibility into the RAG pipeline without cluttering the user experience.

## 🏗️ Architecture Pipeline
0. **Semantic Cache:** Checks if a semantically similar question (Cosine Similarity > 0.95) was already answered. If yes, returns the cached answer and chunks instantly without hitting the LLM or Vector DB.
1. **Document Ingestion & Semantic Chunking:** Extracts text and tables (via PyMuPDF + pdfplumber). Filters out false-positive tables (like 2-column text) and applies Semantic Chunking to text blocks to preserve topic integrity, while keeping actual tables intact.
2. **Query Processing:** Resolves follow-up questions using chat history and generates multiple expanded queries.
3. **Hybrid Retrieval:** Fetches candidates using both Vector Search (ChromaDB) and BM25.
4. **Fusion & Smart Reranking:** Combines results using RRF and scores them with a local Cross-Encoder, applying **Dynamic Table Boosting** for numeric queries.
5. **Generation:** Streams the final answer using Claude Haiku 5.5 based strictly on the retrieved context.
6. **Deterministic Verification:** Post-generation, a rule-based Python layer audits the LLM's answer sentence-by-sentence. It checks for numeric hallucinations, validates calculated operands, and feeds the result to the HITL UI.


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

**Verifier Regex Constraints:** While the deterministic verifier catches numeric hallucinations and directionality errors effectively, it relies on regex. It may still struggle with highly complex semantic mismatches (e.g., a number that exists in the text but is attributed to the wrong entity in a very dense paragraph).

**Complex PDF Layouts:** Multi-column PDFs or highly irregular table structures may still pose extraction challenges. (Upcoming roadmap includes Camelot integration for structured table extraction).

**Multilingual Support:** The current pipeline is optimized for English. Future iterations will focus on Persian/Arabic RTL support, requiring different embedding models (e.g., BGE-M3) and updated regex patterns for the verifier.

## 🛠️ Tech Stack

- **UI:** Streamlit
- **PDF Processing:** PyMuPDF, pdfplumber
- **Vector Store:** ChromaDB
- **Keyword Search:** rank_bm25
- **Local Inference:** ONNX Runtime, Tokenizers
- **LLMs:** Claude Haiku 5.5 (Generation), GPT-4o-mini (Query Rewriting)
- **Caching & Evaluation:** Diskcache, Custom LLM Judge

