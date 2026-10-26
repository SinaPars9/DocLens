# DocLens — Advanced PDF RAG Assistant

DocLens is an end-to-end Retrieval-Augmented Generation (RAG) system that lets users upload a PDF, retrieve relevant passages, and ask questions about its content through a conversational interface. 

This project focuses on building a **experimental / learning project** document analysis pipeline. It goes beyond basic RAG by implementing Hybrid Search, Smart Table Reranking, Streaming, and local ONNX model inference to run efficiently even on low-spec hardware.

## Features

- **Hybrid Search (BM25 + Vector):** Combines semantic vector search (ChromaDB) with keyword-based BM25 (using Reciprocal Rank Fusion) to drastically improve retrieval accuracy for numeric data, specific entities, and tables.
- **Smart Table Boost:** The Cross-Encoder reranker is enhanced to automatically boost the scores of table chunks when the user's question contains numbers or financial symbols, ensuring tabular data is never ignored.
- **Streaming Responses:** Built with Streamlit, generating answers token-by-token for a ChatGPT-like user experience with near-zero perceived latency.
- **Persistent LLM Caching:** Utilizes `diskcache` to locally store query rewrites and generated answers, saving API costs and delivering instant responses for repeated questions.
- **Batched ONNX Inference:** Optimizes local CPU inference for both embedding (`bekko-embedding`) and reranking (`ms-marco-MiniLM`) models, preventing memory spikes.
- **Domain-Agnostic Prompts:** LLM prompts are generalized to handle financial reports, academic papers, or technical manuals with equal accuracy.
- **Transparent UI:** Displays retrieved chunks, page numbers, and rerank scores in a dedicated context panel for full explainability.

## Architecture Pipeline

1. **Query Processing:** Resolves follow-up questions using chat history (e.g., resolving pronouns and missing years) and generates multiple expanded queries.
2. **Hybrid Retrieval:** Fetches candidates using both Vector Search and BM25.
3. **Fusion:** Combines results using Reciprocal Rank Fusion (RRF).
4. **Smart Reranking:** Scores the fused candidates using a local Cross-Encoder ONNX model, with conditional boosting for table data.
5. **Generation:** Streams the final answer using an LLM based strictly on the retrieved context.

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
- `retrieval_evaluation_report.json` — contains a sample evaluation report.
- `test.pdf` — the sample PDF used in the example workflow.

To run the evaluation script:

```bash
python example/evaluate_retrieval.py
```

Check the script for any required arguments or configuration before running it.

## Limitations

Multi-hop Reasoning on Follow-ups: While the retrieval system correctly identifies the relevant tables, smaller LLMs (like gpt-4o-mini) may sometimes take "shortcuts" during follow-up questions (e.g., returning 2025 data instead of calculating 2024 data). Future iterations will explore stronger LLMs or fine-tuning for better multi-hop reasoning.

Complex PDF Layouts: Multi-column PDFs or highly irregular table structures may still pose extraction challenges.

Multilingual Support: The current pipeline is optimized for English. Future iterations will focus on Persian/Arabic RTL support, requiring different embedding models (e.g., BGE-M3).
## Tech Stack

- Python
- streamlit
- PDF Processing: PyMuPDF, pdfplumber
- Keyword Search: rank_bm25
- LangChain text utilities
- Vector Store: ChromaDB
- Local Inference: ONNX Runtime, Tokenizers
- LLM & Caching: OpenAI SDK, diskcache

