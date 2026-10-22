# DocLens — PDF RAG Assistant

DocLens is an educational Retrieval-Augmented Generation (RAG) project that lets users upload a PDF, retrieve relevant passages, and ask questions about its content through a conversational interface.

The project focuses on exploring the core components of a document-based RAG pipeline, including PDF text and table extraction, chunking, embeddings, vector search, query rewriting, reranking, and answer generation.

## Features

- **Hybrid Search (BM25 + Vector):** Combines semantic vector search with keyword-based BM25 (using Reciprocal Rank Fusion) to drastically improve retrieval accuracy for numeric data, specific entities, and tables.
- **Streaming Responses:** Generates answers token-by-token, providing a ChatGPT-like user experience with near-zero perceived latency.
- **Persistent LLM Caching:** Utilizes `diskcache` to locally store query rewrites and generated answers, saving API costs and delivering instant responses for repeated questions.
- **Batched Processing:** Optimizes local ONNX model inference by processing embeddings and reranking in batches, preventing memory spikes on low-spec hardware.
- **Transparent UI:** Built with Gradio, displaying retrieved chunks, page numbers, and rerank scores in a dedicated context panel for full explainability.
- **Local Embedding & Reranking:** Uses ONNX Runtime for fast, local CPU inference of the `bekko-embedding` and `ms-marco-MiniLM` models.

## Architecture

1. **Query Processing:** Resolves follow-up questions using chat history and generates multiple expanded queries.
2. **Hybrid Retrieval:** Fetches candidates using both Vector Search (ChromaDB) and BM25.
3. **Fusion:** Combines results using Reciprocal Rank Fusion (RRF).
4. **Reranking:** Scores the fused candidates using a local Cross-Encoder ONNX model.
5. **Generation:** Streams the final answer using an LLM based strictly on the retrieved cont

## Project Structure

```text
DocLens/
├── app.py
├── chunk_pdf.py
├── embedding.py
├── generate.py
├── inspect_pdf.py
├── pipeline.py
├── query_rewriter.py
├── reranker.py
├── retriever.py
├── vector_store.py
├── requirements.txt
├── README.md
├── .gitignore
├── example/
│   ├── create_pdf.py
│   ├── evaluate_retrieval.py
│   ├── evaluation_questions.json
│   ├── retrieval_evaluation_report.json
│   └── test.pdf
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

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
py -3.14 -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

DocLens uses local model files for embedding and reranking. Configure their paths to match your environment.

If answer generation requires an API key, configure it using the environment-variable names expected by the code. Keep credentials in your local `.env` file or environment, and **never commit secrets to GitHub**.

Before running the project, verify that:

- The embedding model path is correct.
- The reranker model and tokenizer paths are correct.
- The API endpoint, model name, and credentials are configured.
- The configured model and API are accessible.

## Running the Application

Start the Gradio interface:
*Note: The system automatically creates an `llm_cache/` folder to store previous responses. If you want to clear the cache and force the system to fetch fresh answers from the API, simply delete this folder.*
```bash
python app.py
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


Complex PDF Layouts: Multi-column PDFs or those with highly irregular table structures may still pose extraction challenges.

Multilingual Support: The current pipeline is optimized for English. Future iterations will focus on Persian/Arabic RTL support, requiring different embedding models (e.g., BGE-M3) and text normalization.
## Tech Stack

- Python
- Gradio
- PyMuPDF
- pdfplumber
- LangChain text utilities
- ChromaDB
- ONNX Runtime
- Tokenizers
- HTTPX

## Project Status

DocLens is an ongoing learning project focused on understanding and improving document-based RAG systems through practical experimentation.

## License

No license has been added yet. Unless a license is added to the repository, others should not assume they have permission to redistribute or reuse the project.
