# DocLens — PDF RAG Assistant

DocLens is an educational Retrieval-Augmented Generation (RAG) project that lets users upload a PDF, retrieve relevant passages, and ask questions about its content through a conversational interface.

The project focuses on exploring the core components of a document-based RAG pipeline, including PDF text and table extraction, chunking, embeddings, vector search, query rewriting, reranking, and answer generation.

## Features

- **PDF ingestion:** Extracts text and tables from PDF documents.
- **Configurable chunking:** Splits extracted content into chunks for retrieval.
- **Local embeddings:** Uses the `hotchpotch/bekko-embedding-v1-a8m` embedding model.
- **Vector storage:** Stores and retrieves document chunks using ChromaDB.
- **Query rewriting:** Resolves follow-up questions and expands queries to improve retrieval.
- **Reranking:** Reorders retrieved passages using a local ONNX reranker.
- **Context-aware answers:** Generates answers using retrieved document context.
- **Interactive interface:** Provides a Gradio UI for uploading PDFs, asking questions, and inspecting retrieved chunks.
- **Retrieval evaluation:** Includes example scripts and evaluation data for testing retrieval quality.

## Architecture

The pipeline follows these main steps:

1. **PDF loading:** Extract text and tables from the uploaded PDF.
2. **Chunking:** Split document content into manageable chunks while preserving useful context.
3. **Embedding:** Convert chunks into vector representations using a local embedding model.
4. **Vector storage:** Store embeddings and metadata in ChromaDB.
5. **Query processing:** Rewrite or expand the user's question when needed.
6. **Retrieval:** Search the vector database for relevant chunks.
7. **Reranking:** Reorder retrieved chunks according to relevance.
8. **Answer generation:** Generate a response using the retrieved context.
9. **Inspection:** Display the retrieved chunks in the interface for easier debugging and evaluation.

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

DocLens is an educational project and is not intended as a production-ready document-analysis system.

- Retrieval quality depends on PDF extraction, chunk boundaries, embeddings, and reranking.
- Complex tables and unusual PDF layouts may not be extracted perfectly.
- Generated answers can be incorrect or unsupported by the source document.
- Model initialization and inference can be slow on CPU-only or low-memory systems.
- Model paths and API configuration may require manual setup.
- The included evaluation resources are intended for experimentation, not as a comprehensive benchmark.

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
