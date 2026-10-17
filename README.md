# DocLens — PDF RAG Assistant

DocLens is an educational Retrieval-Augmented Generation (RAG) project that allows users to ask questions about PDF documents and receive answers grounded in the document's content.

The project focuses on exploring document processing, embedding models, vector search, query rewriting, reranking, and LLM-based answer generation in a practical application.

## Features

- **PDF Processing:** Extracts text and detects tables from PDF documents.
- **Table-Aware Extraction:** Converts detected tables into structured text to help preserve relationships between labels and values.
- **Document Chunking:** Splits extracted content into smaller pieces for retrieval.
- **Local Embeddings:** Uses the Bekko embedding model through ONNX Runtime.
- **Vector Database:** Stores and retrieves document chunks using ChromaDB.
- **Query Rewriting:** Resolves follow-up questions and expands queries to improve retrieval.
- **Reranking:** Reranks retrieved chunks using an ONNX-based reranker.
- **Context-Aware Answers:** Generates answers using retrieved document content and conversation history.
- **Gradio Interface:** Provides a web-based interface for uploading PDFs, asking questions, and inspecting retrieved context.

## Architecture

The application follows a modular pipeline:

1. Extract text and tables from a PDF.
2. Split the extracted content into chunks.
3. Generate embeddings for the chunks.
4. Store the embeddings and associated metadata in ChromaDB.
5. Process the user's question and rewrite or expand it when necessary.
6. Retrieve relevant chunks from the vector database.
7. Rerank the retrieved chunks.
8. Generate an answer using the selected context.
9. Display the answer and retrieved chunks in the Gradio interface.

## Project Structure

```text
DocLens/
├── app.py              # Gradio user interface
├── embedding.py        # Embedding generation
├── chunk_pdf.py        # PDF chunking utilities
├── vector_store.py     # ChromaDB operations
├── retriever.py        # Document retrieval
├── query_rewriter.py   # Query rewriting and expansion
├── reranker.py         # Result reranking
├── generate.py         # LLM answer generation
├── pipeline.py         # RAG pipeline orchestration
├── inspect_pdf.py      # PDF extraction and inspection
├── example/
│   ├── create_pdf.py
│   ├── evaluate_retrieval.py
│   └── evaluation_questions.json
├── .env.example        # Environment variable template
├── .gitignore
└── README.md
```
## Requirements

- Python
- A compatible LLM API endpoint
- An API key for the configured provider
- The embedding and reranker model files
- The Python dependencies required by the project

The embedding and reranking components use ONNX Runtime, while ChromaDB handles vector storage and similarity search.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SinaPars9/DocLens.git
cd DocLens
```

Replace `YOUR_USERNAME/DocLens` with the actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the dependencies used by the project:

```bash
pip install -r requirements.txt
```

A `requirements.txt` file should be maintained with the project's actual dependencies.

### 4. Configure environment variables

Create a .env file in the project root and add the API configuration required by your LLM provider, including the API endpoint, model name, and API key.

**Never commit your `.env` file or API keys to version control.**

### 5. Configure model paths

Download the embedding and reranker models separately and configure their local paths as required by the project.

Model files are not included in this repository.

## Usage

Run the application:

```bash
python app.py
```

Then:

1. Upload a PDF document.
2. Click **Build Database** and wait for processing to finish.
3. Ask questions about the document.
4. Review the generated answer.
5. Inspect the retrieved chunks displayed alongside the conversation.

The answers depend on the quality of the PDF extraction, retrieval results, reranking, and the configured language model.

## Evaluation

The `example/` directory contains utilities for generating a sample PDF and evaluating retrieval performance.

The evaluation workflow can be used to investigate how well the retriever finds relevant document chunks for a set of predefined questions.

Retrieval quality and answer quality are separate concerns: retrieving a relevant chunk does not guarantee that the generated answer will be correct.

## Limitations

- Designed as a learning project, not a production-ready document intelligence system.
- Extraction quality depends on PDF structure and table layout.
- Scanned PDFs may require OCR, which is not currently guaranteed.
- Retrieval and reranking can miss relevant information.
- Language-model responses may still contain errors.
- Processing time depends on the hardware, model sizes, and configured API provider.
- Local model files and API access must be configured separately.

## Technologies

- Python
- PyMuPDF
- pdfplumber
- LangChain document abstractions
- ChromaDB
- ONNX Runtime
- Hugging Face / ONNX tokenization tools
- Gradio
- LLM API integration

## Project Status

DocLens is an ongoing educational project exploring practical RAG system design, retrieval evaluation, and document question answering.

## License

No license has been specified yet. Add a license before explicitly permitting others to reuse, modify, or distribute the project.
