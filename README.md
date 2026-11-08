# DocLens — Advanced PDF RAG Assistant

DocLens is an end-to-end Retrieval-Augmented Generation (RAG) system that lets users upload a PDF, retrieve relevant passages, and ask questions about its content through a conversational interface. 

This project goes beyond basic RAG by implementing Hybrid Search, Smart Table Extraction, Deterministic Verification, Human-in-the-Loop (HITL) feedback, and local ONNX model inference to run efficiently even on low-spec hardware.

## 🚀 Key Features

### 🔍 Retrieval & Parsing
- **Advanced Hybrid Parsing (Camelot + pdfplumber):** Uses PyMuPDF for text and Camelot's `lattice` parser for bordered tables (converted to clean Markdown). A custom fallback and a **Ghost Table Filter** (regex-based) drop empty blocks and 2-column text falsely detected as tables.
- **Hybrid Search (BM25 + Vector):** Combines semantic vector search (ChromaDB) with keyword-based BM25 using Reciprocal Rank Fusion (RRF) to drastically improve retrieval accuracy for numeric data, specific entities, and tables.
- **Semantic Chunking:** Text is split based on semantic topic shifts, while tables remain atomic to preserve their structure for the LLM.

### 🧠 Generation & Verification
- **Streaming Responses:** Built with Streamlit, generating answers token-by-token for a ChatGPT-like experience.
- **Hybrid LLM Strategy:** Uses `gpt-4o-mini` for cost-effective query rewriting, and `Claude Haiku 5.5` for final generation to ensure accurate math reasoning and strict adherence to context.
- **Deterministic Answer Verification:** A zero-dependency, pure-Python verifier (`verifier.py`) that acts as a strict auditor. It normalizes numbers, extracts operands for mathematical exceptions (e.g., sums/differences), and mathematically proves if the LLM's answer is grounded, calculated, or hallucinated. It automatically ignores page numbers and headings to prevent false positives.

### 🛡️ UI & Human-in-the-Loop (HITL)
- **Verify Answer & Sources UI:** Highlights exactly which sentence is ✅ Verified, ❌ Hallucinated, or 🧮 Calculated. It displays the exact source table/text from the PDF right below the claim.
- **Feedback Logging:** If the user flags a claim with a 👎, the system logs the query, the source snippet, and the user's reason into a local `feedback_log.json` file, building a "Golden Dataset" for future fine-tuning.
- **Developer Mode Toggle:** Hides the verification panel and exposes internal query-rewriting steps (Query Inspector) for researchers and developers.

### ⚡ Caching & Performance
- **Document-Isolated Layered Caching:** A dual-cache system (exact-match disk cache + ChromaDB semantic cache). Both caches are strictly isolated by a unique `document_id` hash to prevent cross-document cache pollution when switching between files.
- **Batched ONNX Inference:** Optimizes local CPU inference for both embedding (`bekko-embedding`) and reranking (`ms-marco-MiniLM`) models, preventing memory spikes.

---

## 🏗️ Architecture Pipeline

0. **Document-Isolated Semantic Cache:** Checks if a semantically similar question (Cosine Similarity > 0.95) was already answered **for the current document**. If yes, returns the cached answer instantly.
1. **Document Ingestion & Hybrid Extraction:** Extracts text (PyMuPDF) and tables (Camelot, with pdfplumber fallback). Filters out false-positive and empty tables. Text undergoes Semantic Chunking while tables remain atomic.
2. **Query Processing:** Resolves follow-up questions using chat history and generates multiple expanded queries.
3. **Hybrid Retrieval:** Fetches candidates using both Vector Search (ChromaDB) and BM25.
4. **Fusion & Smart Reranking:** Combines results using RRF and scores them with a local Cross-Encoder, applying **Dynamic Table Boosting** for numeric queries.
5. **Generation:** Streams the final answer using Claude Haiku 5.5 based strictly on the retrieved context (with injected Page/Type metadata).
6. **Deterministic Verification:** Post-generation, a rule-based Python layer audits the LLM's answer sentence-by-sentence, checking for hallucinations and validating calculated operands.

---

## 🧪 Evaluation Suite

DocLens includes two levels of evaluation to ensure regression testing and continuous quality improvement.

### 1. Quick Demo (`example/`)
Contains a sample financial PDF and basic evaluation scripts to demonstrate the workflow.

### 2. Robust Regression Suite (`eval_suite/`)
Includes a diverse set of PDF documents to stress-test the pipeline across different domains:
- `northstar_financial.pdf`: Dense financial tables and calculations.
- `transformers_paper.pdf`: Academic paper with complex table layouts.
- `safety_science.pdf`: Long definition tables and conceptual diagrams.

The suite runs curated questions (`questions.json`) covering:
- **Numerical Calculations:** Testing the LLM's math and the Verifier's operand checks.
- **Table Extraction:** Testing Camelot's ability to extract clean Markdown tables.
- **Abstention:** Testing the system's ability to say "I don't know" when information is missing.

To run the evaluation suite:
```bash
python eval_suite/run_eval.py

Note: Set SKIP_BUILD = True in run_eval.py to reuse existing databases and speed up testing.

Check the script for any required arguments or configuration before running it.

## Project Structure

```text
DocLens/
├── app.py                 # Streamlit UI (HITL, Dev Mode)
├── chunk_pdf.py           # Semantic chunking logic
├── config.py              # Model paths & chunking configs
├── embedding.py           # ONNX Embedding model wrapper
├── generate.py            # LLM streaming generation & disk cache
├── inspect_pdf.py         # PyMuPDF + Camelot extraction + Ghost Table filter
├── pipeline.py            # Main RAG orchestrator & document_id assignment
├── query_rewriter.py      # Query resolution and expansion
├── reranker.py            # ONNX Cross-Encoder wrapper with Smart Boost
├── retriever.py           # Hybrid Search (Vector + BM25) & RRF
├── semantic_cache.py      # ChromaDB-based semantic cache (document-isolated)
├── vector_store.py        # ChromaDB management
├── verifier.py            # Pure-Python deterministic answer verification
├── requirements.txt
├── example/               # Demo scripts and sample PDF
├── eval_suite/            # Regression testing suite (data, questions, runner)
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

