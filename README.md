# DocLens — Advanced PDF RAG Assistant

DocLens is an end-to-end Retrieval-Augmented Generation (RAG) system that lets users upload one or multiple PDFs, retrieve relevant passages, and ask questions about their content through a conversational interface. 

This project goes beyond basic RAG by implementing Hybrid Search, Smart Table Extraction, Deterministic Verification, Human-in-the-Loop (HITL) feedback, Multi-Document processing, and local ONNX model inference to run efficiently even on low-spec hardware.

## 🚀 Key Features

### 🔍 Retrieval & Parsing
- **Multi-Document RAG:** Upload multiple documents and query across them. Uses a unified ChromaDB with `source_document` metadata filtering to isolate documents during search.
- **Document Search Scope:** A UI feature that allows users to dynamically select which uploaded documents to target for a specific question, preventing context pollution.
- **Advanced Hybrid Parsing (Camelot + pdfplumber):** Uses PyMuPDF for text and Camelot's `lattice` parser for bordered tables (converted to clean Markdown). A custom fallback and a **Ghost Table Filter** (regex-based) drop empty blocks and 2-column text falsely detected as tables.
- **Contextual Retrieval:** Generates a unique summary for every chunk (using GPT-4o-mini or fast page-headers) and prepends it to the chunk text before embedding. This drastically improves retrieval accuracy for context-dependent chunks.
- **Parent-Child Architecture:** Retrieves using small, precise **Child** chunks. However, it deduplicates and sends the larger, full-page **Parent** chunks to the LLM. This provides complete context and reduces token usage by up to 60% while maintaining 100% retrieval accuracy.
- **Hybrid Search (BM25 + Vector):** Combines semantic vector search (ChromaDB) with keyword-based BM25 using Reciprocal Rank Fusion (RRF) to drastically improve retrieval accuracy.
- **Semantic Chunking:** Text is split based on semantic topic shifts, while tables remain atomic to preserve their structure for the LLM.
- **Document & Scope-Isolated Layered Caching:** A dual-cache system (exact-match disk cache + ChromaDB semantic cache). Both caches are strictly isolated by a **content hash** of the PDF and the selected search scope. The cache is automatically invalidated if the same document name is rebuilt with new content.

### 🧠 Generation & Verification
- **Intelligent Meta-Query Handling:** Uses strict regex to detect summarization requests (e.g., "what is this file about") and fetches initial pages directly, bypassing the reranker for generic overviews.
- **Balanced Multi-Document Retrieval:** Applies per-document quotas and deduplication, guaranteeing that each selected document contributes at least one chunk to the final context, while filtering out low-scoring noise.
- **Streaming Responses:** Built with Streamlit, generating answers token-by-token for a ChatGPT-like experience.
- **Hybrid LLM Strategy:** Uses `gpt-4o-mini` for cost-effective query rewriting, and `Claude Haiku 5.5` for final generation to ensure accurate math reasoning and strict adherence to context.
- **Deterministic Answer Verification:** A zero-dependency, pure-Python verifier (`verifier.py`) that acts as a strict auditor. It normalizes numbers, extracts operands for mathematical exceptions, and mathematically proves if the LLM's answer is grounded, calculated, or hallucinated. It automatically ignores page numbers and headings to prevent false positives.
- **Intelligent Abstention Layer:** To save API costs and prevent hallucinations, the pipeline checks the absolute score of the top-ranked chunk. If the highest-scoring chunk falls below a strict irrelevance threshold, the system immediately abstains from calling the LLM.

### 🛡️ UI & Human-in-the-Loop (HITL)
- **Advanced Deterministic Verifier:** A pure-Python layer that audits LLM responses sentence-by-sentence. It classifies claims into `✅ Verified`, `✅ Calculated`, `🧮 Partially Supported` (some numbers found), `⚠️ Contradicted` (directionality error), or `❌ Hallucination`.
- **Feedback Logging:** If the user flags a claim with a 👎, the system logs the query, the source snippet, and the user's reason into a local `feedback_log.json` file, building a "Golden Dataset" for future fine-tuning.
- **Developer Mode Toggle:** Hides the verification panel and exposes internal query-rewriting steps (Query Inspector) for researchers and developers.
- **Smart UI Controls:** Includes a sidebar toggle button fix, instant toast notifications for scope changes, and a confirmation dialog to prevent accidental data loss when changing files.

### ⚡ Caching & Performance
- **Document & Scope-Isolated Layered Caching:** A dual-cache system (exact-match disk cache + ChromaDB semantic cache). Both caches are strictly isolated by a unique `document_id` hash and the selected search scope to prevent cross-document cache pollution.
- **Batched ONNX Inference:** Optimizes local CPU inference for both embedding (`bekko-embedding`) and reranking (`ms-marco-MiniLM`) models, preventing memory spikes.

---

## 🏗️ Architecture Pipeline

0. **Document-Isolated Semantic Cache:** Checks if a semantically similar question was already answered for the current document and selected scope.
1. **Document Ingestion & Hybrid Extraction:** Extracts text (PyMuPDF) and tables (Camelot). Applies Semantic Chunking to create **Child** chunks, generates context summaries, and maps them to their **Parent** pages.
2. **Query Processing & Meta-Detection:** Resolves follow-ups and applies strict regex for meta-queries (summaries). For content queries, it filters the retrieval space based on the user's `Search Scope`.
3. **Balanced Hybrid Retrieval:** Fetches candidates using Vector Search and BM25, filtered by `source_document`. For multi-document queries, applies per-document quotas to ensure fair representation.
4. **Fusion & Smart Reranking:** Combines results using RRF and scores them with a local Cross-Encoder. Guarantees the top chunk from each selected document survives the compression filter.
5. **Generation (Parent-Chunking):** Deduplicates the top-ranked child chunks by their `parent_id` and sends only the full Parent pages to Claude Haiku 5.5.
6. **Deterministic Verification:** Post-generation, a rule-based Python layer audits the LLM's answer sentence-by-sentence.

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

**Metrics Evaluated (LLM-as-a-Judge):**
The suite uses `gpt-4o-mini` to judge the final answer against the `expected_answer` (Answer Accuracy), alongside automated checks for:
- **Retrieval Hit Rate:** Are the correct pages retrieved?
- **Verifier Pass Rate:** Are all non-abstention claims verified?
- **Category Breakdown:** Metrics are split by question type (numerical, table_extraction, abstention, etc.).
Results are saved to `eval_report.json` for version comparison.

To run the evaluation suite:
```bash
python eval_suite/run_eval.py
```
*Note: Set SKIP_BUILD = True in run_eval.py to reuse existing databases and speed up testing.

## Project Structure

```text
DocLens/
├── app.py                 # Streamlit UI (HITL, Dev Mode, Multi-Doc)
├── chunk_pdf.py           # Semantic chunking logic
├── config.py              # Model paths & chunking configs
├── embedding.py           # ONNX Embedding model wrapper
├── contextualizer.py      # Generates context summaries for chunks (LLM + Diskcache)
├── generate.py            # LLM streaming generation & disk cache
├── inspect_pdf.py         # PyMuPDF + Camelot extraction + Ghost Table filter
├── pipeline.py            # Main RAG orchestrator (Multi-doc logic & Meta-query)
├── query_rewriter.py      # Query resolution and expansion
├── reranker.py            # ONNX Cross-Encoder wrapper with Smart Boost
├── retriever.py           # Hybrid Search (Vector + BM25) & RRF
├── semantic_cache.py      # ChromaDB-based semantic cache (document-isolated)
├── vector_store.py        # ChromaDB management (Multi-doc support)
├── verifier.py            # Pure-Python deterministic answer verification
├── requirements.txt
├── example/               # Demo scripts and sample PDF
├── eval_suite/            # Regression testing (data, questions, run_eval.py)
├── eval_judge_cache/      # Diskcache for LLM-as-a-Judge (ignored by git)
├── eval_report.json       # Latest evaluation results (ignored by git)
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

Upload one or two PDFs, build the vector database, ask a question, and inspect the retrieved chunks alongside the generated answer.



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

