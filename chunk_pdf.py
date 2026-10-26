import re
from typing import Optional, List
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from inspect_pdf import load_pdf_documents
from embedding import Embeddings
from config import (
    SEMANTIC_BUFFER_SIZE, 
    SEMANTIC_THRESHOLD_PERCENTILE, 
    FALLBACK_CHUNK_SIZE, 
    FALLBACK_CHUNK_OVERLAP
)

fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=FALLBACK_CHUNK_SIZE, 
    chunk_overlap=FALLBACK_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def semantic_chunking(text: str, embed_model: Embeddings) -> List[str]:
    sentences = _split_sentences(text)
    if len(sentences) <= SEMANTIC_BUFFER_SIZE:
        return [text]

    windows = []
    for i in range(0, len(sentences), SEMANTIC_BUFFER_SIZE):
        windows.append(" ".join(sentences[i:i + SEMANTIC_BUFFER_SIZE]))

    embeddings = embed_model.embed_texts(windows)

    distances = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i+1])
        distances.append(1 - sim)

    if not distances:
        return [text]
    
    threshold = np.percentile(distances, SEMANTIC_THRESHOLD_PERCENTILE)
    breakpoints = [i for i, d in enumerate(distances) if d > threshold]

    chunks = []
    start_idx = 0
    for bp in breakpoints:
        end_idx = (bp + 1) * SEMANTIC_BUFFER_SIZE
        chunk_text = " ".join(sentences[start_idx:end_idx]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        start_idx = end_idx
    
    chunk_text = " ".join(sentences[start_idx:]).strip()
    if chunk_text:
        chunks.append(chunk_text)

    final_chunks = []
    for c in chunks:
        if len(c) > FALLBACK_CHUNK_SIZE:
            final_chunks.extend(fallback_splitter.split_text(c))
        else:
            final_chunks.append(c)

    return final_chunks

def load_chunks(pdf_path: str, embed_model: Optional[Embeddings] = None) -> List[Document]:
    documents = load_pdf_documents(pdf_path)
    
    if embed_model is None:
        embed_model = Embeddings()

    all_chunks = []
    
    for doc in documents:
        if doc.metadata.get("type") == "table":
            all_chunks.append(doc)
        else:
            semantic_texts = semantic_chunking(doc.page_content, embed_model)
            for text in semantic_texts:
                if len(text) > 50:
                    new_doc = Document(
                        page_content=text,
                        metadata=doc.metadata.copy()
                    )
                    all_chunks.append(new_doc)

    return all_chunks