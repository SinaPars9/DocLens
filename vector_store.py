from pathlib import Path
import time
import os
from typing import List

import chromadb

from chunk_pdf import load_chunks
from embedding import Embeddings

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "vector_db"


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        self.embedding_model = Embeddings()

    def build(self, pdf_paths: List[str], use_llm_context: bool = True):

        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        all_texts = []
        all_metadatas = []
        all_ids = []

        start = time.perf_counter()

        for pdf_path in pdf_paths:
            doc_name = os.path.basename(pdf_path)

            if doc_name.startswith("temp_"):
                doc_name = doc_name[5:]

            print(f"Processing {doc_name}...")

            chunks = load_chunks(
                pdf_path, self.embedding_model, use_llm_context=use_llm_context
            )

            if not chunks:
                continue

            for i, chunk in enumerate(chunks):
                all_texts.append(chunk.page_content)

                metadata = dict(chunk.metadata)
                metadata["source_document"] = doc_name
                metadata["chunk_index"] = len(all_texts) - 1

                clean_metadata = {
                    key: value
                    for key, value in metadata.items()
                    if isinstance(value, (str, int, float, bool))
                }
                all_metadatas.append(clean_metadata)
                all_ids.append(f"chunk_{len(all_texts) - 1}")

        if not all_texts:
            raise ValueError("No chunks found in the provided PDFs.")

        print(f"Embedding {len(all_texts)} chunks from {len(pdf_paths)} documents...")
        vectors = self.embedding_model.embed_texts(all_texts)

        batch_size = 100
        for start_idx in range(0, len(all_texts), batch_size):
            end_idx = start_idx + batch_size

            self.collection.add(
                ids=all_ids[start_idx:end_idx],
                documents=all_texts[start_idx:end_idx],
                embeddings=vectors[start_idx:end_idx].tolist(),
                metadatas=all_metadatas[start_idx:end_idx],
            )
            print(f"Stored {min(end_idx, len(all_texts))}/{len(all_texts)} chunks")

        elapsed = time.perf_counter() - start
        print(f"Vector store built in {elapsed:.2f} seconds")

    def get_collection(self):
        return self.collection
