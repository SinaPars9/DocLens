
from pathlib import Path
import time

import chromadb

from chunk_pdf import load_chunks
from embedding import Embeddings


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "vector_db"


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        self.embedding_model = Embeddings()

    def build(self,pdf_path):
        chunks = load_chunks(pdf_path)

        if not chunks:
            raise ValueError("No chunks found in the source PDF.")

        
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        start = time.perf_counter()
        print(f"Building vector store from {len(chunks)} chunks...")

        texts = [chunk.page_content for chunk in chunks]

        vectors = self.embedding_model.embed_texts(texts)

        ids = [f"chunk_{i}" for i in range(len(chunks))]

        metadatas = []
        for i, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata)
            metadata["chunk_index"] = i

            
            clean_metadata = {
                key: value
                for key, value in metadata.items()
                if isinstance(value, (str, int, float, bool))
            }
            metadatas.append(clean_metadata)

        batch_size = 100

        for start_idx in range(0, len(chunks), batch_size):
            end_idx = start_idx + batch_size

            self.collection.add(
                ids=ids[start_idx:end_idx],
                documents=texts[start_idx:end_idx],
                embeddings=vectors[start_idx:end_idx].tolist(),
                metadatas=metadatas[start_idx:end_idx],
            )

            print(f"Stored {min(end_idx, len(chunks))}/{len(chunks)} chunks")

        elapsed = time.perf_counter() - start
        print(f"Vector store built in {elapsed:.2f} seconds")
        print(f"Collection count: {self.collection.count()}")

    def get_collection(self):
        return self.collection


