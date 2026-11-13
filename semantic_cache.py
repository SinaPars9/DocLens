import chromadb
from chromadb.config import Settings
from pathlib import Path
import json
import uuid  
import time 
from typing import List, Dict, Any, Optional, Tuple


class SemanticCache:
    def __init__(
        self, embedding_model, persist_dir: str = "chroma_db", threshold: float = 0.95
    ):
        self.embedding_model = embedding_model
        self.threshold = threshold

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="semantic_cache", metadata={"hnsw:space": "cosine"}
        )

    def get(
        self, query: str, document_id: str
    ) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        if self.collection.count() == 0:
            return None

        query_vector = self.embedding_model.embed_texts([query])[0]

        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=1,
            where={"document_id": document_id},
            include=["distances", "metadatas"],
        )

        if not results or not results.get("distances") or not results["distances"][0]: #type:ignore
            return None

        distance = results["distances"][0][0] #type:ignore
        distance = max(0.0, distance)
        
        if distance <= (1 - self.threshold):
            print(f"[Semantic Cache Hit] Distance: {distance:.4f}")
            metadata = results["metadatas"][0][0] #type:ignore
            answer = metadata["answer"]
            chunks = json.loads(metadata["chunks"]) #type:ignore
            return answer, chunks #type:ignore

        return None

    def put(
        self, query: str, answer: str, chunks: List[Dict[str, Any]], document_id: str
    ):
        query_vector = self.embedding_model.embed_texts([query])[0]
        
        
        doc_id = f"cache_{uuid.uuid4().hex}"
        
        chunks_json = json.dumps(chunks, ensure_ascii=False)

        self.collection.add(
            ids=[doc_id],
            embeddings=[query_vector.tolist()],
            documents=[query],
            metadatas=[
                {
                    "answer": answer, 
                    "chunks": chunks_json, 
                    "document_id": document_id,
                    "timestamp": time.time() 
                }
            ],
        )
        print(f"[Semantic Cache Store] Saved: {query[:30]}...")

    
    def invalidate_document(self, document_id: str):
        """Remove all cache entries associated with a specific document."""
        try:
            self.collection.delete(where={"document_id": document_id})
            print(f"[Semantic Cache] Invalidated cache for doc: {document_id}")
        except Exception as e:
            print(f"[Semantic Cache] Failed to invalidate: {e}")

    
    def clear(self):
        try:
            self.client.delete_collection("semantic_cache")
            self.collection = self.client.create_collection(
                name="semantic_cache", metadata={"hnsw:space": "cosine"}
            )
            print("[Semantic Cache] Cleared all entries.")
        except Exception as e:
            print(f"[Semantic Cache] Failed to clear: {e}")