import re
from typing import List, Dict, Any, Optional
from vector_store import VectorStore
from rank_bm25 import BM25Okapi


class Retriever:
    def __init__(self, store: Optional[VectorStore] = None) -> None:
        self.store = store or VectorStore()
        self.collection = self.store.get_collection()

        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus: List[Dict[str, Any]] = []

        self._build_bm25_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9£$,.]+", text.lower())

    def _build_bm25_index(self) -> None:
        self.bm25 = None
        self.bm25_corpus = []
        count = self.collection.count()
        if count == 0:
            return

        all_data = self.collection.get(include=["documents", "metadatas"])
        if not all_data:
            return

        ids = all_data.get("ids") or []
        documents = all_data.get("documents") or []
        metadatas = all_data.get("metadatas") or []

        if not ids or not documents:
            return

        tokenized_corpus = []
        for chunk_id, text, meta in zip(ids, documents, metadatas):
            if text is None:
                continue
            safe_meta = meta if meta else {}
            self.bm25_corpus.append(
                {"id": chunk_id, "text": text, "metadata": safe_meta}
            )
            tokenized_corpus.append(self._tokenize(text))

        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"[BM25] Index built with {len(tokenized_corpus)} chunks.")

    def _vector_search(
        self, query: str, top_k: int, selected_documents: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        query_vector = self.store.embedding_model.embed_texts([query])[0]

        where_clause = None
        if selected_documents:
            where_clause = {"source_document": {"$in": selected_documents}}

        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, self.collection.count()),
            where=where_clause,  # type: ignore
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            for i in range(len(ids)):
                chunks.append(
                    {
                        "id": ids[i],
                        "text": documents[i] if i < len(documents) else "",
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "distance": distances[i] if i < len(distances) else 0.0,
                    }
                )
        return chunks

    def _bm25_search(
        self, query: str, top_k: int, selected_documents: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.bm25_corpus:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        chunks = []
        for idx in top_indices:
            chunk = dict(self.bm25_corpus[idx])

            if selected_documents:
                source_doc = chunk.get("metadata", {}).get("source_document")
                if source_doc not in selected_documents:
                    continue

            chunk["bm25_score"] = float(scores[idx])
            chunks.append(chunk)
            if len(chunks) >= top_k:
                break

        return chunks

    def _reciprocal_rank_fusion(
        self, results_lists: List[List[Dict[str, Any]]], k: int = 60
    ) -> List[Dict[str, Any]]:
        fused_scores = {}
        chunk_data = {}

        for results in results_lists:
            for rank, chunk in enumerate(results):
                chunk_id = chunk["id"]
                if chunk_id not in fused_scores:
                    fused_scores[chunk_id] = 0.0
                    chunk_data[chunk_id] = chunk
                fused_scores[chunk_id] += 1.0 / (k + rank + 1)

        sorted_ids = sorted(
            fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True
        )

        final_results = []
        for chunk_id in sorted_ids:
            item = dict(chunk_data[chunk_id])
            item["rrf_score"] = fused_scores[chunk_id]
            final_results.append(item)
        return final_results

    def retrieve_many(
        self,
        queries: List[str],
        top_k: int = 15,
        selected_documents: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not queries:
            return []

        all_results_lists = []
        for query in queries:
            vector_res = self._vector_search(
                query, top_k=top_k, selected_documents=selected_documents
            )
            all_results_lists.append(vector_res)

            bm25_res = self._bm25_search(
                query, top_k=top_k, selected_documents=selected_documents
            )
            all_results_lists.append(bm25_res)

        fused_results = self._reciprocal_rank_fusion(all_results_lists)
        return fused_results[:top_k]

    def retrieve(
        self, query: str, top_k: int = 5, selected_documents: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        return self.retrieve_many(
            [query], top_k=top_k, selected_documents=selected_documents
        )

    def get_initial_chunks(self, doc_name: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Fetch the first few chunks of a document for summarization."""
        results = self.collection.get(
            where={"source_document": doc_name}, include=["documents", "metadatas"]
        )
        chunks = []
        if results and results.get("ids"):
            for i in range(min(limit, len(results["ids"]))):
                chunks.append(
                    {
                        "id": results["ids"][i],
                        "text": results["documents"][i],  # type: ignore
                        "metadata": results["metadatas"][i],  # type: ignore
                        "rerank_score": 10.0,
                    }
                )
        return chunks
