from typing import Optional

from vector_store import VectorStore
from retriever import Retriever
from reranker import Reranker
from query_rewriter import resolve_and_expand, create_rerank_query
from generate import generate_answer


class RAGPipeline:
    def __init__(self):
        print("=" * 60)
        print("Initializing RAG pipeline...")
        print("=" * 60)

        self.vector_store = VectorStore()
        self.retriever = Retriever(store=self.vector_store)
        self.reranker = Reranker()

        self.is_built = self.vector_store.collection.count() > 0

        print(
            f"Pipeline ready. "
            f"Vector store: {self.vector_store.collection.count()} chunks."
        )

    def build(self, pdf_path: str) -> int:
        self.vector_store.build(pdf_path)
        self.retriever.collection = self.vector_store.get_collection()

        count = self.vector_store.collection.count()
        self.is_built = count > 0
        return count

    def answer(
        self,
        question: str,
        chat_history: Optional[list] = None,
        top_k_retrieve: int = 15,
        top_k_final: int = 5,
    ) -> dict:
        if not self.is_built:
            raise RuntimeError(
                "Pipeline has no data. Call build(pdf_path) first."
            )

        chat_history = chat_history or []

        result = resolve_and_expand(question, chat_history)

        if result.get("needs_clarification") or result.get("status") == "unclear":
            return {
                "answer": None,
                "status": "unclear",
                "needs_clarification": True,
                "clarification_queries": result.get("ambiguous_queries", []),
                "queries": [],
                "rerank_query": None,
                "chunks": [],
            }

        if result.get("status") == "resolved":
            standalone_query = result.get("standalone", question)
            expanded_queries = result.get("expanded", [])
            
            queries = [standalone_query] + expanded_queries
            
            chunks = self.retriever.retrieve_many(
                queries, top_k=top_k_retrieve
            )
            
            rerank_query = standalone_query
            ranked_chunks = self.reranker.rerank(
                question=rerank_query,
                chunks=chunks,
                top_k=top_k_final,
            )

        else:
            queries = result.get("ambiguous_queries", [])
            
            chunks = self.retriever.retrieve_many(
                queries, top_k=top_k_retrieve
            )
            
            rerank_query = create_rerank_query(
                question,
                queries,
                chunks,
            )
            
            ranked_chunks = self.reranker.rerank(
                question=rerank_query,
                chunks=chunks,
                top_k=top_k_final,
            )

        if not ranked_chunks:
            return {
                "answer": "No relevant information found in the document.",
                "status": result.get("status", "unknown"),
                "needs_clarification": False,
                "queries": queries,
                "rerank_query": rerank_query,
                "chunks": [],
            }

        answer = generate_answer(
            question=question,
            chunks=ranked_chunks,
            chat_history=chat_history,
        )

        return {
            "answer": answer,
            "status": result.get("status", "resolved"),
            "needs_clarification": False,
            "queries": queries,
            "rerank_query": rerank_query,
            "chunks": ranked_chunks,
        }