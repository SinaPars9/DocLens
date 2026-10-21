from typing import Optional, List, Dict, Any, Generator

from vector_store import VectorStore
from retriever import Retriever
from reranker import Reranker
from query_rewriter import resolve_and_expand, create_rerank_query
from generate import generate_answer, generate_answer_stream


class RAGPipeline:
    def __init__(self) -> None:
        print("=" * 60)
        print("Initializing RAG pipeline...")
        print("=" * 60)

        self.vector_store = VectorStore()
        self.retriever = Retriever(store=self.vector_store)
        self.reranker = Reranker()

        self.is_built: bool = self.vector_store.collection.count() > 0

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
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k_retrieve: int = 15,
        top_k_final: int = 5,
    ) -> Dict[str, Any]:
        if not self.is_built:
            raise RuntimeError("Pipeline has no data. Call build(pdf_path) first.")

        history = chat_history or []
        result = resolve_and_expand(question, history)

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

        chunks: List[Dict[str, Any]] = []
        queries: List[str] = []
        rerank_query: str = question
        
        if result.get("status") == "resolved":
            standalone_query = result.get("standalone", question)
            queries = [standalone_query] + result.get("expanded", [])
            rerank_query = standalone_query
            chunks = self.retriever.retrieve_many(queries, top_k=top_k_retrieve)
        else:
            queries = result.get("ambiguous_queries", [])
            chunks = self.retriever.retrieve_many(queries, top_k=top_k_retrieve)
            rerank_query = create_rerank_query(question, queries, chunks)

        ranked_chunks = self.reranker.rerank(rerank_query, chunks, top_k=top_k_final)

        if not ranked_chunks:
            return {
                "answer": "No relevant information found in the document.",
                "status": result.get("status", "unknown"),
                "needs_clarification": False,
                "queries": queries,
                "rerank_query": rerank_query,
                "chunks": [],
            }

        final_answer = generate_answer(question, ranked_chunks, history)

        return {
            "answer": final_answer,
            "status": result.get("status", "resolved"),
            "needs_clarification": False,
            "queries": queries,
            "rerank_query": rerank_query,
            "chunks": ranked_chunks,
        }

    def answer_stream(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k_retrieve: int = 15,
        top_k_final: int = 5,
    ) -> Generator[Dict[str, Any], None, None]:
        if not self.is_built:
            yield {"type": "error", "message": "Pipeline has no data."}
            return

        history = chat_history or []
        result = resolve_and_expand(question, history)

        if result.get("needs_clarification") or result.get("status") == "unclear":
            yield {
                "type": "error",
                "message": "I need more clarity. Which did you mean?",
                "queries": result.get("ambiguous_queries", [])
            }
            return

        chunks: List[Dict[str, Any]] = []
        queries: List[str] = []
        rerank_query: str = question

        if result.get("status") == "resolved":
            standalone_query = result.get("standalone", question)
            queries = [standalone_query] + result.get("expanded", [])
            rerank_query = standalone_query
            chunks = self.retriever.retrieve_many(queries, top_k=top_k_retrieve)
        else:
            queries = result.get("ambiguous_queries", [])
            chunks = self.retriever.retrieve_many(queries, top_k=top_k_retrieve)
            rerank_query = create_rerank_query(question, queries, chunks)

        ranked_chunks = self.reranker.rerank(rerank_query, chunks, top_k=top_k_final)

        yield {"type": "chunks", "chunks": ranked_chunks}

        if not ranked_chunks:
            yield {"type": "text", "content": "No relevant information found in the document."}
            return

        for token in generate_answer_stream(question, ranked_chunks, history):
            yield {"type": "text", "content": token}