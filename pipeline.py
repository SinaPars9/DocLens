from typing import Optional, List, Dict, Any, Generator, cast

from vector_store import VectorStore
from retriever import Retriever
from reranker import Reranker
from query_rewriter import resolve_and_expand
from generate import generate_answer_stream


class RAGPipeline:
    def __init__(self) -> None:
        print("=" * 60)
        print("Initializing RAG pipeline...")
        print("=" * 60)

        self.vector_store = VectorStore()
        self.retriever = Retriever(store=self.vector_store)
        self.reranker = Reranker()

        self.is_built: bool = self.vector_store.collection.count() > 0
        print(f"Pipeline ready. Vector store: {self.vector_store.collection.count()} chunks.")

    def build(self, pdf_path: str) -> int:
        self.vector_store.build(pdf_path)
        self.retriever.collection = self.vector_store.get_collection()
        self.retriever._build_bm25_index()

        count = self.vector_store.collection.count()
        self.is_built = count > 0
        if not self.is_built:
            raise ValueError("Vector store build failed or resulted in an empty collection.")
        return count

    def answer_stream(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k_retrieve: int = 15,
        top_k_final: int = 8,
    ) -> Generator[Dict[str, Any], None, None]:
        if not self.is_built:
            yield {"type": "error", "message": "Pipeline has no data."}
            return

        history = chat_history or []
        
        result = cast(Dict[str, Any], resolve_and_expand(question, history))
        
        standalone_query = str(result.get("standalone", question))
        expanded = result.get("expanded", [])
        
        queries = [standalone_query] + [str(q) for q in expanded]
        chunks = self.retriever.retrieve_many(queries, top_k=top_k_retrieve)
        
        ranked_chunks = self.reranker.rerank(standalone_query, chunks, top_k=top_k_final)

        yield {"type": "chunks", "chunks": ranked_chunks}

        if not ranked_chunks:
            yield {"type": "text", "content": "No relevant information found in the document."}
            return

        for token in generate_answer_stream(question, ranked_chunks, history):
            yield {"type": "text", "content": token}