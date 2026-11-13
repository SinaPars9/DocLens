import hashlib
import os
from typing import Optional, List, Dict, Any, Generator, cast

from vector_store import VectorStore
from retriever import Retriever
from reranker import Reranker
from query_rewriter import resolve_and_expand
from generate import generate_answer_stream
from config import TOP_K_RETRIEVE, TOP_K_FINAL
from semantic_cache import SemanticCache
from verifier import AnswerVerifier


class RAGPipeline:
    def __init__(self) -> None:
        print("=" * 60)
        print("Initializing RAG pipeline...")
        print("=" * 60)

        self.vector_store = VectorStore()
        self.retriever = Retriever(store=self.vector_store)
        self.reranker = Reranker()
        self.verifier = AnswerVerifier()
        self.cache = SemanticCache(embedding_model=self.vector_store.embedding_model)

        self.is_built: bool = self.vector_store.collection.count() > 0
        self.document_id: str = "default"
        self.ingested_documents: List[str] = []
        print(
            f"Pipeline ready. Vector store: {self.vector_store.collection.count()} chunks."
        )

    def build(self, pdf_paths: List[str], use_llm_context: bool = True) -> int:
        file_hasher = hashlib.md5()
        for path in pdf_paths:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    file_hasher.update(chunk)
        
        self.document_id = file_hasher.hexdigest()[:12]
        
        self.cache.invalidate_document(self.document_id)

        clean_names = [os.path.basename(p).replace("temp_", "") for p in pdf_paths]
        self.ingested_documents = clean_names

        self.vector_store.build(pdf_paths, use_llm_context=use_llm_context)
        self.retriever.collection = self.vector_store.get_collection()
        self.retriever._build_bm25_index()

        count = self.vector_store.collection.count()
        self.is_built = count > 0
        if not self.is_built:
            raise ValueError(
                "Vector store build failed or resulted in an empty collection."
            )
        return count

    def answer_stream(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        selected_documents: Optional[List[str]] = None,
        top_k_retrieve: int = TOP_K_RETRIEVE,
        top_k_final: int = TOP_K_FINAL,
    ) -> Generator[Dict[str, Any], None, None]:
        if not self.is_built:
            yield {"type": "error", "message": "Pipeline has no data."}
            return

        history = chat_history or []

        result = cast(Dict[str, Any], resolve_and_expand(question, history))
        standalone_query = str(result.get("standalone", question))
        expanded = result.get("expanded", [])

        yield {
            "type": "query_info",
            "standalone": standalone_query,
            "expanded": expanded,
        }

        cache_doc_id = f"{self.document_id}_{'_'.join(sorted(selected_documents)) if selected_documents else 'all'}"

        cached_result = self.cache.get(standalone_query, document_id=cache_doc_id)
        if cached_result:
            cached_answer, cached_chunks = cached_result
            yield {"type": "chunks", "chunks": cached_chunks}
            yield {"type": "text", "content": cached_answer}
            verification_result = self.verifier.verify_answer(
                cached_answer, cached_chunks
            )
            yield {"type": "verification", "data": verification_result}
            return

        queries = [standalone_query] + [str(q) for q in expanded]

        queries = [standalone_query] + [str(q) for q in expanded]
        import re

        query_clean = standalone_query.strip().lower().rstrip("?.!,")

        meta_patterns = [
            # "what is this file about", "what are these documents"
            r"^(what|tell me)\s+(is|are)\s+(this|these)\s+(file|files|document|documents|pdf|pdfs)\s*(about|for)?$",
            # "summarize this", "summarize the document", "please summarize the pdf"
            r"^(can you\s+|please\s+)?(summarize|summarise)\s+(this|these|the)\s*(file|files|document|documents|pdf|pdfs|content)?$",
            r"^(summarize|summarise|summary|overview)$",
            # "give me an overview", "give me a summary of the document"
            r"^(give me\s+)?(an?\s+)?(overview|summary)\s+(of\s+)?(this|these|the)\s*(file|files|document|documents|pdf|pdfs|content)?$",
            # "what is in this document", "what is inside these files"
            r"^what\s+(is|are)\s+(in|inside)\s+(this|these)\s+(file|files|document|documents|pdf|pdfs)$",
            # "describe this file", "explain these documents"
            r"^(describ|explain|analy[sz]e)\s+(this|these)\s+(file|files|document|documents|pdf|pdfs)$",
        ]

        is_meta_query = any(re.match(pattern, query_clean) for pattern in meta_patterns)

        if is_meta_query and selected_documents:
            print(f"[Meta-Query Detected]: {standalone_query}")
            chunks = []
            for doc in selected_documents:
                doc_chunks = self.retriever.get_initial_chunks(doc, limit=3)
                chunks.extend(doc_chunks)

            if chunks:
                for c in chunks:
                    c["rerank_score"] = 10.0
                filtered_chunks = chunks
            else:
                filtered_chunks = []

        elif selected_documents and len(selected_documents) > 1:

            combined_ranked_chunks: List[Dict[str, Any]] = []

            chunks_per_doc = max(3, top_k_final // len(selected_documents))

            for doc in selected_documents:
                doc_chunks = self.retriever.retrieve_many(
                    queries, top_k=top_k_retrieve, selected_documents=[doc]
                )
                if doc_chunks:
                    ranked_doc_chunks = self.reranker.rerank(
                        standalone_query, doc_chunks, top_k=chunks_per_doc
                    )
                    combined_ranked_chunks.extend(ranked_doc_chunks)

            if not combined_ranked_chunks:
                yield {"type": "chunks", "chunks": []}
                yield {
                    "type": "text",
                    "content": "No relevant information found in the selected documents.",
                }
                return

            ranked_chunks = sorted(
                combined_ranked_chunks,
                key=lambda x: x.get("rerank_score", 0.0),
                reverse=True,
            )

            top_score = ranked_chunks[0].get("rerank_score", 0.0)
            ABSOLUTE_IRRELEVANCE_THRESHOLD = -10.0
            if top_score < ABSOLUTE_IRRELEVANCE_THRESHOLD:
                yield {"type": "chunks", "chunks": []}
                yield {
                    "type": "text",
                    "content": "The selected documents do not seem to contain relevant information to answer this question.",
                }
                return

            guaranteed_chunks: List[Dict[str, Any]] = []
            for doc in selected_documents:
                doc_top = next(
                    (
                        c
                        for c in ranked_chunks
                        if c.get("metadata", {}).get("source_document") == doc
                    ),
                    None,
                )
                if doc_top:
                    guaranteed_chunks.append(doc_top)

            guaranteed_ids = {c["id"] for c in guaranteed_chunks}
            filtered_chunks = list(guaranteed_chunks)

            MIN_RERANK_SCORE = -5.0
            for c in ranked_chunks:
                if (
                    c["id"] not in guaranteed_ids
                    and c.get("rerank_score", 0.0) >= MIN_RERANK_SCORE
                ):
                    filtered_chunks.append(c)

            filtered_chunks = filtered_chunks[:top_k_final]
        else:
            chunks = self.retriever.retrieve_many(
                queries, top_k=top_k_retrieve, selected_documents=selected_documents
            )
            ranked_chunks = self.reranker.rerank(
                standalone_query, chunks, top_k=top_k_final
            )

            if not ranked_chunks:
                yield {"type": "chunks", "chunks": []}
                yield {
                    "type": "text",
                    "content": "No relevant information found in the selected documents.",
                }
                return

            top_score = ranked_chunks[0].get("rerank_score", 0.0)
            ABSOLUTE_IRRELEVANCE_THRESHOLD = -8.0
            if top_score < ABSOLUTE_IRRELEVANCE_THRESHOLD:
                yield {"type": "chunks", "chunks": []}
                yield {
                    "type": "text",
                    "content": "The selected document does not seem to contain relevant information to answer this question.",
                }
                return

            MIN_RERANK_SCORE = -5.0
            filtered_chunks = [
                c
                for c in ranked_chunks
                if c.get("rerank_score", 0.0) >= MIN_RERANK_SCORE
            ]
            if not filtered_chunks:
                filtered_chunks = ranked_chunks[:3]
                print(
                    "[Context Compression] All chunks below threshold. Kept top 3 as fallback."
                )

        parent_ids_seen = set()
        generation_chunks = []
        ui_display_chunks = []

        for c in filtered_chunks:
            pid = c.get("metadata", {}).get("parent_id")
            unique_id = pid if pid else c.get("id", id(c))

            if unique_id not in parent_ids_seen:
                parent_ids_seen.add(unique_id)
                parent_text = c.get("metadata", {}).get("parent_text", c["text"])

                generation_chunks.append(
                    {"text": parent_text, "metadata": c.get("metadata", {})}
                )
                ui_display_chunks.append(
                    {
                        "text": parent_text,
                        "metadata": c.get("metadata", {}),
                        "rerank_score": c.get("rerank_score", 0.0),
                        "child_count": (
                            sum(
                                1
                                for fc in filtered_chunks
                                if fc.get("metadata", {}).get("parent_id") == pid
                            )
                            if pid
                            else 1
                        ),
                    }
                )

        yield {"type": "chunks", "chunks": ui_display_chunks}

        full_response = ""
        for token in generate_answer_stream(
            question=question,
            standalone_query=standalone_query,
            chunks=generation_chunks,
            chat_history=history,
            document_id=self.document_id,
        ):
            full_response += str(token)
            yield {"type": "text", "content": token}

        self.cache.put(
            standalone_query, full_response, filtered_chunks, document_id=cache_doc_id
        )
        verification_result = self.verifier.verify_answer(
            full_response, filtered_chunks
        )
        yield {"type": "verification", "data": verification_result}
