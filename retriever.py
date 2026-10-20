from vector_store import VectorStore

class Retriever:
    def __init__(self, store=None):
        self.store = store or VectorStore()
        self.collection = self.store.get_collection()

    def _parse_chroma_results(self, results):
        """Helper to safely extract data from chromadb QueryResult."""
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        
        docs_outer = results.get("documents") or [[]]
        metas_outer = results.get("metadatas") or [[]]
        dists_outer = results.get("distances") or [[]]

        documents = docs_outer[0] if docs_outer else []
        metadatas = metas_outer[0] if metas_outer else []
        distances = dists_outer[0] if dists_outer else []

        parsed_chunks = []
        for i in range(len(ids)):
            parsed_chunks.append({
                "id": ids[i],
                "text": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": distances[i] if i < len(distances) else 0.0,
            })
        return parsed_chunks

    def retrieve(self, query, top_k=5):
        query_vector = self.store.embedding_model.embed_texts([query])[0]

        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return self._parse_chroma_results(results)

    def retrieve_many(self, queries, top_k=5):
        retrieved_chunks = []
        for query in queries:
            query_vector = self.store.embedding_model.embed_texts([query])[0]
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            retrieved_chunks.extend(self._parse_chroma_results(results))
                
        unique_chunks = {}

        for chunk in retrieved_chunks:
            chunk_id = chunk["id"]
            if chunk_id not in unique_chunks:
                unique_chunks[chunk_id] = chunk
            elif chunk["distance"] < unique_chunks[chunk_id]["distance"]:
                unique_chunks[chunk_id] = chunk
                
        unique_chunks_list = list(unique_chunks.values())
        unique_chunks_list.sort(key=lambda chunk: chunk["distance"])
        return unique_chunks_list