from vector_store import VectorStore

class Retriever:
    def __init__(self, store=None):
        self.store = store or VectorStore()
        self.collection = self.store.get_collection()

    def retrieve(self, query, top_k=5):
        query_vector = self.store.embedding_model.embed_texts([query])[0]

        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        
        retrieved_chunks = []
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            
            for i in range(len(ids)):
                retrieved_chunks.append({
                    "id": ids[i],
                    "text": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distances[i] if i < len(distances) else 0.0,
                })
        return retrieved_chunks

    def retrieve_many(self, queries, top_k=5):
        if not queries:
            return []

        query_vectors = self.store.embedding_model.embed_texts(queries)
        query_embeddings_list = [vec.tolist() for vec in query_vectors]


        results = self.collection.query(
            query_embeddings=query_embeddings_list,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        
        retrieved_chunks = []
        
        if results and results.get("ids"):
            for q_idx in range(len(queries)):
                q_ids = results["ids"][q_idx] if q_idx < len(results["ids"]) and results["ids"][q_idx] else []
                
                raw_docs = results.get("documents")
                q_docs = raw_docs[q_idx] if raw_docs and q_idx < len(raw_docs) else []
                
                raw_metas = results.get("metadatas")
                q_metas = raw_metas[q_idx] if raw_metas and q_idx < len(raw_metas) else []
                
                raw_dists = results.get("distances")
                q_dists = raw_dists[q_idx] if raw_dists and q_idx < len(raw_dists) else []

                for i in range(len(q_ids)):
                    retrieved_chunks.append({
                        "id": q_ids[i],
                        "text": q_docs[i] if i < len(q_docs) else "",
                        "metadata": q_metas[i] if i < len(q_metas) else {},
                        "distance": q_dists[i] if i < len(q_dists) else 0.0,
                    })
                
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