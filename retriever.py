
from vector_store import VectorStore
from query_rewriter import resolve_query

class Retriever:
    def __init__(self,store=None):
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
        for i in range(len(results["ids"][0])):
            retrieved_chunks.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return retrieved_chunks
    def retrieve_many(self,queries, top_k=5):
        retrieved_chunks = []
        for query in queries:
            query_vector = self.store.embedding_model.embed_texts([query])[0]
            results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
            for i in range(len(results["ids"][0])):
                retrieved_chunks.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })
        unique_chunks = {}

        for chunk in retrieved_chunks:
            chunk_id = chunk["id"]

            if chunk_id not in unique_chunks:
                unique_chunks[chunk_id] = chunk

            elif chunk["distance"] < unique_chunks[chunk_id]["distance"]:
                unique_chunks[chunk_id] = chunk
        unique_chunks = list(unique_chunks.values())
        unique_chunks.sort(key=lambda chunk: chunk["distance"])
        return unique_chunks

