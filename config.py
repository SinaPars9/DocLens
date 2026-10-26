from pathlib import Path

# ==========================================
# Local Model Paths
# ==========================================
EMBEDDING_MODEL_DIR = Path(r"E:\models\bekko-embedding-v1-a8m")
RERANKER_MODEL_DIR = Path(r"E:\models\reranker")

# ==========================================
# RAG & Chunking Settings
# ==========================================

SEMANTIC_BUFFER_SIZE = 2          
SEMANTIC_THRESHOLD_PERCENTILE = 70 


FALLBACK_CHUNK_SIZE = 1000
FALLBACK_CHUNK_OVERLAP = 150

TOP_K_RETRIEVE = 15
TOP_K_FINAL = 8