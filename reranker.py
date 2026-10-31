import re
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class Reranker:
    def __init__(self, model_dir=r"E:\models\reranker"):
        model_dir = Path(model_dir)

        self.onnx_path = model_dir / "onnx" / "model.onnx"
        self.tokenizer_path = model_dir / "tokenizer.json"

        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_names = {item.name for item in self.session.get_inputs()}

        self.stopwords = {
            "what", "was", "the", "of", "in", "for", "a", "is", "to", "and",
            "how", "much", "did", "by", "were", "are", "on", "with", "that", 
            "this", "it", "be", "from", "at", "an", "as", "which", "or", "who",
            "whom", "whose", "where", "when", "why", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "can", "will",
            "just", "should", "now"
        }

    def _extract_query_features(self, question: str):
        numbers = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b", question)

        words = re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())
        keywords = [w for w in words if w not in self.stopwords]

        return numbers, keywords

    def _calculate_table_boost(self, question: str, chunk_text: str) -> float:
        numbers, keywords = self._extract_query_features(question)

        if not numbers and not keywords:
            return 0.0

        chunk_lower = chunk_text.lower()
        boost_score = 0.0

        for num in numbers:
            if num in chunk_text:
                boost_score += 1.5

        for kw in keywords:
            if kw in chunk_lower:
                boost_score += 0.5

        return min(boost_score, 3.0)

    def rerank(
        self, question: str, chunks: list, top_k: int = 5, batch_size: int = 8
    ) -> list:
        if not chunks:
            return []

        pairs = [(question, chunk["text"]) for chunk in chunks]
        all_scores = []

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            encodings = self.tokenizer.encode_batch(batch_pairs)

            input_ids = np.array([item.ids for item in encodings], dtype=np.int64)
            attention_mask = np.array(
                [item.attention_mask for item in encodings], dtype=np.int64
            )
            token_type_ids = np.array(
                [item.type_ids for item in encodings], dtype=np.int64
            )

            inputs = {}
            if "input_ids" in self.input_names:
                inputs["input_ids"] = input_ids
            if "attention_mask" in self.input_names:
                inputs["attention_mask"] = attention_mask
            if "token_type_ids" in self.input_names:
                inputs["token_type_ids"] = token_type_ids

            outputs = self.session.run(None, inputs)
            batch_scores = np.asarray(outputs[0]).reshape(-1)
            all_scores.extend(batch_scores)

        ranked_chunks = []
        for chunk, score in zip(chunks, all_scores):
            item = dict(chunk)
            final_score = float(score)

            if item.get("metadata", {}).get("type") == "table":
                dynamic_boost = self._calculate_table_boost(question, item["text"])
                final_score += dynamic_boost

            item["rerank_score"] = final_score
            ranked_chunks.append(item)

        ranked_chunks.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked_chunks[:top_k]
