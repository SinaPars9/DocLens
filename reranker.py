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

    def _should_boost_tables(self, question: str) -> bool:
        if re.search(r"\d", question):
            return True
        if any(sym in question for sym in ["£", "$", "%", "target", "goal", "margin"]):
            return True
        return False

    def rerank(self, question, chunks, top_k=5, batch_size=8):
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

        boost_tables = self._should_boost_tables(question)

        ranked_chunks = []
        for chunk, score in zip(chunks, all_scores):
            item = dict(chunk)
            final_score = float(score)

            if boost_tables and item.get("metadata", {}).get("type") == "table":
                final_score += 2.0

            item["rerank_score"] = final_score
            ranked_chunks.append(item)

        ranked_chunks.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked_chunks[:top_k]
