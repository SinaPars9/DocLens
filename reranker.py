
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class Reranker:
    def __init__(self, model_dir=r"E:\models\reranker"):
        model_dir = Path(model_dir)

        self.onnx_path = model_dir / "onnx" / "model.onnx"
        self.tokenizer_path = model_dir / "tokenizer.json"

        self.tokenizer = Tokenizer.from_file(
            str(self.tokenizer_path)
        )
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(
            pad_id=0,
            pad_token="[PAD]"
        )

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=["CPUExecutionProvider"]
        )

        self.input_names = {
            item.name for item in self.session.get_inputs()
        }

    def rerank(self, question, chunks, top_k=5):
        if not chunks:
            return []


        pairs = [
            (question, chunk["text"])
            for chunk in chunks
        ]

        encodings = self.tokenizer.encode_batch(pairs)

        input_ids = np.array(
            [item.ids for item in encodings],
            dtype=np.int64
        )
        attention_mask = np.array(
            [item.attention_mask for item in encodings],
            dtype=np.int64
        )
        token_type_ids = np.array(
            [item.type_ids for item in encodings],
            dtype=np.int64
        )

        inputs = {}

        if "input_ids" in self.input_names:
            inputs["input_ids"] = input_ids

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = token_type_ids

        outputs = self.session.run(None, inputs)
        scores = np.asarray(outputs[0]).reshape(-1)

        ranked_chunks = []

        for chunk, score in zip(chunks, scores):
            item = dict(chunk)
            item["rerank_score"] = float(score)
            ranked_chunks.append(item)

        ranked_chunks.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return ranked_chunks[:top_k]
