
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


MODEL_DIR = Path(r"E:\models\bekko-embedding-v1-a8m")
ONNX_PATH = MODEL_DIR / "onnx" / "model.onnx"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

MAX_LENGTH = 1024
EMBED_BATCH_SIZE = 16


class Embeddings:
    def __init__(self):
        print("Loading Bekko ONNX model...")
        start = time.time()

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(ONNX_PATH),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        self.tokenizer.enable_padding(
            pad_id=0,
            pad_token="[PAD]",
        )
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)

        self.input_names = [
            item.name for item in self.session.get_inputs()
        ]

        print(f"Model loaded in {time.time() - start:.1f}s")

    def embed_texts(self, texts):
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_vectors = []

        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            encodings = self.tokenizer.encode_batch(batch)

            input_ids = np.array(
                [item.ids for item in encodings],
                dtype=np.int64,
            )
            attention_mask = np.array(
                [item.attention_mask for item in encodings],
                dtype=np.int64,
            )

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            if "token_type_ids" in self.input_names:
                inputs["token_type_ids"] = np.zeros_like(input_ids)

            outputs = self.session.run(None, inputs)
            last_hidden = outputs[0]

            mask = attention_mask[..., np.newaxis].astype(np.float32)

            summed = (last_hidden * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), 1e-9, None)
            vectors = summed / counts

            all_vectors.append(vectors)

        embeddings = np.concatenate(all_vectors, axis=0)
        embeddings = embeddings.astype(np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings / np.clip(norms, 1e-9, None)
