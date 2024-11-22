from typing import List, Dict, Optional

import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, PreTrainedTokenizer


class MagnusEncoder:

    tokenizer: PreTrainedTokenizer
    model_q: ort.InferenceSession
    model_d: ort.InferenceSession
    seq_length_q: int = 32
    seq_length_d: int = 128

    def __init__(
            self,
            tokenizer_path: str = 'experiment/all_queries/output/magnus_encoder_ckpt/hf_spm_vocab_150k_uncased',
            model_path_q: str ='experiment/all_queries/output/magnus_encoder_ckpt/mme15v1us_query_model_batched.onnx',
            model_path_d: str ='experiment/all_queries/output/magnus_encoder_ckpt/mme15v1us_asin_model.onnx',
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model_q = ort.InferenceSession(model_path_q)
        self.model_d = ort.InferenceSession(model_path_d)

    def encode(self,
               text: List[str],
               seq_length: int,
               model: ort.InferenceSession,
               batch_size: Optional[int] = None) -> np.array:
        tokens = self.tokenizer(text)["input_ids"]
        tokens = [t[:seq_length] + [0] * (seq_length - len(t)) for t in tokens]
        tokens = np.asarray(tokens)
        tokens_masked = np.asarray(tokens > 0, dtype=int)
        batch_size = len(tokens_masked) if batch_size is None else batch_size
        index = 0
        embeddings = []
        while index * batch_size < len(tokens):
            token_ids = tokens[index * batch_size: (index + 1) * batch_size]
            token_mask = tokens_masked[index * batch_size: (index + 1) * batch_size]
            v = model.run(None, {'token_ids': token_ids, 'token_mask': token_mask})[0]
            print(v.shape, index, len(token_ids))
            embeddings.append(v)
            index += 1
        return np.concatenate(embeddings)

    def encode_q(self, text: List[str], batch_size: Optional[int] = None) -> np.array:
        return self.encode(text, seq_length=self.seq_length_q, model=self.model_q, batch_size=batch_size)

    def encode_d(self, text: List[str], batch_size: Optional[int] = None) -> np.array:
        return self.encode(text, seq_length=self.seq_length_d, model=self.model_d, batch_size=batch_size)


if __name__ == '__main__':
    _model = MagnusEncoder(
        tokenizer_path="/Users/asahiu/workplace/SymptomQueryExperiment/experiment/all_queries/output/magnus_encoder_ckpt/hf_spm_vocab_150k_uncased",
        model_path_d="/Users/asahiu/workplace/SymptomQueryExperiment/experiment/all_queries/output/magnus_encoder_ckpt/mme15v1us_asin_model.onnx",
        model_path_q="/Users/asahiu/workplace/SymptomQueryExperiment/experiment/all_queries/output/magnus_encoder_ckpt/mme15v1us_query_model_batched.onnx"
    )
    d = _model.encode_d(["test", "smile smile", "xyz"], batch_size=2)
    print(d.shape)  # (2, 256)
    q = _model.encode_q(["test", "smile"], batch_size=1)
    print(q.shape)  # (2, 256)
