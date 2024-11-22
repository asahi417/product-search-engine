import os
import json
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Union

import torch
from sentence_transformers import SentenceTransformer
from .magnus_encoder import MagnusEncoder
from ..util import get_logger, np_save

logger = get_logger(__name__)


class SemanticSearchTransformers:
    index_path: str
    query_path: str
    index_chunk: int
    query_chunk: int
    index_index2id: Optional[Dict[int, str]]
    query_index2id: Optional[Dict[int, str]]
    index_embedding: Optional[torch.Tensor]
    query_embedding: Optional[torch.Tensor]
    index_corpus: Optional[List[str]]
    query_corpus: Optional[List[str]]
    embedder: Union[SentenceTransformer, MagnusEncoder]

    def __init__(self,
                 index_path: str,
                 query_path: str,
                 index_expansion_path: Optional[str] = None,
                 index_chunk: int = 10_000,
                 query_chunk: int = 10_000,
                 index_expansion_chunk: int = 10_000,
                 model: str = "all-MiniLM-L6-v2",
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 magnus_encoder_tokenizer_path: str = 'experiment/all_queries/output/magnus_encoder_ckpt/hf_spm_vocab_150k_uncased',
                 magnus_encoder_model_path_q: str = 'experiment/all_queries/output/magnus_encoder_ckpt/mme15v1us_query_model_batched.onnx',
                 magnus_encoder_model_path_d: str = 'experiment/all_queries/output/magnus_encoder_ckpt/mme15v1us_asin_model.onnx'):
        self.index_path = index_path
        self.query_path = query_path
        self.index_expansion_path = index_expansion_path
        self.index_chunk = index_chunk
        self.query_chunk = query_chunk
        self.index_expansion_chunk = index_expansion_chunk
        self.index_index2id = None
        self.query_index2id = None
        self.index_embedding = None
        self.query_embedding = None
        self.index_corpus = None
        self.query_corpus = None
        if model == "magnus_encoder":
            self.embedder = MagnusEncoder(
                tokenizer_path=magnus_encoder_tokenizer_path,
                model_path_d=magnus_encoder_model_path_d,
                model_path_q=magnus_encoder_model_path_q
            )
        else:
            self.embedder = SentenceTransformer(model, model_kwargs=model_kwargs, trust_remote_code=True)

    def encode(self,
               corpus: List[str],
               output_dir: str,
               chunk_size: int,
               index2id: Optional[Dict[int, str]] = None,
               batch_size: int = 64,
               prompt_name: Optional[str] = None,
               prompt_prefix: Optional[str] = None,
               prompt_suffix: Optional[str] = None,
               magnus_encoder_is_query: bool = False) -> None:
        logger.info(f"generate embedding for column: {len(corpus)}")
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(f"{output_dir}/corpus.json"):
            with open(f"{output_dir}/corpus.json", "w") as f:
                json.dump({"corpus": corpus}, f)
        if not os.path.exists(f"{output_dir}/index2id.json"):
            if index2id is None:
                index2id = {n: str(n) for n in range(len(corpus))}
            with open(f"{output_dir}/index2id.json", "w") as f:
                json.dump(index2id, f)
        if prompt_suffix:
            logger.info(f"add suffix: {prompt_suffix}")
            corpus = [f"{i}{prompt_suffix}" for i in corpus]
        for start in tqdm(range(0, len(corpus), chunk_size)):
            end = min(start + chunk_size, len(corpus))
            filename = f"{output_dir}/embedding.{start}-{end}.npy"
            if not os.path.exists(filename):
                if type(self.embedder) is MagnusEncoder:
                    if magnus_encoder_is_query:
                        embedding = self.embedder.encode_q(corpus[start: end], batch_size=batch_size)
                    else:
                        embedding = self.embedder.encode_d(corpus[start: end], batch_size=batch_size)
                else:
                    embedding = self.embedder.encode(
                        corpus[start: end],
                        batch_size=batch_size,
                        prompt_name=prompt_name,
                        prompt=prompt_prefix,
                    )
                logger.info(f"embeddings computed for {start}-{end}: {embedding.shape}")
                np_save(embedding, filename)

    def encode_document(self,
                        corpus: List[str],
                        index2id: Optional[Dict[int, str]] = None,
                        batch_size: int = 64,
                        prompt_name: Optional[str] = None,
                        prompt_prefix: Optional[str] = None,
                        prompt_suffix: Optional[str] = None) -> None:
        logger.info(f"generate embedding for document: {len(corpus)}")
        self.encode(
            corpus,
            output_dir=self.index_path,
            chunk_size=self.index_chunk,
            index2id=index2id,
            batch_size=batch_size,
            prompt_name=prompt_name,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
            magnus_encoder_is_query=False
        )

    def encode_query(self,
                     corpus: List[str],
                     index2id: Optional[Dict[int, str]] = None,
                     batch_size: int = 64,
                     prompt_name: Optional[str] = None,
                     prompt_prefix: Optional[str] = None,
                     prompt_suffix: Optional[str] = None) -> None:
        logger.info(f"generate embedding for query: {len(corpus)}")
        self.encode(
            corpus,
            output_dir=self.query_path,
            chunk_size=self.query_chunk,
            index2id=index2id,
            batch_size=batch_size,
            prompt_name=prompt_name,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
            magnus_encoder_is_query=True
        )

    def encode_expansion(self,
                         corpus: List[str],
                         index2id: Optional[Dict[int, str]] = None,
                         batch_size: int = 64,
                         prompt_name: Optional[str] = None,
                         prompt_prefix: Optional[str] = None,
                         prompt_suffix: Optional[str] = None) -> None:
        logger.info(f"generate embedding for document: {len(corpus)}")
        self.encode(
            corpus,
            output_dir=self.index_expansion_path,
            chunk_size=self.index_expansion_chunk,
            index2id=index2id,
            batch_size=batch_size,
            prompt_name=prompt_name,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
            magnus_encoder_is_query=False
        )