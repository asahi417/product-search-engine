import os
import json
from glob import glob
from tqdm import tqdm
from typing import Optional, List, Dict, Union, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from ..util import get_logger, np_save, np_load

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
    embedder: SentenceTransformer

    def __init__(self,
                 index_path: str,
                 query_path: str,
                 index_chunk: int = 10_000,
                 query_chunk: int = 10_000,
                 model: str = "all-MiniLM-L6-v2",
                 model_kwargs: Optional[Dict[str, Any]] = None):
        self.index_path = index_path
        self.query_path = query_path
        self.index_chunk = index_chunk
        self.query_chunk = query_chunk
        self.index_index2id = None
        self.query_index2id = None
        self.index_embedding = None
        self.query_embedding = None
        self.index_corpus = None
        self.query_corpus = None
        self.embedder = SentenceTransformer(model, model_kwargs=model_kwargs, trust_remote_code=True)

    @staticmethod
    def load_index(output_dir) -> (Dict[int, str], List[str], torch.Tensor):
        with open(f"{output_dir}/index2id.json") as f:
            index2id = {int(k): v for k, v in json.load(f).items()}
        with open(f"{output_dir}/corpus.json") as f:
            corpus = json.load(f)["corpus"]
        embedding = []
        for numpy_file in glob(f"{output_dir}/embedding.*.npy"):
            embedding.append(np_load(numpy_file))
        embedding = torch.as_tensor(np.concatenate(embedding))
        return index2id, corpus, embedding

    def encode(self,
               corpus: List[str],
               output_dir: str,
               chunk_size: int,
               index2id: Optional[Dict[int, str]] = None,
               batch_size: int = 64,
               prompt_name: Optional[str] = None,
               prompt_prefix: Optional[str] = None,
               prompt_suffix: Optional[str] = None) -> None:
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
            prompt_suffix=prompt_suffix
        )
        self.index_index2id, self.index_corpus, self.index_embedding = self.load_index(self.index_path)

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
            prompt_suffix=prompt_suffix
        )
        self.query_index2id, self.query_corpus, self.query_embedding = self.load_index(self.query_path)

    def search(self,
               k: int = 16,
               query_chunk_size: int = 100,
               corpus_chunk_size: int = 500000) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        search_result = semantic_search(
            self.query_embedding.to(self.embedder.device),
            self.index_embedding.to(self.embedder.device),
            query_chunk_size=query_chunk_size,
            corpus_chunk_size=corpus_chunk_size,
            top_k=k
        )
        full_output = {}
        for n, result in enumerate(search_result):
            output = []
            for r in result:
                output.append({
                    "id": self.index_index2id[r["corpus_id"]],
                    "text": self.index_corpus[r["corpus_id"]],
                    "score": r["score"]
                })
            full_output[self.query_index2id[n]] = output
        return full_output
