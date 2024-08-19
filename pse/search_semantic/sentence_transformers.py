import os
import json
from typing import Optional, List, Dict, Union, Any
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from ..util import get_logger, np_save, np_load

logger = get_logger(__name__)


class SemanticSearchTransformers:
    index_path: str
    embedder: SentenceTransformer
    index2id: Optional[Dict[int, str]]
    embedding: Optional[torch.Tensor]
    corpus: Optional[List[str]]

    def __init__(self,
                 index_path: str,
                 model: str = "all-MiniLM-L6-v2",
                 model_kwargs: Optional[Dict[str, Any]] = None):
        self.index_path = index_path
        self.embedder = SentenceTransformer(model, model_kwargs=model_kwargs, trust_remote_code=True)
        self.index2id = None
        self.embedding = None
        self.corpus = None

    def load_index(self) -> None:
        with open(f"{self.index_path}/index2id.json") as f:
            self.index2id = {int(k): v for k, v in json.load(f).items()}
        with open(f"{self.index_path}/corpus.json") as f:
            self.corpus = json.load(f)["corpus"]
        self.embedding = torch.as_tensor(np_load(f"{self.index_path}/embedding.npy"))

    def create_index(self,
                     corpus: List[str],
                     index2id: Optional[Dict[int, str]] = None,
                     batch_size: int = 64,
                     prompt_name: Optional[str] = None,
                     prompt_prefix: Optional[str] = None,
                     prompt_suffix: Optional[str] = None,
                     overwrite: bool = False) -> None:
        if (not overwrite and os.path.exists(f"{self.index_path}/corpus.json") and
                os.path.exists(f"{self.index_path}/embedding.npy") and
                os.path.exists(f"{self.index_path}/index2id.json")):
            return

        os.makedirs(self.index_path, exist_ok=True)
        logger.info("create corpus")
        if overwrite or not os.path.exists(f"{self.index_path}/corpus.json"):
            with open(f"{self.index_path}/corpus.json", "w") as f:
                json.dump({"corpus": corpus}, f)
        if overwrite or not os.path.exists(f"{self.index_path}/embedding.npy"):
            logger.info(f"generate embedding for column: {len(corpus)}")
            if prompt_suffix:
                corpus = [f"{i}{prompt_suffix}" for i in corpus]
            embedding = self.embedder.encode(
                corpus,
                batch_size=batch_size,
                prompt_name=prompt_name,
                prompt=prompt_prefix,
            )
            logger.info(f"embeddings computed. Shape: {embedding.shape}")
            np_save(embedding, f"{self.index_path}/embedding.npy")
        if overwrite or not os.path.exists(f"{self.index_path}/index2id.json"):
            with open(f"{self.index_path}/index2id.json", "w") as f:
                json.dump(index2id, f)

    def search(self,
               query: List[str],
               k: int = 16,
               batch_size: int = 64,
               prompt_name: Optional[str] = None,
               prompt_prefix: Optional[str] = None,
               prompt_suffix: Optional[str] = None,
               query_chunk_size: int = 100,
               corpus_chunk_size: int = 500000) -> List[List[Dict[str, Union[str, float]]]]:
        if self.index2id is None or self.embedding is None or self.corpus is None:
            raise ValueError("load index before search")
        logger.info(f"embed queries: {len(query)}")
        if prompt_suffix:
            query = [f"{i}{prompt_suffix}" for i in query]
        query_embedding = self.embedder.encode(
            query,
            convert_to_tensor=True,
            batch_size=batch_size,
            prompt_name=prompt_name,
            prompt=prompt_prefix,
        )
        search_result = semantic_search(
            query_embedding.to(self.embedder.device),
            self.embedding.to(self.embedder.device),
            query_chunk_size=query_chunk_size,
            corpus_chunk_size=corpus_chunk_size,
            top_k=k
        )
        full_output = []
        for result in search_result:
            output = []
            for r in result:
                output.append({
                    "id": self.index2id[r["corpus_id"]],
                    "text": self.corpus[r["corpus_id"]],
                    "score": r["score"]
                })
            full_output.append(output)
        return full_output
