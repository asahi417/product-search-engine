import json
import os
import bm25s
from typing import Optional, List, Dict, Union
from ..util import get_logger, get_corpus_from_hf


logger = get_logger(__name__)


class LexicalSearchBM25S:
    index_path: str
    retriever: bm25s.BM25
    id2index: Optional[Dict[int, str]]

    def __init__(self, index_path: str):
        self.index_path = index_path
        self.retriever = bm25s.BM25()
        self.id2index = None

    def load_index(self) -> None:
        self.retriever = bm25s.BM25.load(self.index_path, load_corpus=True)
        with open(f"{self.index_path}/id2index.json") as f:
            self.id2index = {int(k): v for k, v in json.load(f).items()}

    def create_index(self,
                     dataset_path: str,
                     dataset_name: Optional[str] = None,
                     dataset_split: Optional[str] = "train",
                     dataset_id_column: Optional[str] = None,
                     dataset_column_names: Optional[List[str]] = None,
                     overwrite: bool = False) -> None:
        if not overwrite and os.path.exists(self.index_path) and os.path.exists(f"{self.index_path}/id2index.json"):
            return
        logger.info("create corpus")
        corpus, id2index = get_corpus_from_hf(
            dataset_path, dataset_name, dataset_split, dataset_column_names, dataset_id_column
        )
        if overwrite or not os.path.exists(self.index_path):
            logger.info(f"create the BM25 model and index the corpus: {len(corpus)} docs")
            corpus_tokens = bm25s.tokenize(texts=corpus, stopwords="en")
            self.retriever.index(corpus_tokens)
            self.retriever.save(self.index_path, corpus=corpus)
        if overwrite or not os.path.exists(f"{self.index_path}/id2index.json"):
            with open(f"{self.index_path}/id2index.json", "w") as f:
                json.dump(id2index, f)

    def search(self, query: List[str], k: int = 16) -> List[List[Dict[str, Union[str, float]]]]:
        if self.id2index is None:
            raise ValueError("load index before search")
        query_tokens = bm25s.tokenize(query)
        results, scores = self.retriever.retrieve(query_tokens, corpus=self.retriever.corpus, k=k)
        full_output = []
        for result, score in zip(results, scores):
            output = []
            for x, y in zip(result.tolist(), score.tolist()):
                x.update({"score": float(y)})
                if self.id2index:
                    x["id"] = self.id2index[x["id"]]
                output.append(x)
            full_output.append(output)
        return full_output
