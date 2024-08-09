import json
import os
import bm25s
from typing import Optional, List, Dict, Union
from ..util import get_logger


logger = get_logger(__name__)


class LexicalSearchBM25S:
    index_path: str
    retriever: bm25s.BM25
    index2id: Optional[Dict[int, str]]

    def __init__(self, index_path: str):
        self.index_path = index_path
        self.retriever = bm25s.BM25()
        self.index2id = None

    def load_index(self) -> None:
        self.retriever = bm25s.BM25.load(self.index_path, load_corpus=True)
        with open(f"{self.index_path}/index2id.json") as f:
            self.index2id = {int(k): v for k, v in json.load(f).items()}

    def create_index(self,
                     corpus: List[str],
                     index2id: Dict[int, str],
                     overwrite: bool = False) -> None:
        if not overwrite and os.path.exists(self.index_path) and os.path.exists(f"{self.index_path}/index2id.json"):
            return
        logger.info("create corpus")
        if overwrite or not os.path.exists(self.index_path):
            logger.info(f"create the BM25 model and index the corpus: {len(corpus)} docs")
            corpus_tokens = bm25s.tokenize(texts=corpus, stopwords="en")
            self.retriever.index(corpus_tokens)
            self.retriever.save(self.index_path, corpus=corpus)
        if overwrite or not os.path.exists(f"{self.index_path}/index2id.json"):
            with open(f"{self.index_path}/index2id.json", "w") as f:
                json.dump(index2id, f)

    def search(self, query: List[str], k: int = 16) -> List[List[Dict[str, Union[str, float]]]]:
        if self.index2id is None:
            raise ValueError("load index before search")
        query_tokens = bm25s.tokenize(query)
        results, scores = self.retriever.retrieve(query_tokens, corpus=self.retriever.corpus, k=k)
        full_output = []
        for result, score in zip(results, scores):
            output = []
            for x, y in zip(result.tolist(), score.tolist()):
                output.append({"id": self.index2id[x["id"]], "score": float(y), "text": x["text"]})
            full_output.append(output)
        return full_output
