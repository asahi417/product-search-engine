import logging
import json
from gc import collect
from glob import glob
from typing import List, Dict, Union

import numpy as np
import torch
from sentence_transformers.util import semantic_search


def clear_cache():
    torch.cuda.empty_cache()
    collect()


def np_save(array: np.ndarray, path: str) -> None:
    with open(path, 'wb') as f:
        np.save(f, array)


def np_load(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        return np.load(f)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    return logger


def get_semantic_search_result(
        query_path: str,
        index_path: str,
        k: int = 16,
        query_chunk_size: int = 400,
        corpus_chunk_size: int = 400000) -> Dict[str, List[Dict[str, Union[str, float]]]]:
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

    logger = get_logger(__name__)
    query_index2id, query_corpus, query_embedding = load_index(query_path)
    logger.info(f"load query: {query_embedding.shape}")
    index_index2id, index_corpus, index_embedding = load_index(index_path)
    logger.info(f"load document: {query_embedding.shape}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    search_result = semantic_search(
        query_embedding.to(device),
        index_embedding.to(device),
        query_chunk_size=query_chunk_size,
        corpus_chunk_size=corpus_chunk_size,
        top_k=k
    )
    full_output = {}
    for n, result in enumerate(search_result):
        output = []
        for r in result:
            output.append({
                "id": index_index2id[r["corpus_id"]],
                "text": index_corpus[r["corpus_id"]],
                "score": r["score"]
            })
        full_output[query_index2id[n]] = output
    return full_output
