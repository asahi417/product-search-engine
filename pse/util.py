import os
import logging
import json
from gc import collect
from glob import glob
from typing import List, Dict, Union, Optional

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
        index_expansion_path: Optional[str] = None,
        inbdex_meta_embedding_path: Optional[str] = None,
        k: int = 16,
        query_chunk_size: int = 400,
        corpus_chunk_size: int = 400000) -> Dict[str, List[Dict[str, Union[str, float]]]]:
    def load_index(output_dir) -> (Dict[int, str], List[str], torch.Tensor):
        with open(f"{output_dir}/index2id.json") as f:
            index2id = {int(k): v for k, v in json.load(f).items()}
        with open(f"{output_dir}/corpus.json") as f:
            corpus = json.load(f)["corpus"]
        numpy_files = []
        flags = []
        for numpy_file in glob(f"{output_dir}/embedding.*.npy"):
            start, end = numpy_file.split("/embedding.")[-1].replace(".npy", "").split("-")
            start, end = int(start), int(end)
            assert start not in flags and end - 1 not in flags, f"{start}, {end - 1}"
            flags += list(range(start, end))
            numpy_files.append([start, numpy_file])
        numpy_files = sorted(numpy_files, key=lambda x: x[0])
        embedding = []
        for _, numpy_file in numpy_files:
            embedding.append(np_load(numpy_file))
        embedding = torch.as_tensor(np.concatenate(embedding))
        assert len(embedding) == len(index2id) == len(corpus), f"{len(embedding)}, {len(index2id)}, {len(corpus)}"
        return index2id, corpus, embedding

    logger = get_logger(__name__)
    query_index2id, query_corpus, query_embedding = load_index(query_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    logger.info(f"load query: {query_embedding.shape}")
    index_index2id, index_corpus, index_embedding = load_index(index_path)
    logger.info(f"load document: {index_embedding.shape}")
    if index_expansion_path:
        assert inbdex_meta_embedding_path
        if not os.path.exists(f"{inbdex_meta_embedding_path}/embedding.npy"):
            index_expansion_index2id, index_expansion_corpus, index_expansion_embedding = load_index(index_expansion_path)
            logger.info(f"load document: {index_expansion_embedding.shape}")
            index_expansion_id2index = {v: k for k, v in index_expansion_index2id.items()}
            index_id2index = {v: k for k, v in index_index2id.items()}
            index_group = {}
            for k, v in index_expansion_index2id.items():
                corpus_id, _ = v.split(".")
                if index_id2index[corpus_id] not in index_group:
                    index_group[index_id2index[corpus_id]] = [index_expansion_id2index[v]]
                else:
                    index_group[index_id2index[corpus_id]] += [index_expansion_id2index[v]]

            index_meta_embedding = []
            for index_key, expansion_keys in index_group.items():
                v_index = index_embedding[index_key].unsqueeze(0).to(device)
                v_expansions = torch.stack([index_expansion_embedding[i] for i in expansion_keys]).to(device)
                weight = torch.nn.functional.softmax(torch.inner(v_index, v_expansions))
                v_expansion = torch.mm(weight, v_expansions)
                v_meta = (v_index + v_expansion)/2
                index_meta_embedding.append(v_meta.cpu().numpy())
            index_embedding = np.concatenate(index_meta_embedding)

            # save
            os.makedirs(inbdex_meta_embedding_path, exist_ok=True)
            with open(f"{inbdex_meta_embedding_path}/corpus.json", "w") as f:
                json.dump({"corpus": index_corpus}, f)
            with open(f"{inbdex_meta_embedding_path}/index2id.json", "w") as f:
                json.dump(index_index2id, f)
            np_save(index_embedding, f"{inbdex_meta_embedding_path}/embedding.npy")
        index_embedding = torch.as_tensor(np_load(f"{inbdex_meta_embedding_path}/embedding.npy"))
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

