import logging
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm
import numpy as np
from datasets import load_dataset


def np_save(array: np.ndarray, path: str) -> None:
    with open(path, 'wb') as f:
        np.save(f, array)


def np_load(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        return np.load(f)


def get_corpus_from_hf(dataset_path: str,
                       dataset_name: Optional[str] = None,
                       dataset_split: Optional[str] = "train",
                       dataset_column_names: Optional[List[str]] = None,
                       dataset_id_column: Optional[str] = None) -> Tuple[List[str], Dict[int, str]]:
    dataset = load_dataset(dataset_path, dataset_name, split=dataset_split)
    dataset_column_names = dataset.column_names if dataset_column_names is None else dataset_column_names
    if any(c not in dataset.column_names for c in dataset_column_names):
        raise ValueError(f"{dataset_column_names} contains invalid columns: {dataset.column_names}")
    corpus = []
    for row in tqdm(dataset):
        corpus.append("\n\n".join([row[c] for c in dataset_column_names if row[c]]))
    if dataset_id_column:
        dataset_id = dataset[dataset_id_column]
        id2index = dict(list(enumerate(dataset_id)))
    else:
        id2index = {i: str(i) for i in range(len(corpus))}
    return corpus, id2index


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    return logger
