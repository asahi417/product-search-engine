from typing import List, Optional, Tuple, Dict
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset


def get_label_from_hf(dataset_path: str = "asahi417/amazon-product-search",
                      dataset_name: Optional[str] = "query_product_label.us",
                      dataset_split: Optional[str] = None,
                      dataset_query_id_name: str = "query_id",
                      dataset_label_name: str = "esci_label",
                      dataset_product_name: str = "product_id") -> Dict[str, Dict[str, str]]:
    if dataset_split:
        df = load_dataset(dataset_path, dataset_name, split=dataset_split).to_pandas()
    else:
        dataset = load_dataset(dataset_path, dataset_name)
        df = pd.concat([dataset[d].to_pandas() for d in dataset])
    label_dict = {}
    for key, g in tqdm(df.groupby(dataset_query_id_name)):
        label_dict[str(key)] = {
            r[dataset_product_name]: r[dataset_label_name] for _, r in g[[dataset_product_name, dataset_label_name]].iterrows()
        }
    return label_dict


def get_query_from_hf(dataset_path: str = "asahi417/amazon-product-search",
                      dataset_name: Optional[str] = "query_detail.us",
                      dataset_split: Optional[str] = "train",
                      dataset_query_id_name: str = "query_id",
                      dataset_query_name: str = "query") -> Dict[int, str]:
    dataset = load_dataset(dataset_path, dataset_name, split=dataset_split).to_pandas()
    return {r[dataset_query_id_name]: r[dataset_query_name] for _, r in dataset.iterrows()}


def get_corpus_from_hf(dataset_path: str = "asahi417/amazon-product-search",
                       dataset_name: Optional[str] = "product_detail.us",
                       dataset_split: Optional[str] = "train",
                       dataset_column_names: Optional[List[str]] = None,
                       dataset_id_column: Optional[str] = "product_id") -> Tuple[List[str], Dict[int, str]]:
    dataset = load_dataset(dataset_path, dataset_name, split=dataset_split)
    dataset_column_names = ["product_title", "product_description", "product_bullet_point", "product_brand", "product_color"] if dataset_column_names is None else dataset_column_names
    if any(c not in dataset.column_names for c in dataset_column_names):
        raise ValueError(f"{dataset_column_names} contains invalid columns: {dataset.column_names}")
    corpus = []
    for row in tqdm(dataset):
        corpus.append("\n\n".join([row[c] for c in dataset_column_names if row[c]]))
    if dataset_id_column:
        dataset_id = dataset[dataset_id_column]
        index2id = dict(list(enumerate(dataset_id)))
    else:
        index2id = {i: str(i) for i in range(len(corpus))}
    return corpus, index2id
