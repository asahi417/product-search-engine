from statistics import mean
from typing import List, Optional, Dict, Any, Tuple
from sklearn.metrics import ndcg_score


def convert_search_result(search_result: Dict[str, List[Dict[str, Any]]]):
    query_id = []
    y_true = []
    y_score = []
    y_rank = []
    label_map = {"E": 3, "S": 2, "C": 1, "I": 0, "None": 0}
    for k, v in search_result.items():
        query_id.append(k)
        y_true.append([label_map[i["label"]] for i in v])
        y_score.append([i["score"] for i in v])
        y_rank.append([i["ranking"] for i in v])
    return y_true, y_score, y_rank, query_id


def binary_f1_score(y_true: List[List[int]], y_pred: List[List[float]], top_k: Optional[int] = None) -> Tuple[float, float, float]:
    full_precision, full_recall, full_f1 = [], [], []
    for label, prediction in zip(y_true, y_pred):
        label = list(zip(*sorted(zip(label, prediction), key=lambda x: x[1], reverse=True)))
        binary_label = [int(i > 0) for i in label]
        binary_label_top_k = binary_label[:top_k]
        precision = sum(binary_label_top_k)/len(binary_label_top_k)
        recall = sum(binary_label_top_k)/sum(binary_label)
        f1 = 2 * precision * recall / (precision + recall)
        full_precision.append(precision)
        full_recall.append(recall)
        full_f1.append(f1)
    return mean(full_precision), mean(full_recall), mean(full_f1)


def calculate_metric(search_result: Dict[str, List[Dict[str, Any]]], top_k: Optional[List[int]] = None) -> Dict[str, float]:
    y_true, y_score, y_rank, query_id = convert_search_result(search_result)
    top_k = [None] if top_k is None else top_k
    metric = {}
    for k in top_k:
        metric[f"ndcg@{k}"] = ndcg_score(y_true, y_score, k=top_k)
        precision, recall, f1 = binary_f1_score(y_true, y_score, top_k=k)
        metric[f"precision@{k}"] = precision
        metric[f"recall@{k}"] = recall
        metric[f"f1@{k}"] = f1
    return metric


