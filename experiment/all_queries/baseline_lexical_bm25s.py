import os
import json
from tqdm import tqdm
from pse.search_lexical import LexicalSearchBM25S
from pse.dataset_util import get_corpus_from_hf, get_query_from_hf, get_label_from_hf

index_path = "./experiment/all_queries/output/cache/lexical_bm25s"
result_path = "./experiment/all_queries/output/result/lexical_bm25s.json"
result_label_path = "./experiment/all_queries/output/result/lexical_bm25s.label.json"
os.makedirs(os.path.dirname(result_path), exist_ok=True)

# run experiment
if not os.path.exists(result_path):
    # search engine
    pipe = LexicalSearchBM25S(index_path=index_path)
    if not os.path.exists(index_path):
        corpus, index2id = get_corpus_from_hf()
        pipe.create_index(corpus=corpus, index2id=index2id)
    pipe.load_index()
    corpus, index2id = get_query_from_hf()
    result = pipe.search(corpus, index2id=index2id, k=64)
    with open(result_path, "w") as f:
        json.dump(result, f)
with open(result_path) as f:
    search_result = json.load(f)

# compute metric
if not os.path.exists(result_label_path):
    labels = get_label_from_hf()
    labeled_search = {}
    for k, v in tqdm(search_result.items()):
        labeled_search[k] = []
        for rank, hit in enumerate(v):
            if hit["id"] in labels[k]:
                labeled_search[k].append({"id": hit["id"], "label": labels[k][hit["id"]], "ranking": rank + 1})
        for product_id, label in labels[k].items():
            if product_id not in labeled_search[k]:
                labeled_search[k].append({"id": product_id, "label": label, "ranking": -100})
    with open(result_label_path, "w") as f:
        json.dump(labeled_search, f)
with open(result_label_path) as f:
    labeled_search = json.load(f)



