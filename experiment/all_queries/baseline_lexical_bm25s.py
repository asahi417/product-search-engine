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
    query = get_query_from_hf(dataset_split="train")
    result = pipe.search(list(query.values()), k=64)
    with open(result_path, "w") as f:
        json.dump({q: r for q, r in zip(query.keys(), result)}, f)
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

# TODO: Implement metric calculation
# `labeled_search` is a dictionary where the key is query id and the value is a list of "id", "label", "ranking", eg)
# [{'id': 'B00XBZFWWM', 'label': 'E', 'ranking': 5},
#  {'id': 'B07X3Y6B1V', 'label': 'E', 'ranking': 10},
#  {'id': 'B06W2LB17J', 'label': 'E', 'ranking': 11},
#  {'id': 'B075ZBF9HG', 'label': 'E', 'ranking': 14},
#  {'id': 'B01N5Y6002', 'label': 'E', 'ranking': 22},
#  {'id': 'B07JY1PQNT', 'label': 'E', 'ranking': 25},
#  {'id': 'B001E6DMKY', 'label': 'E', 'ranking': 26},
#  {'id': 'B07QJ7WYFQ', 'label': 'E', 'ranking': 41},
#  {'id': 'B07WDM7MQQ', 'label': 'E', 'ranking': 45},
#  {'id': 'B003O0MNGC', 'label': 'E', 'ranking': 59},
#  {'id': 'B07RH6Z8KW', 'label': 'E', 'ranking': 60},
#  {'id': 'B000MOO21W', 'label': 'I', 'ranking': -100},
#  {'id': 'B07X3Y6B1V', 'label': 'E', 'ranking': -100},
#  {'id': 'B07WDM7MQQ', 'label': 'E', 'ranking': -100},
#  {'id': 'B07RH6Z8KW', 'label': 'E', 'ranking': -100},
#  {'id': 'B07QJ7WYFQ', 'label': 'E', 'ranking': -100},
#  {'id': 'B076Q7V5WX', 'label': 'E', 'ranking': -100},
#  {'id': 'B075ZBF9HG', 'label': 'E', 'ranking': -100},
#  {'id': 'B06W2LB17J', 'label': 'E', 'ranking': -100},
#  {'id': 'B07JY1PQNT', 'label': 'E', 'ranking': -100},
#  {'id': 'B01MZIK0PI', 'label': 'E', 'ranking': -100},
#  {'id': 'B011RX6PNO', 'label': 'I', 'ranking': -100},
#  {'id': 'B00XBZFWWM', 'label': 'E', 'ranking': -100},
#  {'id': 'B00MARNO5Y', 'label': 'E', 'ranking': -100},
#  {'id': 'B003O0MNGC', 'label': 'E', 'ranking': -100},
#  {'id': 'B001E6DMKY', 'label': 'E', 'ranking': -100},
#  {'id': 'B01N5Y6002', 'label': 'E', 'ranking': -100}]
# Here, ranking=-100 means the product is not in the top-64 search result.


