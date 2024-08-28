import os
import json
from tqdm import tqdm
from pse.search_lexical import LexicalSearchBM25S
from pse.dataset_util import get_corpus_from_hf, get_query_from_hf, get_label_from_hf

expansion_file = "expansion_1"
index_path = f"./experiment/all_queries/output/cache/lexical_bm25s.title_{expansion_file}"
result_path = f"./experiment/all_queries/output/result/lexical_bm25s.title_{expansion_file}.json"
result_label_path = f"./experiment/all_queries/output/result/lexical_bm25s.title_{expansion_file}.label.json"
os.makedirs(os.path.dirname(result_path), exist_ok=True)

# load symptom tokens
with open(f"./experiment/all_queries/output/expansion/{expansion_file}.json") as f:
    expansion_dict = json.load(f)

# run experiment
if not os.path.exists(result_path):
    # search engine
    pipe = LexicalSearchBM25S(index_path=index_path)
    if not os.path.exists(index_path):
        corpus, index2id = get_corpus_from_hf(dataset_column_names=["product_title"])

        # expand document
        id2index = {v: k for k, v in index2id.items()}
        expansion_dict = {id2index[k]: v for k, v in expansion_dict.items()}
        corpus = [i if n not in expansion_dict else f"{i}\n{expansion_dict[n]}" for n, i in enumerate(corpus)]

        pipe.create_index(corpus=corpus, index2id=index2id)
    pipe.load_index()
    corpus, index2id = get_query_from_hf()
    result = pipe.search(corpus, index2id=index2id, k=64)
    with open(result_path, "w") as f:
        json.dump(result, f)
with open(result_path) as f:
    search_result = json.load(f)

# compute metric
labels = get_label_from_hf()
labeled_search = {}
for k, v in tqdm(search_result.items()):
    labeled_search[k] = []
    for rank, hit in enumerate(v):
        if hit["id"] in labels[k]:
            labeled_search[k].append({"id": hit["id"], "label": labels[k][hit["id"]], "ranking": rank + 1, "score": hit["score"]})
        else:
            labeled_search[k].append({"id": hit["id"], "label": "None", "ranking": rank + 1, "score": hit["score"]})
    for product_id, label in labels[k].items():
        if product_id not in labeled_search[k]:
            labeled_search[k].append({"id": product_id, "label": label, "ranking": -100, "score": -100})
with open(result_label_path, "w") as f:
    json.dump(labeled_search, f)



