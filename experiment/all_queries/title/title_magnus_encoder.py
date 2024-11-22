# https://huggingface.co/spaces/mteb/leaderboard
import os
import json
from tqdm import tqdm
from pse.search_semantic import SemanticSearchTransformers
from pse.util import get_semantic_search_result
from pse.dataset_util import get_corpus_from_hf, get_label_from_hf

model = "magnus_encoder"
batch_size_index = 512
model_kwargs = None
prompt_name_index = None
prompt_prefix_index = None
prompt_suffix_index = None

# config
index_path = f"./experiment/all_queries/output/cache/magnus_encoder.title.index"
query_path = f"./experiment/all_queries/output/cache/magnus_encoder.query"
result_path = f"./experiment/all_queries/output/result/magnus_encoder.title.json"
result_label_path = f"./experiment/all_queries/output/result/magnus_encoder.title.label.json"
os.makedirs(os.path.dirname(result_path), exist_ok=True)

# run experiment
if not os.path.exists(result_path):
    # search engine
    pipe = SemanticSearchTransformers(
        index_path=index_path,
        query_path=query_path,
        model=model,
        model_kwargs=model_kwargs,
        index_chunk=batch_size_index * 20,
    )
    corpus, index2id = get_corpus_from_hf(dataset_column_names=["product_title"])
    pipe.encode_document(
        corpus=corpus,
        index2id=index2id,
        batch_size=batch_size_index,
        prompt_name=prompt_name_index,
        prompt_prefix=prompt_prefix_index,
        prompt_suffix=prompt_suffix_index
    )
    result = get_semantic_search_result(
        index_path=index_path,
        query_path=query_path,
        k=64
    )
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
            labeled_search[k].append({"id": product_id, "label": label, "ranking": -100, "score": 0})
with open(result_label_path, "w") as f:
    json.dump(labeled_search, f)
