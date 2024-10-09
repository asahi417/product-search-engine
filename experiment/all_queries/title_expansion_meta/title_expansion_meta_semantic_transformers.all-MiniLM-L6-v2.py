# https://huggingface.co/spaces/mteb/leaderboard
import os
import json
from tqdm import tqdm
from itertools import chain

from pse.search_semantic import SemanticSearchTransformers
from pse.util import get_semantic_search_result
from pse.dataset_util import get_corpus_from_hf, get_query_from_hf, get_label_from_hf

model = "sentence-transformers/all-MiniLM-L6-v2"
batch_size_index_expansion = 4096
model_kwargs = None
prompt_name_index_expansion = None
prompt_prefix_index_expansion = None
prompt_suffix_index_expansion = None

# config
expansion_file = "expansion_1"
index_expansion_path = f"./experiment/all_queries/output/cache/semantic_transformers.{expansion_file}.{os.path.basename(model)}.index"
index_meta_embedding_path = f"./experiment/all_queries/output/cache/semantic_transformers.title_{expansion_file}_meta.{os.path.basename(model)}.index"
index_path = f"./experiment/all_queries/output/cache/semantic_transformers.title.{os.path.basename(model)}.index"
query_path = f"./experiment/all_queries/output/cache/semantic_transformers.{os.path.basename(model)}.query"
result_path = f"./experiment/all_queries/output/result/semantic_transformers.title_{expansion_file}_meta.{os.path.basename(model)}.json"
result_label_path = f"./experiment/all_queries/output/result/semantic_transformers.title_{expansion_file}_meta.{os.path.basename(model)}.label.json"
os.makedirs(os.path.dirname(result_path), exist_ok=True)

# load symptom tokens
with open(f"./experiment/all_queries/output/expansion/{expansion_file}.json") as f:
    expansion_dict = json.load(f)
    corpus = [i.split(", ") for i in expansion_dict.values()]
    ids = list(chain(*[[f"{x}.{n}" for n in range(len(y))] for x, y in zip(expansion_dict.keys(), corpus)]))
    corpus = list(chain(*corpus))
    index2id = {n: str(i) for n, i in enumerate(ids)}

# run experiment
if not os.path.exists(result_path):
    # search engine
    pipe = SemanticSearchTransformers(
        index_path=index_path,
        query_path=query_path,
        index_expansion_path=index_expansion_path,
        model=model,
        model_kwargs=model_kwargs,
        index_expansion_chunk=batch_size_index_expansion * 20
    )
    pipe.encode_expansion(
        corpus=corpus,
        index2id=index2id,
        batch_size=batch_size_index_expansion,
        prompt_name=prompt_name_index_expansion,
        prompt_prefix=prompt_prefix_index_expansion,
        prompt_suffix=prompt_suffix_index_expansion
    )
    result = get_semantic_search_result(
        index_path=index_path,
        query_path=query_path,
        index_expansion_path=index_expansion_path,
        index_meta_embedding_path=index_meta_embedding_path,
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
