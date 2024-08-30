# TODO
# https://huggingface.co/spaces/mteb/leaderboard
import os
import json
from tqdm import tqdm
import torch
from pse.search_semantic import SemanticSearchTransformers
from pse.util import get_semantic_search_result
from pse.dataset_util import get_corpus_from_hf, get_query_from_hf, get_label_from_hf

model = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
batch_size_query = 256
batch_size_index = 4 * 2
prompt_name_query = "query"
model_kwargs = None
prompt_name_index = None
prompt_prefix_index = None
prompt_suffix_index = None
prompt_prefix_query = None
prompt_suffix_query = None

# config
expansion_file = "expansion_1"
index_path = f"./experiment/all_queries/output/cache/semantic_transformers.title_{expansion_file}.{os.path.basename(model)}.index"
query_path = f"./experiment/all_queries/output/cache/semantic_transformers.{os.path.basename(model)}.query"
result_path = f"./experiment/all_queries/output/result/semantic_transformers.title_{expansion_file}.{os.path.basename(model)}.json"
result_label_path = f"./experiment/all_queries/output/result/semantic_transformers.title_{expansion_file}.{os.path.basename(model)}.label.json"
os.makedirs(os.path.dirname(result_path), exist_ok=True)

# load symptom tokens
with open(f"./experiment/all_queries/output/expansion/{expansion_file}.json") as f:
    expansion_dict = json.load(f)


# run experiment
if not os.path.exists(result_path):
    # search engine
    pipe = SemanticSearchTransformers(
        index_path=index_path,
        query_path=query_path,
        model=model,
        model_kwargs=model_kwargs,
        index_chunk=batch_size_index * 20,
        query_chunk=batch_size_query * 20
    )
    corpus, index2id = get_corpus_from_hf(dataset_column_names=["product_title"])

    # expand document
    id2index = {v: k for k, v in index2id.items()}
    expansion_dict = {id2index[k]: v for k, v in expansion_dict.items()}
    corpus = [i if n not in expansion_dict else f"{i}\n{expansion_dict[n]}" for n, i in enumerate(corpus)]

    pipe.encode_document(
        corpus=corpus,
        index2id=index2id,
        batch_size=batch_size_index,
        prompt_name=prompt_name_index,
        prompt_prefix=prompt_prefix_index,
        prompt_suffix=prompt_suffix_index
    )
    # corpus, index2id = get_query_from_hf()
    # pipe.encode_query(
    #     corpus=corpus,
    #     index2id=index2id,
    #     batch_size=batch_size_query,
    #     prompt_name=prompt_name_query,
    #     prompt_prefix=prompt_prefix_query,
    #     prompt_suffix=prompt_suffix_query
    # )
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
            labeled_search[k].append({"id": product_id, "label": label, "ranking": -100, "score": -100})
with open(result_label_path, "w") as f:
    json.dump(labeled_search, f)
