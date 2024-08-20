# https://huggingface.co/spaces/mteb/leaderboard
import os
import json
import torch
from tqdm import tqdm
from pse.search_semantic import SemanticSearchTransformers
from pse.dataset_util import get_corpus_from_hf, get_query_from_hf, get_label_from_hf

# default config
batch_size = 1
prompt_name_index = None
prompt_prefix_index = None
prompt_suffix_index = None
prompt_name_query = None
prompt_prefix_query = None
prompt_suffix_query = None
model_kwargs = None

# MODEL: BAAI/bge-en-icl
# model = "BAAI/bge-en-icl"
# prompt_prefix_query = """<instruct>Given a web search query, retrieve relevant passages that answer the query.\n<query>what
# is a virtual interface\n<response>A virtual interface is a software-defined abstraction that mimics the behavior and
# characteristics of a physical network interface. It allows multiple logical network connections to share the same
# physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used
# in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring
# dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security
# and management purposes.\n\n<instruct>Given a web search query, retrieve relevant passages that answer the query.
# \n<query>causes of back pain in female for a week\n<response>Back pain in females lasting a week can stem from various
# factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like
# herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic
# inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may
# exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management.
# \n\n<instruct> Given a web search query, retrieve relevant passages that answer the query.\n<query> """
# prompt_suffix_query = "\n<response>"
# model_kwargs = {"device_map": "balanced", "torch_dtype": torch.float16}

# MODE: SFR-Embedding-2_R
# model = "Salesforce/SFR-Embedding-2_R"
# prompt_prefix_query = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
# model_kwargs = {"device_map": "balanced", "torch_dtype": torch.bfloat16}

# MODEL: stella_en_1.5B_v5
# model = "dunzhang/stella_en_1.5B_v5"
# batch_size = 32
# prompt_name_query = "s2p_query"

# MODEL: stella_en_400M_v5
# model = "dunzhang/stella_en_400M_v5"
# batch_size = 128
# prompt_name_query = "s2p_query"

# MODEL: gte-Qwen2-1.5B-instruct
# model = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
# batch_size = 32
# prompt_name_query = "query"

# MODEL: gte-large-en-v1.5
model = "Alibaba-NLP/gte-large-en-v1.5"
batch_size = 128

# config
index_path = f"./experiment/all_queries/output/cache/semantic_transformers.{os.path.basename(model)}.index"
query_path = f"./experiment/all_queries/output/cache/semantic_transformers.{os.path.basename(model)}.query"
result_path = f"./experiment/all_queries/output/result/semantic_transformers.{os.path.basename(model)}.json"
result_label_path = f"./experiment/all_queries/output/result/semantic_transformers.{os.path.basename(model)}.label.json"
os.makedirs(os.path.dirname(result_path), exist_ok=True)

# run experiment
if not os.path.exists(result_path):
    # search engine
    pipe = SemanticSearchTransformers(
        index_path=index_path,
        query_path=index_path,
        model=model,
        model_kwargs=model_kwargs
    )
    corpus, index2id = get_corpus_from_hf()
    pipe.encode_document(
        corpus=corpus,
        index2id=index2id,
        batch_size=batch_size,
        prompt_name=prompt_name_index,
        prompt_prefix=prompt_prefix_index,
        prompt_suffix=prompt_suffix_index
    )
    corpus, index2id = get_query_from_hf()
    pipe.encode_query(
        corpus=corpus,
        index2id=index2id,
        batch_size=batch_size,
        prompt_name=prompt_name_index,
        prompt_prefix=prompt_prefix_index,
        prompt_suffix=prompt_suffix_index
    )
    result = pipe.search(k=64)
    with open(result_path, "w") as f:
        json.dump(result, f)


# with open(result_path) as f:
#     search_result = json.load(f)
#
# # compute metric
# if not os.path.exists(result_label_path):
#     labels = get_label_from_hf()
#     labeled_search = {}
#     for k, v in tqdm(search_result.items()):
#         labeled_search[k] = []
#         for rank, hit in enumerate(v):
#             if hit["id"] in labels[k]:
#                 labeled_search[k].append({"id": hit["id"], "label": labels[k][hit["id"]], "ranking": rank + 1})
#         for product_id, label in labels[k].items():
#             if product_id not in labeled_search[k]:
#                 labeled_search[k].append({"id": product_id, "label": label, "ranking": -100})
#     with open(result_label_path, "w") as f:
#         json.dump(labeled_search, f)
# with open(result_label_path) as f:
#     labeled_search = json.load(f)
