# https://huggingface.co/spaces/mteb/leaderboard
import os
import json
import torch
from tqdm import tqdm
from pse.search_semantic import SemanticSearchTransformers
from pse.dataset_util import get_corpus_from_hf, get_query_from_hf, get_label_from_hf

# default config
prompt_name_index = None
prompt_prefix_index = None
prompt_suffix_index = None
prompt_name_query = "s2p_query"
prompt_prefix_query = None
prompt_suffix_query = None
model_kwargs = {"device_map": "balanced", "torch_dtype": torch.float16}


# MODEL: BAAI/bge-en-icl
model = "BAAI/bge-en-icl"
batch_size = 1
prompt_prefix_query = """<instruct>Given a web search query, retrieve relevant passages that answer the query.\n<query>what 
is a virtual interface\n<response>A virtual interface is a software-defined abstraction that mimics the behavior and 
characteristics of a physical network interface. It allows multiple logical network connections to share the same
physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used 
in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring
dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security 
and management purposes.\n\n<instruct>Given a web search query, retrieve relevant passages that answer the query.
\n<query>causes of back pain in female for a week\n<response>Back pain in females lasting a week can stem from various 
factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like 
herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic 
inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may 
exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management.
\n\n<instruct> Given a web search query, retrieve relevant passages that answer the query.\n<query> """
prompt_suffix_query = "\n<response>"

# model = "Salesforce/SFR-Embedding-2_R"
# batch_size = 1


# model = "Alibaba-NLP/gte-Qwen2-7B-instruct"
# batch_size = 1
# prompt_name_index =
# prompt_index =
# prompt_name_query = "s2p_query"
# prompt_query =
# model_kwargs = {"device_map": "balanced", "torch_dtype": torch.float16}

# MODEL: stella_en_1.5B_v5
model = "dunzhang/stella_en_1.5B_v5"
batch_size = 32
prompt_name_query = "s2p_query"

# config
index_path = f"./experiment/all_queries/output/cache/semantic_transformers.{os.path.basename(model)}"
result_path = f"./experiment/all_queries/output/result/semantic_transformers.{os.path.basename(model)}.json"
result_label_path = f"./experiment/all_queries/output/result/semantic_transformers.{os.path.basename(model)}.label.json"
os.makedirs(os.path.dirname(result_path), exist_ok=True)

# run experiment
if not os.path.exists(result_path):
    # search engine
    pipe = SemanticSearchTransformers(index_path=index_path, model=model, model_kwargs=model_kwargs)
    if not os.path.exists(index_path):
        corpus, index2id = get_corpus_from_hf()
        pipe.create_index(
            corpus=corpus,
            index2id=index2id,
            batch_size=batch_size,
            prompt_name=prompt_name_index,
            prompt_prefix=prompt_prefix_index,
            prompt_suffix=prompt_suffix_index
        )
    pipe.load_index()
    query = get_query_from_hf()
    result = pipe.search(
        list(query.values()),
        k=64,
        batch_size=batch_size,
        prompt_name=prompt_name_query,
        prompt_prefix=prompt_prefix_query,
        prompt_suffix=prompt_suffix_query
    )
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



