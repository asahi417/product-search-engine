import os
import json
from glob import glob
from pprint import pprint
import pandas as pd
from pse.eval_util import calculate_metric

root_path = "./experiment/all_queries/output/result/*.label.json"
output = "./experiment/all_queries/output/result/partial_queries.result.json"
top_k = [8, 16, 32, 64]


with open("./experiment/partial_queries/query_set_1.json") as f:
    query_set = list(json.load(f).keys())

metric = {}
if os.path.exists(output):
    with open(output, "r") as f:
        metric.update(json.load(f))

for file in glob(root_path):
    model_name = os.path.basename(file).replace(".label.json", "")
    if model_name not in metric:
        with open(file) as f:
            search_result = json.load(f)
        search_result = {i: search_result[i] for i in query_set}
        metric[model_name] = calculate_metric(search_result, top_k=top_k)
pprint(metric)
with open(output, "w") as f:
    json.dump(metric, f)


def clean_name(name: str) -> str:
    if "lexical" in name:
        return name.replace("lexical_bm25s", "bm25").replace(".", "/")
    name = name.replace("semantic_transformers.", "")
    if "title" in name or "expansion" in name:
        mode = name.split(".")[0].replace("baseline_", "")
        name = ".".join(name.split(".")[1:])
        return f"{name}/{mode}"
    return name


df = (pd.DataFrame(metric) * 100).round(1)
df.columns = [clean_name(i) for i in df.columns]
df = df[sorted(df.columns)].T
print(df.to_markdown())
print(df)
