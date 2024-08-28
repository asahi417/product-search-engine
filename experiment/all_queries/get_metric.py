import os
import json
from glob import glob
from pprint import pprint
import pandas as pd
from pse.eval_util import calculate_metric

root_path = "./experiment/all_queries/output/result/*.label.json"
output = "./experiment/all_queries/output/result/all_queries.result.json"
top_k = [8, 16, 32, 64]
metric = {}
if os.path.exists(output):
    with open(output, "r") as f:
        metric.update(json.load(f))

for file in glob(root_path):
    model_name = os.path.basename(file).replace(".label.json", "")
    if model_name not in metric:
        with open(file) as f:
            search_result = json.load(f)
        metric[model_name] = calculate_metric(search_result, top_k=top_k)
pprint(metric)

with open(output, "w") as f:
    json.dump(metric, f)
df = (pd.DataFrame(metric) * 100).round(1)
df.columns = [i.replace("lexical_", "").replace("semantic_", "") for i in df.columns]
df = df[sorted(df.columns)].T
print(df.to_markdown())
