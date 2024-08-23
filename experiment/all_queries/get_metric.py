import os
import json
from glob import glob
from pprint import pprint
from pse.eval_util import calculate_metric

root_path = "./experiment/all_queries/output/result/*.label.json"
output = "./experiment/all_queries/output/result/all_queries.result.json"
top_k = [8, 16, 32, 64]
metric = {}
for file in glob(root_path):
    model_name = os.path.basename(file).replace(".label.json", "")
    with open(file) as f:
        search_result = json.load(f)
    metric[model_name] = calculate_metric(search_result, top_k=top_k)
pprint(metric)

with open(output) as f:
    json.dump(metric, f)
