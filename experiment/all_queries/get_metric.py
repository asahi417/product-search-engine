import os
import json
from glob import glob
from pse.eval_util import calculate_metric

root_path = "./experiment/all_queries/output/result/*.label.json"
top_k = [8, 16, 32, 64]
for file in glob(root_path):
    model_name = os.path.basename(file).replace(".label.json", "")
    with open(file) as f:
        search_result = json.load(f)
    metric = calculate_metric(search_result, top_k=top_k)
