import json
import pandas as pd

df = pd.read_csv('./experiment/partial_queries/annotation.csv')
annotators = ["annotator 1", "annotator 2", "annotator 3"]


def flag(row):
    if any(row[x] == 0 for x in annotators):
        return False
    return True

data = {i.to_dict()["id"]: i.to_dict()["query"] for _, i in df.iterrows() if flag(i)}
with open("./experiment/partial_queries/query_set_1.json", "w") as f:
    json.dump(data, f)
