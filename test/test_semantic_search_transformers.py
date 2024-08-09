from pprint import pprint
from pse.search_semantic import SemanticSearchTransformers

test_query = ["shirt men", "cat toy"]
pipe = SemanticSearchTransformers("./test/test_indexing.semantic_search_transformers")
pipe.create_index(
    dataset_path="asahi417/amazon-product-search",
    dataset_name="product_detail.us",
    dataset_split="train",
)
pipe.load_index()
out = pipe.search(test_query, k=2)
pprint(out)
