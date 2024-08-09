from pprint import pprint
from pse.search_lexical import LexicalSearchBM25S

test_query = ["shirt men", "cat toy"]
pipe = LexicalSearchBM25S("./test_indexing.lexical_search_bm25s")
pipe.create_index(
    dataset_path="asahi417/amazon-product-search",
    dataset_name="product_detail.us",
    dataset_split="train",
)
pipe.load_index()
out = pipe.search(test_query, k=2)
pprint(out)
