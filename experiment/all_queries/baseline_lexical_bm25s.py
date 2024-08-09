from datasets import load_dataset
from pse.search_lexical import LexicalSearchBM25S

load_dataset("asahi417/amazon-product-search", "query_product_label.us")

# instantiate search engine
pipe = LexicalSearchBM25S(index_path="./experiment/output/index/lexical_bm25s")
# create index
pipe.create_index(
    dataset_path="asahi417/amazon-product-search",
    dataset_name="product_detail.us",
    dataset_column_names=["product_title", "product_description", "product_bullet_point", "product_brand", "product_color"],
    dataset_id_column="product_id"
)
# load index
pipe.load_index()
# search
result = pipe.search(
    ["shirt men", "cat toy"],  # list of query
    k=64
)
