from pse.search_lexical.indexing import indexing_hf_datasets


indexing_hf_datasets(
    dataset_path="asahi417/amazon-product-search",
    dataset_name="product_detail.us",
    dataset_split="train",
    index_name="test_indexing.0"
)