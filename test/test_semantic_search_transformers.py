from pse.search_semantic import SemanticSearchTransformers

# instantiate search engine
pipe = SemanticSearchTransformers(index_path="./test/test_indexing.semantic_search_transformers")
# create index
pipe.create_index(
    dataset_path="asahi417/amazon-product-search",
    dataset_name="product_detail.us",
    dataset_column_names=["product_title", "product_description", "product_bullet_point", "product_brand", "product_color"],
    dataset_id_column="product_id"
)
pipe.load_index()
result = pipe.search(
    ["shirt men", "cat toy"],
    k=2
)


from pprint import pprint
pprint(result)
