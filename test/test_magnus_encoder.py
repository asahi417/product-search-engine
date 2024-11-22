from pse.search_semantic import SemanticSearchTransformers

pipe = SemanticSearchTransformers(
    index_path="./test/test_indexing.semantic_search_transformers.index",
    query_path="./test/test_indexing.semantic_search_transformers.query"
)
documents = [
    "Bath salt, lavender",
    "beef stake",
    "cycling"
]
queries = [
    "bath",
    "pork stake",
    "bike"
]
pipe.encode_document(corpus=documents)
pipe.encode_query(corpus=queries)
result = pipe.search(k=2)
print(result)
"""
{'0': [{'id': '0', 'score': 0.5967979431152344, 'text': 'Bath salt, lavender'},
       {'id': '2', 'score': 0.1962255835533142, 'text': 'cycling'}],
 '1': [{'id': '1', 'score': 0.7114149332046509, 'text': 'beef stake'},
       {'id': '0', 'score': 0.1657523512840271, 'text': 'Bath salt, lavender'}],
 '2': [{'id': '2', 'score': 0.742513120174408, 'text': 'cycling'},
       {'id': '1', 'score': 0.09248663485050201, 'text': 'beef stake'}]}
"""
