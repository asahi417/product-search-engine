from pse.search_lexical import LexicalSearchBM25S

# prepare dataset
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
pipe = LexicalSearchBM25S(index_path="./test/test_indexing.semantic_search_transformers")
pipe.create_index(corpus=documents)
pipe.load_index()
result = pipe.search(queries, k=2)
print(result)
"""
{'0': [{'id': '0', 'score': 0.3202707767486572, 'text': 'Bath salt, lavender'},                                                                                                                                                                                                                                                                                                              
       {'id': '2', 'score': 0.0, 'text': 'cycling'}],
 '1': [{'id': '1', 'score': 0.39233168959617615, 'text': 'beef stake'},
       {'id': '2', 'score': 0.0, 'text': 'cycling'}],
 '2': [{'id': '2', 'score': 0.0, 'text': 'cycling'},
       {'id': '1', 'score': 0.0, 'text': 'beef stake'}]}
"""
