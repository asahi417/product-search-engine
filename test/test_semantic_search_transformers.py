from pse.search_semantic import SemanticSearchTransformers
from pse.dataset_util import get_corpus_from_hf

# prepare dataset
corpus, index2id = get_corpus_from_hf()
# instantiate search engine
pipe = SemanticSearchTransformers(index_path="./test/test_indexing.semantic_search_transformers")
# create index
pipe.create_index(
    corpus=corpus,  # a list of document
    index2id=index2id,  # a map from index to custom id (eg. query_id)
    batch_size=2048
)
pipe.load_index()
result = pipe.search(
    ["shirt men", "cat toy"],
    k=2
)

from pprint import pprint
pprint(result)
