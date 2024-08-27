# ESCI Minimal Product Search Engine
This repository contains a minimal python library to perform search over [the public ESCI product search dataset](https://huggingface.co/datasets/asahi417/amazon-product-search).

## Setup
```shell
git clone https://github.com/asahi417/product-search-engine
pip install -e .
```
- Flash Attention
```shell
pip install wheel
pip install flash-attn --no-build-isolation
pip install xformers
```

## Usage
Different types of search engines are available including lexical matching and semantic matching, and all the API can be used 
in the same manner:
1. Instantiate search engine 
2. Create index
3. Load index
4. Query

Once you created the index, you can load it by specifying `index_path` when instantiating the search engine. 

### Lexical Search

```python
from pse.search_lexical import LexicalSearchBM25S
from pse.dataset_util import get_corpus_from_hf

# prepare dataset
corpus, index2id = get_corpus_from_hf()
# instantiate search engine
pipe = LexicalSearchBM25S(index_path="./lexical_search")
# create index
pipe.create_index(
    corpus=corpus,  # a list of document
    index2id=index2id  # a map from index to custom id (eg. query_id)
)
pipe.load_index()
result = pipe.search(
    ["shirt men", "cat toy"],
    k=2
)
print(result)
>>> [
    [{'id': 'B07N8BY2J1', 'score': 5.2286763191223145, 'text': 'mom tank-top 6x for water tank top hamster punk tops for men...'},
     {'id': 'B07NCH7X2X', 'score': 5.215675354003906, 'text': 'Zacatecas it Shirts for Men Black Gang surf Goodfellas tan Win Key...'}],
    [{'id': 'B08624XZ9L', 'score': 7.4331512451171875, 'text': 'Roller Cat Toy by 7 Ruby Road - Double Layer Wooden Track Balls...'},
     {'id': 'B08KVSMXQG', 'score': 7.360993385314941, 'text': 'YRCP Newest Cat Toys Boredom Busters Kitten Toy Cats Supplies...'}]
]
```

### Semantic Search
```python
from pse.search_semantic import SemanticSearchTransformers
from pse.dataset_util import get_corpus_from_hf

# prepare dataset
corpus, index2id = get_corpus_from_hf()
# instantiate search engine
pipe = SemanticSearchTransformers(index_path="./semantic_search")
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
print(result)
>>> [
    [{'id': 'B07TKMBXMM', 'score': 0.695092499256134, 'text': "Men's Stylish 3D Printed Short Sleeve T-Shirts Mens Cotton Tops...'"},
     {'id': 'B0732HVRXD', 'score': 0.6769567131996155, 'text': 'Men of the Cloth'}],
    [{'id': 'B079RTMPRF', 'score': 0.7526216506958008, 'text': 'Cat'},
     {'id': 'B082LWS7FK', 'score': 0.7500082850456238, 'text': 'Toys & Pets'}]
]
```