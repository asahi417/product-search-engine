# ESCI Minimal Product Search Engine
This repository contains a minimal python library to perform search over [the public ESCI product search dataset](https://huggingface.co/datasets/asahi417/amazon-product-search).

## Setup
```shell
git clone https://github.com/asahi417/product-search-engine
pip install -e .
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

# instantiate search engine
pipe = LexicalSearchBM25S(index_path="test/test_indexing.lexical_search_bm25s")
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
    ["shirt men", "cat toy"],  # batch list of queries
    k=2  # top_k
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
