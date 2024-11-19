# Minimal Product Search Engine for ESCI Data
This repository contains a minimal python library to perform search over [the public ESCI product search dataset](https://huggingface.co/datasets/asahi417/amazon-product-search).

## Setup
```shell
git clone ssh://git.amazon.com/pkg/SymptomQueryExperiment
cd SymptomQueryExperiment
pip install -e .
```

Flash Attention
```shell
pip install wheel
pip install flash-attn --no-build-isolation
pip install xformers
```

## Experiments
- [Baseline](https://code.amazon.com/packages/SymptomQueryExperiment/trees/mainline/--/experiment/all_queries/baseline): Use all the meta data including title and catalogue data.
- [Baseline with Expansion](https://code.amazon.com/packages/SymptomQueryExperiment/trees/mainline/--/experiment/all_queries/baseline): Baseline + Symptom Token Expansion
- [Title](https://code.amazon.com/packages/SymptomQueryExperiment/trees/mainline/--/experiment/all_queries/title): Use title only.
- [Title with Expansion](https://code.amazon.com/packages/SymptomQueryExperiment/trees/mainline/--/experiment/all_queries/title_expansion): Title + Symptom Token Expansion 
- [Meta Embedding](https://code.amazon.com/packages/SymptomQueryExperiment/trees/mainline/--/experiment/all_queries/title_expansion_meta): Meta embedding method explained at [here](https://quip-amazon.com/warbAaLnbZ6I/WIP-Generative-Query-Ingestion#temp:C:JJKdf0328445ab64a769904d2803).

To get the result with symptom queries only, see [here](https://code.amazon.com/packages/SymptomQueryExperiment/trees/mainline/--/experiment/partial_queries).

The final symptom query annotation is avalable [here](https://code.amazon.com/packages/SymptomQueryExperiment/blobs/mainline/--/experiment/partial_queries/annotation.csv).
