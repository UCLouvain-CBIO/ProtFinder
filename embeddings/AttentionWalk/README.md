Attention Walk code was obtained from [here](https://github.com/benedekrozemberczki/AttentionWalk).

This directory provides an implementation of Attention Walk as described in the paper:

> Watch Your Step: Learning Node Embeddings via Graph Attention.
> Sami Abu-El-Haija, Bryan Perozzi, Rami Al-Rfou, Alexander A. Alemi.
> NIPS, 2018.
> [[Paper]](http://papers.nips.cc/paper/8131-watch-your-step-learning-node-embeddings-via-graph-attention)

### Datasets
The code takes an input graph in a tab-separated file. Every row indicates an edge between two nodes separated by a tab. The first row is a header. Nodes should be indexed starting with 0. The right format of the processed data (which is copied to the `input` directory) is created by `src/preprocess.py`. This adds a `.edgelist` and `.csv` file to the `input` directory. The `.edgelist` file is tab-separated whereas `.csv` stores the subcellular location information which will be used to map the output embeddings with the true annotations.

### Options
Learning of the embedding is handled by the `src/main.py` script which outputs `output/trial_attentionwalk.csv`.

### Output Format Conversion
Using the generated output and the saved `.csv` file from the `input` directory, the final training and testing files are generated in `src/embedding_csv.ipynb`.
