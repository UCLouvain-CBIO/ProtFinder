# AttentionWalk

AttentionWalk code was obtained from [here](https://github.com/benedekrozemberczki/AttentionWalk). Please create an `input` and `output` directory along with `src`.

This directory provides an implementation of Attention Walk as described in the paper:

> Watch Your Step: Learning Node Embeddings via Graph Attention.
> Sami Abu-El-Haija, Bryan Perozzi, Rami Al-Rfou, Alexander A. Alemi.
> NIPS, 2018.
> [[Paper]](http://papers.nips.cc/paper/8131-watch-your-step-learning-node-embeddings-via-graph-attention)

### Steps
- The `input` directory stores sample inputs to the embedding generation model. Note that this directory must also contain the `string_bioplex.csv` which can be found [here](https://drive.google.com/file/d/1o3gvzdcqLgZ5O0alFoqtEhXL0YvXjuDr/view?usp=sharing).

- The `src` directory contains multiple source code files.
  1. `preprocess.py` converts the `string_bioplex.csv` in the right format for the attentionwalk algorithm to work. It outputs two files -- one is a .csv file which will be used to map the locations of the proteins, and the other is also the .edgelist file which will be input for the attentionwalk algorithm.
  2. `main.py` will then generate the protein embeddings from the edgelist that was generated in the previous step. This is a .csv file which is then stored in the `output` directory. Details regarding the arguments can be found in the `param_parser.py` file. Note that this file needs to be run from outside the `src` directory as `python src/main.py` with the appropriate arguments.
  3. Now, run the `embedding_csv.ipynb`. This will generate a csv with protein identifier, its embedding and its corresponding subcellular location.
  4. Use `visualize_data.ipynb` to generate the t-SNE plots for the generated embeddings.
