This `embeddings` directory contains source codes and figures relating to two different embedding-generation methods that we tried -- [GraphSAGE](https://arxiv.org/pdf/1706.02216.pdf), [Node2Vec](https://snap.stanford.edu/node2vec/) and [AttentionWalk](http://papers.nips.cc/paper/8131-watch-your-step-learning-node-embeddings-via-graph-attention). 

Some minor changes will be needed to be made in the code for generating embeddings for independent STRING or BioPlex data (like file path, column name etc).

# Node2Vec

- The `data` directory stores sample inputs and outputs to the embedding generation model. Note that this directory must also contain the `string_bioplex.csv` which can be found [here](https://drive.google.com/file/d/1uVRoAZFNjormaa496YLwd3nYpoiwArbv/view?usp=sharing).

- The `src` directory contains multiple source code files.
  1. `preprocess.py` converts the `string_bioplex.csv` in the right format for the node2vec algorithm to work. It outputs two files -- one is a .csv file which will be used to map the locations of the proteins, and the other is the .edgelist file which will be input for the node2vec algorithm.
  2. `node2vec_embeddings.py` will then generate the protein embeddings from the edgelist that was generated in the previous step. This is a .emb file which is then stored in the `data` directory. Details regarding the arguments can be found in the file.
  3. Before running another code, please manually remove line 1 of the .emb file. It will contain 2 integers -- number of proteins and embedding dimension. Please remove that line.
  4. Now, run the `embedding_csv.ipynb`. This will generate a csv with protein identifier, its embedding and its corresponding subcellular location.
  5. Use `visualize_data.ipynb` to generate the t-SNE plots for the generated embeddings.
 
