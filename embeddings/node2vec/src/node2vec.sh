#!/bin/bash

python ./node2vec_embeddings.py --input ../data/input_all.edgelist --output ../data/all.emb --weighted --workers 1
