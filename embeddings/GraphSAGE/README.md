## GraphSage: Representation Learning on Large Graphs

This directory contains code necessary to run the GraphSage algorithm. We used the [official implementation](https://github.com/williamleif/GraphSAGE) for the same.
GraphSage can be viewed as a stochastic generalization of graph convolutions, and it is especially useful for massive, dynamic graphs that contain rich feature information.
See the [paper](https://arxiv.org/pdf/1706.02216.pdf) for details on the algorithm.

### Input format
As input, at minimum the code requires that a --train_prefix option is specified which specifies the following data files:

* <train_prefix>-G.json -- A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets, respectively.
* <train_prefix>-id_map.json -- A json-stored dictionary mapping the graph node ids to consecutive integers.
* <train_prefix>-class_map.json -- A json-stored dictionary mapping the graph node ids to classes.
* <train_prefix>-feats.npy [optional] --- A numpy-stored array of node features; ordering given by id_map.json. Can be omitted and only identity features will be used.
* <train_prefix>-walks.txt [optional] --- A text file specifying random walk co-occurrences (one pair per line) (*only for unsupervised version of graphsage)

The above files are generated using `network.py`.

### Training
This is followed by training the GraphSAGE mode in the unsupervised setting by using `graphsage/unsupervised_train.py`. 
