# ProtFinder

This is the official implementation of 

> *ProtFinder: finding subcellular locations of proteins using 
> protein interaction networks*. Aayush Grover, Laurent Gatto.
> bioRxiv: 2022.01.11.475836; doi: https://doi.org/10.1101/2022.01.11.475836

Protein subcellular localization prediction plays a crucial role in improving our understanding of different diseases and consequently assists in building drug targeting and drug development pipelines. Proteins are known to co-exist at multiple subcellular locations which make the task of prediction extremely challenging. A protein interaction network is a graph that captures interactions between different proteins. It is safe to assume that if two proteins are interacting, they must share some subcellular locations. With this regard, we propose ProtFinder - the first deep learning-based model that exclusively relies on protein interaction networks to predict the multiple subcellular locations of proteins. We also integrate biological priors like the cellular component of Gene Ontology to make ProtFinder a more biology-aware intelligent system. ProtFinder is trained and tested using the STRING and BioPlex databases whereas the annotations of proteins are obtained from the Human Protein Atlas. Our model obtained an AUC-ROC score of 90.00% and an MCC score of 83.42% on a held-out set of proteins. We also apply ProtFinder to annotate proteins that currently do not have confident location annotations. We observe that ProtFinder is able to confirm some of these unreliable location annotations, while in some cases complementing the existing databases with novel location annotations.

The repository is structured such that each directory has its own set of instructions.

### custom_data
This directory contains all the data acquisition, integration and processing steps.

### embeddings
This directory contains different source code and instructions to run different graph embedding generation strategies.

### node2loc
This directory contains the code used to reproduce the [node2loc algorithm](https://ieeexplore.ieee.org/document/9431661/). Node2loc also uses protein interaction networks to predict subcellular localization of proteins (one location per protein). However, proteins are known to co-exist and hence, we build upon node2loc to build a multi-label classifier for subcellular locations of proteins.

### protfinder
This directory contains the source code and instructions to run our ProtFinder model. It also contains scripts to draw inferences and visualize the results of ProtFinder.
