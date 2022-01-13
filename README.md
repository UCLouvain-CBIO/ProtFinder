# Aayush internship (summer 2020)


> *Predicting protein subcellular location using learned distributed
> representations from a protein-protein network*. Xiaoyong Pan, Lei
> Chen, Min Liu, Tao Huang, Yu-Dong Cai. bioRxiv 768739; doi:
> https://doi.org/10.1101/768739

Functions of proteins are in general related to their subcellular
locations. To identify the functions of a protein, we first need know
where this protein is located. Interacting proteins tend to locate in
the same subcellular location. Thus, it is imperative to take the
protein-protein interactions into account for computational
identification of protein subcellular locations.In this study, we
present a deep learning-based method, node2loc, to predict protein
subcellular location. node2loc first learns distributed
representations of proteins in a protein-protein network using
node2vec, which acquires representations from unlabeled data for
downstream tasks. Then the learned representations are further fed
into a recurrent neural network (RNN) to predict subcellular
locations. Considering the severe class imbalance of different
subcellular locations, Synthetic Minority Over-sampling Technique
(SMOTE) is applied to artificially boost subcellular locations with
few proteins.We construct a benchmark dataset with 16 subcellular
locations and evaluate node2loc on this dataset. node2loc yields a
Matthews correlation coefficient (MCC) value of 0.812, which
outperforms other baseline methods. The results demonstrate that the
learned presentations from a protein-protein network have strong
discriminate ability for classifying protein subcellular locations and
the RNN is a more powerful classifier than traditional machine
learning models. node2loc is freely available at
https://github.com/xypan1232/node2loc.


## Data sources

### Resources

There data are **generic**, i.e they aren't specific for a specific
organism or experimental conditions. Some might be derived from
experimental data, even though that information is generally not
directly or systematically available.

- Protein-protein interaction (PPI) data, such as that extracted from
  the [STRING](https://string-db.org/) or
  [InAct](https://www.ebi.ac.uk/intact/) databases. The STRING
  database is what was used in the reference above.
  
- Cellular component (CC) of the [Gene
  Ontology](http://geneontology.org/) (GO). Also available through
  various R packages - see for example
  [here](http://lgatto.github.io/pRoloc/articles/v05-pRoloc-transfer-learning.html#sec:goaux)
  for a relevant application.

### Experimental data

Experimental data is recorded for a specific cell type, under
**specific** conditions.

- Microscopy-based sub-cellular localisation, notably the [Human
  Protein
  Atlas](https://www.proteinatlas.org/humanproteome/cell). [Here](http://lgatto.github.io/hpar/articles/hpar.html)
  and
  [here](http://lgatto.github.io/pRoloc/articles/v05-pRoloc-transfer-learning.html#sec:hpaaux)
  are two examples describing how to access the data in R. The latest
  data can also be downloaded directly from the HPA's page as a
  table. As far as I know, the images aren't available.

- Quantitative mass spectrometry-based data (spatial proteomics),
  readily available [here](https://github.com/lgatto/pRolocdata/).

## Other

- Some researchers have used features computed from the protein
  sequence such as the pseudo amino acid composition. These aren't
  very reliable and better ignored.

The paper [Learning from Heterogeneous Data Sources: An Application in
Spatial
Proteomics](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004920)
(see this
[vignette](http://lgatto.github.io/pRoloc/articles/v05-pRoloc-transfer-learning.html)
to reproduce the analysis) provides a description of these various
data sources and how they have been combined.


