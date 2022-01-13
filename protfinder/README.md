# ProtFinder

This directory follows the structure as given below: 
```

│   README.md
└───src
│   │   data_loader.py
│   │   .
│   │   .
│   │   .
└───data
|   │   train_all_0.6.csv
|   │   test_all_0.6.csv
|   |   infer_all_0.6.csv
└───dag_info
|   │   location_dag.txt
└───models
|   │   final_lstm_model.pt
└───results
    |   output.csv

```

To train and test the ProtFinder model, one must first copy the files generated in the [embedding directory](https://github.com/UCLouvain-CBIO/ProtFinder/tree/main/embeddings) to `data` directory here.

### Training
Training is done by calling `src/train.py` with the appropriate arguments. An example of the command can be found in `src/train.sh`. To train using the GraphSAGE model, one must run `src/graphsage_train.py` using a similar set of arguments. If `--save` flag is used the model will be saved in the `models` directory.

### Hyperparameter Optimization
Hyperparameter optimization is done using (Facebook's Ax)[https://github.com/facebook/Ax]. This can be done by using similar arguments as training to `src/hyperparameter_opt.py`.

### Inference
To predict the location annotations for proteins that do not have confident annotations in the Human Protein Atlas, one can use `src/infer.py`. The arguments for the same can be found in `src/infer.sh`. If the `--save` option is used, the probabilities will be saved in the `results` directory.
