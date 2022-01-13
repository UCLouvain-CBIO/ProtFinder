# Reproducing node2loc in Pytorch

This directory focuses on reproducing the results produced by [node2loc](https://ieeexplore.ieee.org/document/9431661).

## data_loader.py
Creates an object of the `Dataset` class. This is build using the training data of [node2loc](https://github.com/xypan1232/node2loc). The input to the model is a 500 dimensional vector generated from [node2vec](https://github.com/aditya-grover/node2vec). The target is the class label from 1 to 16. 

In this class, the data is oversampled using [SMOTE](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html). The classes are one-hot encoded to obtain a 16 dimensional vector for each class.

Finally, the torch tensors of sizes __(1, embedding_dim)__ and __(1, n_classes)__ are returned using the `__getitem__` function.

__Note -__ embedding_dim = 500, n_classes = 16

## model.py 
Creates an LSTM-based classifier that takes in a batch of input __(batch_size, embedding_dim)__ and returns the batch of output __(batch_size, n_classes)__ where each row gives the log likelihood of classes for the particular datapoint.

The datapoint is first passed through the LSTM layer which captures the relations between each of the embedding dimensions. The output is then passed through the Linear layer that converts the __hidden_dim__ to __output_dim__, which is same as __n_classes__. The LogSoftmax is applied then to compute the log likelihood for each class.

Update - A Directed Acyclic Graph (DAG) is generated from the GO IDs of the location classes. This was done using the [GOView](http://www.webgestalt.org/2017/GOView/) software. The DAG can be found in `/data/location_dag.txt` where each row depicts a chain of the DAG. This DAG will be used for multi-label classification.

## loss.py
There are two functions in this file - 
1. `classification_loss` which computes the __CrossEntropyLoss__ between the target class and the predicted log likelihoods. 
2. `get_correct` returns total number of datapoints and total number of datapoints that were predicted correctly in a batch.

## train.py
### Training
To train the node2loc, go to the `src` directory and run the project. This can be done as -

```bash
cd src/
python train.py --datapath ../data/train_dataset.csv --nclass 16 --epochs 60 --gpu 
```

This will train the model on `train_dataset.csv` which has 16 different subcellular locations as classes. The model will train for 60 epochs on the GPU. The trained model gets stored in the `models` directory after every epoch. This allows the model to continue training after some number of epochs.

You can experiment with the training using the following flags :

-   The path of dataset (`--datapath`)
-   Number of class. Must be at least 2 aka two-classification. (`--nclass`, default = 2)
-   Number of training epochs (`--epochs`, default = 20)
-   Size of each training batch (`--batchsize`, default = 16)
-   Number of folds. Must be at least 2. (`--kfolds`, default = 10)
-   Number of LSTM layers (`--nlayers`, default = 1)
-   Pseudo-random number generator state used for shuffling (`--randomseed`, default = 0)
-   Learning rate (`--learningrate`, default = 1e-2)
-   GPU to use. Use this to train on GPU. Otherwise, it will train on CPU. (`--gpu`)
-   Use model for testing. Use this flag when only inference is to be made. (`--test`)

### Testing/Evaluation
To test the node2loc, go to the `src` directory and run the project with `--test` flag. This can be done as -

```bash
cd src/
python train.py --datapath ../data/train_dataset.csv --nclass 16 --gpu --test
```

This will evaluate the trained model (which is picked from the `models` directory) on `train_dataset.csv` which has 16 different subcellular locations as classes.

### Plots
Another directory, called `runs` is created. This is due to [SummaryWriter() of TensorboardX](https://tensorboardx.readthedocs.io/en/latest/tutorial.html). Run the following command to see the plot -

```bash
cd src/
tensorboard --logdir=runs
```

Once this is done, one can see a link getting printed on the terminal/command-prompt. One can see the training loss and training accuracy plots at this link.

## Experiments
### Experiment 1
The training data is 100% of `data/train_dataset.csv` and the test data is 20% of that same file. The model is trained for 60 epochs.

To see the training and testing plots, please refer to `src/runs` directory using the tensorboard command as follows - 

```bash
cd src/
tensorboard --logdir=runs
```
Once this is done, one can see a link getting printed on the terminal/command-prompt. One can see the training loss and training accuracy plots at this link.

### Experiment 2
The training data is 80% of `data/train_dataset.csv` and the test data is 20% of that same file. The training and test data do not overlap. The model is trained for 60 epochs.

To see the training and testing plots, please refer to `src/runs2` directory using the tensorboard command as follows - 

```bash
cd src/
tensorboard --logdir=runs2
```

Once this is done, one can see a link getting printed on the terminal/command-prompt. One can see the training loss and training accuracy plots at this link.

If you want to see it simultaneously with the results of experiment 1, use `--port=6007`.

### Experiment 3
The training data is 100% of `data/train_dataset.csv` and the test data is 100% of `data/test_dataset.csv`. The model is trained for 60 epochs.

To see the training and testing plots, please refer to `src/runs3` directory using the tensorboard command as follows - 

```bash
cd src/
tensorboard --logdir=runs3
```

Once this is done, one can see a link getting printed on the terminal/command-prompt. One can see the training loss and training accuracy plots at this link.

If you want to see it simultaneously with the results of experiments 1 and 2, use `--port=6008`.

