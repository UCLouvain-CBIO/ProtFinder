from tqdm import tqdm
import time
from os import makedirs
from os.path import exists
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_loader import TestDataset
from model import Classifier, SimpleClassifier

__author__ = 'Grover'

all_cols = [
    'Protein',
    'Actin filaments',
    'Cell Junctions',
    'Centriolar satellite',
    'Centrosome',
    'Cytokinetic bridge',
    'Cytoplasmic bodies',
    'Cytosol',
    'Endoplasmic reticulum',
    'Endosomes',
    'Focal adhesion sites',
    'Golgi apparatus',
    'Intermediate filaments',
    'Lipid droplets',
    'Lysosomes',
    'Microtubules',
    'Midbody',
    'Midbody ring',
    'Mitochondria',
    'Mitotic spindle',
    'Nuclear bodies',
    'Nuclear membrane',
    'Nuclear speckles',
    'Nucleoli',
    'Nucleoli fibrillar center',
    'Nucleoplasm',
    'Peroxisomes',
    'Plasma membrane',
    'Vesicles'
]

def test(dl, model, batch_size, device, save=False):
    '''
    The testing loop.

    input : dl <class torch.utils.data.DataLoader> : Test Dataloader
    input : model <class torch.nn.Module> : LSTM Classifier
    input : batch_size <int> : Size of each batch
    input : flag <bool> : Whether to compute classwise accuracies
    '''
    #Get the model in eval mode
    model.eval()
    df = pd.DataFrame(columns=all_cols) 

    for X, ids in tqdm(dl):
        if X.shape[0] == batch_size:
            pred = model(X, batch_size)

            pred = pred.cpu().detach().numpy()
            pred = pred[:,:,1]
            pred = pred.squeeze()
            pred = np.exp(pred)
            df1 = pd.DataFrame(pred, columns=all_cols[1:])
            df1['Protein'] = ids
            df = pd.concat([df, df1] ,ignore_index=True, sort=False)

    if save:
        path = '../results/'
        if not exists(path):
            makedirs(path)
        fname = 'output.csv'
        df.to_csv((path+fname), index=False)

def get_args():
    '''
    Parses arguments using ArgumentsParser
    
    returns : <class argparse.Namespace> : argument values
    '''
    parser = ArgumentParser(description="This a reimplementation of node2loc",
                        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--nclass", type=int,
                        help="Number of class. Must be at least 2 aka two-classification.", default=2)
    parser.add_argument("-n", "--nlayers", type=int,
                        help="Number of LSTM layers.", default=1)
    parser.add_argument("-d", "--datapath", type=str,
                        help="The path of inference dataset.", default='../data/infer_all_0.6.csv')
    parser.add_argument("-m", "--modelpath", type=str,
                        help="The path for loading model.", default='../models/fcc_model.pt')
    parser.add_argument("-g", "--gpu",
                        help='GPU to use', action="store_true")
    parser.add_argument("-s", "--save",
                        help='Save output', action="store_true")
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    start_time = time.time()
    
    args = get_args()
    start_epoch = 0

    torch.manual_seed(args.randomseed)
    device = torch.device("cuda" if args.gpu else "cpu")

    #processing test data
    test_data = TestDataset(args.datapath, device)
    
    #initialising model (choose between non-DAG and DAG connected models) 
    model_save_path = args.modelpath
    # model = SimpleClassifier(train_data.get_embed_dim(), args.nclass,num_layers=args.nlayers)
    model = Classifier(test_data.get_embed_dim(), args.nclass, num_layers=args.nlayers, device=device)

    #loading model from checkpoint
    try: 
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model'])
        print('\nModel loaded\n')
        batch_size = checkpoint['batch_size']
    except:
        raise Exception("Model cannot be loaded.")
    
    model.to(device)

    test_loader = DataLoader(test_data, batch_size=batch_size)

    if args.save:
        save=True
    else:
        save=False

    test(test_loader, model, batch_size, save=save, device=device)

    end_time = time.time()  
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
