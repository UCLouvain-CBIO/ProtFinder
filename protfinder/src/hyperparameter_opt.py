from tqdm import tqdm
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data_loader import FeatureDataset
from model import Classifier, SimpleClassifier
from loss import multilabel_loss, get_correct
from visualize import violin_plot
from utils import hyperparam_opt

__author__ = 'Grover'


def get_args():
    '''
    Parses arguments using ArgumentsParser
    
    returns : <class argparse.Namespace> : argument values
    '''
    parser = ArgumentParser(description="This a reimplementation of node2loc",
                        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--nclass", type=int,
                        help="Number of class. Must be at least 2 aka two-classification.", default=2)
    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of training epochs.", default=20)
    parser.add_argument("-b", "--batchsize", type=int,
                        help="Size of each training batch", default=16)
    parser.add_argument("-n", "--nlayers", type=int,
                        help="Number of LSTM layers.", default=1)
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("-f", "--fragment", type=int,
                        help="Specifying the `length` of sequences fragment.", default=1)
    parser.add_argument("-d", "--trainpath", type=str,
                        help="The path of training dataset.", required=True)
    parser.add_argument("-t", "--testpath", type=str,
                        help="The path of test dataset.", default='../data/test_bioplex.csv')
    parser.add_argument("-m", "--modelpath", type=str,
                        help="The path for saving model.", default='../models/bioplex_exp1_0.7.pt')
    parser.add_argument("-lr", "--learningrate", type=float,
                        help="Learning rate.", default=1e-2)
    parser.add_argument("-g", "--gpu",
                        help='GPU to use', action="store_true")
    parser.add_argument("-s", "--save",
                        help='Save model', action="store_true")

    args = parser.parse_args()

    print("\nRNN HyperParameters:")
    print("\nN-classes:{0}, Training epochs:{1}, Learning rate:{2}, Sequences fragment length: {3}".format(
            args.nclass, args.epochs, args.learningrate, args.fragment))
    print("\nCross-validation info:")
    print("\nK-fold:", args.kfolds, ", Random seed is", args.randomseed)

    return args

if __name__ == "__main__":
    start_time = time.time()
    
    args = get_args()
    start_epoch = 0

    torch.manual_seed(args.randomseed)
    device = torch.device("cuda" if args.gpu else "cpu")

    #processing training data    
    train_data = FeatureDataset(args.trainpath, device)
    train_loader = DataLoader(train_data, batch_size=args.batchsize)

    #processing test data
    test_data = FeatureDataset(args.testpath, device)
    test_loader = DataLoader(test_data, batch_size=args.batchsize)

    #initialising model (choose between non-DAG and DAG connected models) 
    model_save_path = args.modelpath
    # model = SimpleClassifier(train_data.get_embed_dim(), args.nclass,num_layers=args.nlayers)
    model = Classifier(train_data.get_embed_dim(), args.nclass,num_layers=args.nlayers, device=device)

    #loading model from checkpoint
    try: 
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model'])
        print('\nModel loaded\n')
        start_epoch = checkpoint['epoch'] + 1
    except:
        pass
    model.to(device)

    #initialising loading optimizer from checkpoint
    optim = torch.optim.Adam(model.parameters(), lr=args.learningrate)
    try: 
        optim.load_state_dict(checkpoint['optim'])
        print('Optimizer loaded\n')    
    except:
        pass

    
    print('Total number of parameters = ', sum(p.numel() for p in model.parameters()))
    print('Total number of trainable parameters = ', sum(p.numel() 
                                for p in model.parameters() if p.requires_grad))

    if args.save:
        save=True
    else:
        save=False

    hyperparam_opt(
        train_dl=train_loader,
        test_dl=test_loader,
        model=model, 
        optim=optim,
        device=device,
        batch_size=args.batchsize
    )