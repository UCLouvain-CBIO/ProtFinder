from tqdm import tqdm
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data_loader import FeatureDataset
from model import Classifier
from loss import classification_loss, get_correct

__author__ = 'Grover'


def test(dl, model, batch_size):
    '''
    The testing loop.

    input : dl <class torch.utils.data.DataLoader> : Test Dataloader
    input : model <class torch.nn.Module> : LSTM Classifier
    input : batch_size <int> : Size of each batch
    '''
    #Get the model in training mode
    model.eval()

    epoch_loss = 0.0
    correct = 0
    total = 0

    for X, y in tqdm(dl):
        if y.shape[0] == batch_size:

            pred = model(X, batch_size)
            loss = classification_loss(pred, y)
            loss = torch.sum(loss, dim=0)

            n_correct, n_points = get_correct(pred, y)
            correct += n_correct
            total += n_points
            epoch_loss += loss.item()
    
    acc = correct/total
    avg_loss = epoch_loss/total

    print(f'Test Loss: {avg_loss}')
    print(f'Test Accuracy: {acc}')
    
    return acc, avg_loss


def train(train_dl, test_dl, model, optim, epochs, batch_size, save, start_epoch=0, model_save_path=None):
    '''
    The training loop.

    input : dl <class torch.utils.data.DataLoader> : Training Dataloader
    input : model <class torch.nn.Module> : LSTM Classifier
    input : optim <class torch.optim> : Adam Optimizer
    input : epochs <int> : Number of epochs to train
    input : batch_size <int> : Size of each batch
    input : save <bool> : True if model needs to be saved
    input : start_epoch <int> : The epoch to start training from
    input : model_save_path <str> : Path where checkpoint needs to be saved
    '''

    #Get the model in training mode
    writer = SummaryWriter('bioplex')

    for epoch in range(start_epoch, epochs):
        
        print(f'Epoch: {epoch}')
        epoch_loss = 0.0
        correct = 0
        total = 0
        model.train()

        for X, y in tqdm(train_dl):
            if y.shape[0] == batch_size:

                pred = model(X, batch_size)
                loss = classification_loss(pred, y)
                loss = torch.sum(loss, dim=0)

                n_correct, n_points = get_correct(pred, y)
                correct += n_correct
                total += n_points

                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item()


        acc = correct/total
        avg_loss = epoch_loss/total

        print(f'Train Loss: {epoch_loss}')
        print(f'Train Accuracy: {acc}')

        test_acc, test_loss = test(test_dl, model, batch_size)
        
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Accuracy/Train', acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)

        if save:
            print('Saving model')
            checkpoint = {
                'model': model.state_dict(), 
                'optim': optim.state_dict(),
                'epoch': epoch    
            }
            torch.save(checkpoint, model_save_path)
    
    writer.close()


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
    parser.add_argument("-u", "--nlayers", type=int,
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
                        help="The path for saving model.", default='../models/bioplex.pt')
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

    #initialising model 
    model_save_path = args.modelpath
    model = Classifier(train_data.get_embed_dim(), args.nclass,num_layers=args.nlayers)

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

    print("\nTraining Start...")

    train(
        train_dl=train_loader,
        test_dl=test_loader,
        model=model, 
        optim=optim, 
        epochs=args.epochs,
        batch_size=args.batchsize,
        save=save,
        start_epoch=start_epoch,
        model_save_path=model_save_path)

    end_time = time.time()  
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
