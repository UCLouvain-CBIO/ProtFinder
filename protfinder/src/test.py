from tqdm import tqdm
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from data_loader import FeatureDataset
from model import Classifier, SimpleClassifier
from loss import multilabel_loss, get_correct
from visualize import violin_plot

__author__ = 'Grover'


max_mcc = 0

def plot_grad_flow(named_parameters):
    '''
    Plots the flow of gradients to understand backpropagation
    '''
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('../grad_flow.png')

def plot_differences(target, pred):
    target = torch.cat(target, dim=0)
    pred = torch.cat(pred, dim=0)
    violin_plot(target.data, pred.data)


def test(dl, model, batch_size, device, flag=False, final=False, epoch=-1):
    '''
    The testing loop.

    input : dl <class torch.utils.data.DataLoader> : Test Dataloader
    input : model <class torch.nn.Module> : LSTM Classifier
    input : batch_size <int> : Size of each batch
    input : flag <bool> : Whether to compute classwise accuracies
    '''
    #Get the model in training mode
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0
    iteration = 0
    acc = 0
    sens_total = 0
    spec_total = 0
    auc_total = 0
    mcc_total = 0
    classwise_scores = list()
    global max_mcc
    global c

    target_list = list()
    pred_list = list()

    for X, y in tqdm(dl):
        if y.shape[0] == batch_size:
            pred = model(X, batch_size)
            loss = multilabel_loss(pred, y)

            n_correct, t, (sens, spec, auc, mcc) = get_correct(pred, y, device=device, flag=flag, final=final)
            
            if final:
                classwise_scores.append(t.view(1,-1))
            correct += n_correct
            total += batch_size
            iteration += 1
            try:
                spec_total += spec
                sens_total += sens
                auc_total += auc
                mcc_total += mcc
            except:
                pass
            target_list.append(y)
            pred_list.append(pred)
            epoch_loss += loss.item()

    if len(classwise_scores) != 0:
        classwise_scores = torch.cat(classwise_scores, dim=0)
        classwise_scores = classwise_scores.mean(dim=0)
        print(classwise_scores)
    
    acc = correct/(total*28)
    avg_loss = epoch_loss/total

    print(f'Test Loss: {avg_loss}')
    print(f'Test Accuracy: {acc}')

    if spec is not None and sens is not None and auc is not None and mcc is not None:
        sens = sens_total/iteration
        spec = spec_total/iteration
        auc = auc_total/total
        mcc = mcc_total/iteration
        print(f'Sensitivity: {sens}')
        print(f'Specificity: {spec}')
        print(f'AUC-ROC: {auc}')
        print(f'MCC: {mcc}')


def get_args():
    '''
    Parses arguments using ArgumentsParser
    
    returns : <class argparse.Namespace> : argument values
    '''
    parser = ArgumentParser(description="This is the official implementation of ___",
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--nclass", type=int,
                        help="Number of class. Must be at least 2 aka two-classification.", default=2)
    parser.add_argument("-n", "--nlayers", type=int,
                        help="Number of LSTM layers.", default=1)
    parser.add_argument("-d", "--datapath", type=str,
                        help="The path of dataset.", required=True)
    parser.add_argument("-m", "--modelpath", type=str,
                        help="The path of model.", required=True)
    parser.add_argument("-g", "--gpu",
                        help='GPU to use', action="store_true")
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    start_time = time.time()
    
    args = get_args()
    device = torch.device("cuda" if args.gpu else "cpu")

    #processing test data
    test_data = FeatureDataset(args.datapath, device)

    #initialising model (choose between non-DAG and DAG connected models) 
    model_save_path = args.modelpath
    # model = SimpleClassifier(train_data.get_embed_dim(), args.nclass,num_layers=args.nlayers)
    model = Classifier(test_data.get_embed_dim(), args.nclass, num_layers=args.nlayers, device=device)

    #loading model from checkpoint
    try: 
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model'])
        print('\nModel loaded\n')
        start_epoch = checkpoint['epoch'] + 1
    except:
        print('Model not found.')
        quit()
    model.to(device)

    batchsize = checkpoint['batch_size']

    test_loader = DataLoader(test_data, batch_size=batchsize)


    test(
        dl=test_loader,
        model=model, 
        batch_size=batchsize,
        device=device,
        flag=True,
        final=True)

    end_time = time.time()  
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))