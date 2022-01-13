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
from utils import merge1, merge2, get_distwise_mcc, merge_class, get_classwise_scores

__author__ = 'Grover'


max_mcc, max_epoch = 0, -1

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
    global max_epoch
    global c

    target_list = list()
    pred_list = list()
    meta_info = dict()
    final_pred_c = dict()
    final_tar_c = dict()

    for X, y in tqdm(dl):
        if y.shape[0] == batch_size:
            pred = model(X, batch_size)
            loss = multilabel_loss(pred, y)

            n_correct, t, (sens, spec, auc, mcc), (pr_c, meta, tr_c) = get_correct(pred, y, device=device, flag=flag, final=final)
            
            if final:
                final_pred_c = merge1(final_pred_c, pr_c)   
                final_tar_c = merge1(final_tar_c, tr_c)
                meta_info = merge2(meta_info, meta)
                # classwise_scores.append(t.view(1,-1))
                classwise_scores = merge_class(classwise_scores, t)
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
        print('Target dist: ', final_tar_c)
        print('Pred dist: ', final_pred_c)
        distwise_score = get_distwise_mcc(meta_info)
        print('Dist.wise mcc :', distwise_score)
        # classwise_scores = torch.cat(classwise_scores, dim=0)
        # classwise_scores = classwise_scores.mean(dim=0)
        classwise_scores = get_classwise_scores(classwise_scores)
        print('Classwise mcc: ', classwise_scores)
    
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
        if mcc > max_mcc:
            max_mcc = mcc
            max_epoch = epoch
            c = 0
            print("Max updated")
        else:
            c += 1
        # if c == 5:
        #     print(f'Max MCC is {max_mcc} at epoch {max_epoch}')
        #     quit()

    # if final:
    #     plot_differences(target_list, pred_list)

    return acc, avg_loss


def train(train_dl, test_dl, model, optim, epochs, batch_size, save, device, start_epoch=0, model_save_path=None):
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
    with torch.autograd.set_detect_anomaly(True):

        #Get the model in training mode
        writer = SummaryWriter('bioplex_exp1_0.7')
        print(start_epoch, epochs)
        for epoch in range(start_epoch, epochs):
            
            print(f'Epoch: {epoch}')
            epoch_loss = 0.0
            correct = 0
            total = 0
            flag = False
            model.train()
            global c

            for X, y in tqdm(train_dl):
                if y.shape[0] == batch_size:
                    
                    pred = model(X, batch_size)
                    
                    loss = multilabel_loss(pred, y)

                    n_correct,_,_,_ = get_correct(pred, y, device)
                    correct += n_correct
                    total += batch_size

                    optim.zero_grad()
                    loss.backward()
                    # plot_grad_flow(model.named_parameters())
                    optim.step()
                    epoch_loss += loss.item()

            acc = correct/(total*28)
            avg_loss = epoch_loss/total

            print(f'Train Loss: {avg_loss}')
            print(f'Train Accuracy: {acc}')

            test_acc, test_loss = test(test_dl, model, batch_size, flag=flag, device=device, epoch=epoch)
            
            writer.add_scalar('Loss/Train', avg_loss, epoch)
            writer.add_scalar('Loss/Test', test_loss, epoch)
            writer.add_scalar('Accuracy/Train', acc, epoch)
            writer.add_scalar('Accuracy/Test', test_acc, epoch)

            if save and epoch == (epochs-1):
                print('Saving model')
                checkpoint = {
                    'model': model.state_dict(), 
                    'optim': optim.state_dict(),
                    'epoch': epoch,
                    'batch_size': batch_size
                }
                torch.save(checkpoint, model_save_path)
        
        writer.close()

        print("Predicting final score ---")
        test_acc, test_loss = test(test_dl, model, batch_size, device=device, flag=True, final=True)

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

    # print("\nRNN HyperParameters:")
    # print("\nN-classes:{0}, Training epochs:{1}, Learning rate:{2}, Sequences fragment length: {3}".format(
    #         args.nclass, args.epochs, args.learningrate, args.fragment))
    # print("\nCross-validation info:")
    # print("\nK-fold:", args.kfolds, ", Random seed is", args.randomseed)

    return args

if __name__ == "__main__":
    start_time = time.time()
    
    args = get_args()
    start_epoch = 0

    torch.manual_seed(args.randomseed)
    device = torch.device("cuda" if args.gpu else "cpu")

    #processing training data    
    # train_data = FeatureDataset(args.trainpath, device)
    # train_loader = DataLoader(train_data, batch_size=args.batchsize)
    train_data, train_loader = None, None

    #processing test data
    test_data = FeatureDataset(args.testpath, device)
    test_loader = DataLoader(test_data, batch_size=args.batchsize)

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

    print(f"\nTraining Start at epoch {start_epoch}...")

    train(
        train_dl=train_loader,
        test_dl=test_loader,
        model=model, 
        optim=optim, 
        epochs=args.epochs,
        batch_size=args.batchsize,
        save=save,
        device=device,
        start_epoch=start_epoch,
        model_save_path=model_save_path)

    end_time = time.time()  
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
