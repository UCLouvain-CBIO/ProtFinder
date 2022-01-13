import torch
import torch.nn as nn

from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd

from visualize import violin_plot
from math import log

__author__ = 'Grover'

'''
# For Linear model
thres_list =[
    0.5158,
    0.62,
    0.64,
    0.58,
    0.51,
    0.5036,
    0.42447,
    0.402,
    0.523,
    0.5413,
    0.4283,
    0.459,
    0.506,
    0.5102,
    0.4854,
    0.504,
    0.5087,
    0.45,
    0.502,
    0.5717,
    0.582,
    0.4832,
    0.4307,
    0.5232,
    0.5695,
    0.5,
    0.5594,
    0.5163
]
'''

# For LSTM model
thres_list =[
    0.5156,
    0.5840,
    0.5057,
    0.5743,
    0.5021,
    0.5054,
    0.5318,
    0.4802,
    0.4809,
    0.5200,
    0.4852,
    0.3569,
    0.5033,
    0.5090,
    0.4693,
    0.5088,
    0.5048,
    0.5986,
    0.5002,
    0.5795,
    0.5241,
    0.4758,
    0.5410,
    0.5396,
    0.5683,
    0.5075,
    0.5524,
    0.4645
]

def get_bin2(y, t):
    x = torch.randn(y.size())
    if t is None:
        for i in range(x.size()[0]):
            t = thres_list[i]
            for j in range(x.size()[1]):
                if y[i,j] > log(t):
                    x[i,j] = 1
                else:
                    x[i,j] = 0
    else: 
        for i in range(x.size()[0]):
            for j in range(x.size()[1]):
                if y[i,j] > log(t):
                    x[i,j] = 1
                else:
                    x[i,j] = 0
    _, x = x.max(dim=1)
    return x

def get_bin3(y, t):
    x = torch.randn(y.size())
    for i in range(x.size()[0]):
        for j in range(x.size()[1]):
            for k in range(x.size()[2]):
                if y[i,j,k] > log(t):
                    x[i,j,k] = 1
                else:
                    x[i,j,k] = 0
    _, x = x.max(dim=2)
    return x

def binarize3(x, class_id, device, thres_list=thres_list):
    threshold = thres_list[class_id]
    bin_x = get_bin3(x, threshold)
    return bin_x.to(device)

def binarize2(x, device, class_id=None, thres_list=thres_list):
    if class_id is None:
        threshold = None
    else:
        threshold = thres_list[class_id]
    bin_x = get_bin2(x, threshold)
    return bin_x.to(device)

def confusion(pred, target):
    '''
    Computes sensitivity and specificity

    input : pred <Torch tensor> : predictions of shape [batch_size, n_classes, 2]. The dimension with 2 holds the loglikelihoods 
    input : target <Torch tensor> : target of shape [batch_size, n_classes]
    
    return : TPR, TNR <float, float> : sensitivity and specificity values
    '''
    cnf = confusion_matrix(pred.tolist(), target.tolist())
    FP = cnf[0][1]  
    FN = cnf[1][0]
    TP = cnf[0][0]
    TN = cnf[1][1]

    # AUC-ROC
    try:
        AUC = roc_auc_score(target.tolist(), pred.tolist())
    except:
        AUC = None

    return TP, FP, TN, FN, AUC

    
def multilabel_loss(pred, target):
    '''
    Computes the loss between predictions and the target values

    input : pred <Torch tensor> : predictions of shape [batch_size, n_classes, 2]. The dimension with 2 holds the loglikelihoods 
    input : target <Torch tensor> : target of shape [batch_size, n_classes]
    
    return : loss <Torch tensor> : average loss across different classes. A tensor of shape [batch_size]
    '''
    loss_fn = nn.CrossEntropyLoss()
    all_loss = list()
    for i in range(target.shape[1]):
        tar = target[:, i].contiguous()
        pr = pred[:,i,:].contiguous()
        loss = loss_fn(pr, tar.long())
        sum_loss = torch.sum(loss, dim=0).view(1)
        all_loss.append(sum_loss)
    
    x = torch.cat(all_loss)
    sum_x = torch.sum(x).item()
    for i in range(x.shape[0]):
        x[i] *= x[i]/sum_x
    return torch.sum(x)

def get_correct(pred, target, device, flag=False, final=False, thresh=None):
    '''
    Computes the total number of hits

    input : pred <Torch tensor> : predictions of shape [batch_size, n_classes, 2]. The dimension with 2 holds the loglikelihoods 
    input : target <Torch tensor> : target of shape [batch_size, n_classes]
    input : flag <bool> : Whether to return classwise accuracies
    input : final <bool> : Whether the test results are for the final epoch
    
    return : n_correct, class_corr <int, Torch tensor> : Number of hits, tensor containing class accuracies
    '''

    n_correct = 0
    class_corr = list()
    TP, FP, TN, FN, AUC = 0, 0, 0, 0, 0.0
    tar_count = dict()
    pred_count = dict()
    confusion_info = dict()
    
    # if final:
    for i in range(target.shape[0]):
        pr = pred[i,:,:].contiguous()
        tr = target[i,:].contiguous()
        # _, pr = pr.max(dim=1)
        if thresh is not None:
            pr = binarize2(pr, device=device, thres_list=thresh)
        else:
            pr = binarize2(pr, device=device)
        flag = True
        try:
            tp, fp, tn, fn, auc = confusion(pr, tr)
        except:
            flag = False 

        pred_i = str(int((pr == 1).float().sum().item()))
        tar_i = str(int((tr == 1).float().sum().item()))
        try:
            tar_count[tar_i] += 1
        except:
            tar_count[tar_i] = 1
        try:
            pred_count[pred_i] += 1
        except:
            pred_count[pred_i] = 1
        if pred_i == '18':
            print('For protein X: ', tar_i)

        if flag:
            try:
                confusion_info[pred_i]['tp'] += tp
                confusion_info[pred_i]['fp'] += fp
                confusion_info[pred_i]['tn'] += tn
                confusion_info[pred_i]['fn'] += fn
            except:
                info = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
                confusion_info[pred_i] = info

            TP += tp
            FP += fp
            TN += tn
            FN += fn
            if auc:
                AUC += auc

        # if i == 0:
        #     print(f'Target: \n{tr.data}')
        #     print(f'Prediction: \n{pr.data}')

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    
    # MCC
    MCC = ((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)

    #Pick the argmax of target and pred for each class
    for i in range(target.shape[1]):
        tar = target[:, i].contiguous()
        pr = pred[:,i,:].contiguous()
        # _, pr = pr.max(dim=1)
        if thresh is not None:
            pr = binarize2(pr, device=device, class_id=i, thres_list=thresh)
        else:
            pr = binarize2(pr, device=device, class_id=i)
        corr = (pr == tar).float().sum()
        n_correct += corr
        #class_corr.append(corr/target.shape[0])
        try:
            tp, fp, tn, fn, _ = confusion(pr, tar)
            # print(i, tp, fp, tn, fn)
            # mcc = ((tp*tn)-(fp*fn))/(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)
        except:
            tp, fp, tn, fn = 0, 0, 0, 0
        class_corr.append([tp, fp, tn, fn])        

    # if not flag:
        # return n_correct, None, (None, None, None)
    class_corr = torch.FloatTensor(class_corr)
    return n_correct, class_corr, (TPR, TNR, AUC, MCC), (pred_count, confusion_info, tar_count)

if __name__ == "__main__":
    y_hat = np.zeros((4,7))
    y = np.ones((4,7,2))

    device = torch.device("cuda")
    y = torch.from_numpy(y).to(device)
    y_hat = torch.from_numpy(y_hat).to(device)
    
    loss = multilabel_loss(y, y_hat)
    _,_,sens, spec = get_correct(y, y_hat, flag=True, final=True)
    print(loss, sens, spec)

