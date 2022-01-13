import numpy as np
from pandas.io.formats import style
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

__author__ = "Grover"

def violin_plot(target, pred):
    target = target.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    
    pred = np.exp(pred)     # Since logsoftmax was used for training
    pred = pred[:, :, 1]
    y = target-pred
    n_class = y.shape[1]

    y_list = list()

    fn_list = list()
    fp_list = list()
    fn_near = list()
    fp_near = list()
    for i in range(n_class):
        arr = y[:,i]
        y_list.append(arr)
        fn = arr[arr > 0.5]
        fp = arr[arr < -0.5]
        try:
            fn_near.append(max(arr[arr < 0.5]))
        except :
            fn_near.append(0)
        try:
            fp_near.append(max(fp))
        except :
            fp_near.append(0)
        fn_list.append(len(fn))
        fp_list.append(len(fp))

    print(fn_list)
    print(fn_near)
    print(fp_list)
    print(fp_near)


    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(10,5))
    
    ax[0].boxplot(y_list[:14])
    ax[0].set_xlabel('Class ID')
    ax[0].set_xticks(range(14))
    ax[0].set_ylabel('Target-Pred')
    ax[0].yaxis.grid(True)
    
    ax[1].boxplot(y_list[14:], positions=range(15,29))
    ax[1].set_xlabel('Class ID')
    ax[1].set_xticks(range(14,28))
    ax[1].set_ylabel('Target-Pred')
    ax[1].yaxis.grid(True)

    fig.tight_layout()
    fig.savefig('../combined_more_exp1_0.7.png')

def barplot():
    tar = {'1': 2232, '3': 11, '2': 445, '0': 0, '18': 0}
    pred = {'1': 1899, '0': 349, '3': 22, '2': 417, '18': 1}
    X = [0, 1, 2, 3, 0, 1, 2, 3]
    y = [0, 2232, 445, 11, 349, 1899, 417, 22]
    hue = ['target', 'target', 'target', 'target', 'prediction', 'prediction', 'prediction', 'prediction']
    df = pd.DataFrame({'Number of locations': X, 'Number of proteins': y, 'Distribution':hue})
    sns.set(style='darkgrid')
    g = sns.barplot(x='Number of locations', y='Number of proteins', hue='Distribution', data=df)
    # g.set_xticks(range(5))
    # g.set_xticklabels([0,1,2,3,18])
    plt.title('Output Distributions (Number of subcellular locations)')
    plt.savefig('../output_dist.png', dpi=300)

if __name__ == '__main__':
    barplot()