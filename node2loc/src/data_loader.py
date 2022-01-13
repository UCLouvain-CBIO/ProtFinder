import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import OneHotEncoder  
from imblearn.over_sampling import SMOTE

__author__ = 'Grover'


class FeatureDataset(Dataset):
    '''
    The dataset where column 1 is class label.
    Remaining 500 columns together form the node2vec embedding of a protein.
    '''
    def __init__(self, data_path, device):
        '''
        Initializes class variables.

        input : data_path <str> : path to the data
        input : device <torch.device> : gpu or cpu
        '''
        self.device = device
        self.df = pd.read_csv(data_path)
        self.n_samples = 0 
        self.X, self.y = self.clean_data()

    def clean_data(self):
        '''
        Preprocesses the data in the required format. Over-sampling is done using SMOTE.
        Target classes are one-hot encoded.
        
        return : <torch, torch> : torch tensors of input (n_samples, 500) and lable (n_samples, 16)
        '''
        target = self.df.iloc[:,0].values
        feats = self.df.iloc[:,1:].values
        self.n_samples = target.size

        # seq_length = feats.shape[1]

        #Getting the number of datapoints per class
        # num_categories = np.unique(target).size
        sum_y = np.asarray(np.unique(target.astype(int), return_counts=True))
        df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
        print('\n', df_sum_y)

        #Oversampling data using SMOTE
        # sm = SMOTE(k_neighbors=2)
        # x_resampled, y_resampled = sm.fit_sample(feats, target)
        # np_resampled_y = np.asarray(np.unique(y_resampled, return_counts=True))
        # df_resampled_y = pd.DataFrame(np_resampled_y.T, columns=['class', 'sum'], index=None)
        # print("\nNumber of samples after over sampleing:\n{0}".format(df_resampled_y))
        
        x_resampled, y_resampled = feats, target 
        #Onehot encode the output class labels
        enc = OneHotEncoder(categories='auto', sparse=True, dtype=np.int)
        one_hot_mat = enc.fit(target.reshape(-1, 1))
        # new_target = one_hot_mat.transform(target.reshape(-1, 1)).toarray() 
        new_resampled_target = one_hot_mat.transform(y_resampled.reshape(-1, 1)).toarray() 

        x_resampled = torch.from_numpy(x_resampled).to(self.device)
        new_resampled_target = torch.from_numpy(new_resampled_target).to(self.device)
        return x_resampled, new_resampled_target

    def __len__(self):
        '''
        return : <int> : total samples in the dataset
        '''
        return self.n_samples

    def __getitem__(self, idx):
        '''
        input : idx <int> : extract data at index idx
        return : <torch, torch> : returns the feature and label tensors corresponding to idx. 
        '''
        return self.X[idx], self.y[idx]
    
    def get_embed_dim(self):
        return self.X.shape[1]

if __name__ == "__main__":
    d = FeatureDataset('../data/train_dataset.csv', torch.device('cuda'))
    dl = DataLoader(d, batch_size=64)

    x, y = next(iter(dl))

    print(x.shape, y.shape)
    
