import pandas as pd
import numpy as np

import json
from networkx.readwrite import json_graph
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MultiLabelBinarizer

__author__ = 'Grover'

classes = [
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

def get_list(x):
    x = x[1:-1]
    x = x.replace('"', '')
    x = x.replace("'", "").strip()
    x = x.split(',')
    x = [i.strip() for i in x]
    return np.array(x)

class FeatureDataset(Dataset):
    '''
    The dataset where column 1 is class label.
    Remaining feature_dim columns together form the node2vec embedding of a protein.
    '''
    def __init__(self, data_path, device):
        '''
        Initializes class variables.

        input : data_path <str> : path to the data
        input : device <torch.device> : gpu or cpu
        '''
        super(FeatureDataset, self).__init__()
        self.device = device
        self.loc_id = list()
        self.df = pd.read_csv(data_path)
        self.n_samples = 0 
        self.X, self.y = self.clean_data()

    def clean_data(self):
        '''
        Preprocesses the data in the required format. Over-sampling is done using SMOTE.
        Target classes are one-hot encoded.
        
        return : <torch, torch> : torch tensors of input (n_samples, feature_dim) and label (n_samples, n_classes)
        '''
        self.df['locations'] = self.df['locations'].apply(lambda x: get_list(x))
        target = self.df.iloc[:,0].values.tolist()
        feats = self.df.iloc[:,1:].values
        self.n_samples = len(target)
 
        #MultiLabel encode the output class labels
        enc = MultiLabelBinarizer(classes=classes)
        mat = enc.fit_transform(target)
        new_resampled_target = np.array(mat) 
        self.loc_id = enc.classes_
        self.loc_id = {k:v for v,k in enumerate(self.loc_id)}
        # print(self.loc_id)
        x_resampled = torch.from_numpy(np.array(feats)).to(self.device)
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
    
    def get_y_dist(self):
        return torch.sum(self.y, 0)
    

class TestDataset(Dataset):
    '''
    The dataset of unlabelled proteins.
    '''
    def __init__(self, data_path, device):
        '''
        Initializes class variables.

        input : data_path <str> : path to the data
        input : device <torch.device> : gpu or cpu
        '''
        super(TestDataset, self).__init__()
        self.device = device
        self.loc_id = list()
        self.df = pd.read_csv(data_path)
        self.n_samples = 0 
        self.X, self.ids = self.clean_data()

    def clean_data(self):
        '''
        Preprocesses the data in the required format. Over-sampling is done using SMOTE.
        Target classes are one-hot encoded.
        
        return : <torch, torch> : torch tensors of input (n_samples, feature_dim) and lable (n_samples, n_classes)
        '''
        ids = self.df.protein.tolist()
        feats = self.df.iloc[:,1:].values
        self.n_samples = len(feats)

        x_resampled = torch.from_numpy(np.array(feats)).to(self.device)
        return x_resampled, ids

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
        return self.X[idx], self.ids[idx]
    
    def get_embed_dim(self):
        return self.X.shape[1]
    


class GraphSAGEDataset(Dataset):
    '''
    The dataset where column 1 is class label.
    Remaining feature_dim columns together form the node2vec embedding of a protein.
    '''
    def __init__(self, data_path, device, test=False):
        '''
        Initializes class variables.

        input : data_path <str> : path to the data
        input : device <torch.device> : gpu or cpu
        input : test <bool> : Whether in test mode
        '''
        super(GraphSAGEDataset, self).__init__()
        self.device = device
        self.test = test
        self.X, self.y = self.load_data(data_path)

    def load_data(self, dataset_dir):
        print("Loading data...")
        fname = dataset_dir.split('/')[-1]
        G = json_graph.node_link_graph(json.load(open(dataset_dir + f"/{fname}-G.json")))
        labels = json.load(open(dataset_dir + f"/{fname}-class_map.json"))
        labels = {int(i):l for i, l in labels.items()}

        if not self.test:
            ids = [n for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']]
            labels = np.array([labels[i] for i in ids])
            if labels.ndim == 1:
                labels = np.expand_dims(labels, 1)
        else:
            ids = [n for n in G.nodes() if G.nodes[n]['test']]
            labels = np.array([labels[i] for i in ids])
        
        return ids, torch.from_numpy(labels).to(device=self.device)

    
    def __len__(self):
        '''
        return : <int> : total samples in the dataset
        '''
        return len(self.X)

    def __getitem__(self, idx):
        '''
        input : idx <int> : extract data at index idx
        return : <torch, torch> : returns the feature and label tensors corresponding to idx. 
        '''
        return self.X[idx], self.y[idx]

if __name__ == "__main__":
    d = FeatureDataset('../data/test_string_new.csv', torch.device('cuda'))
    dl = DataLoader(d, batch_size=64)

    x, y = next(iter(dl))

    print(x.shape, y.shape)
    print(y[0])
