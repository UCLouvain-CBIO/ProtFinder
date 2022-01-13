import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

__author__ = 'Grover'

class Classifier(nn.Module):
    '''
    Returns class log likelihoods for a batch of datapoints.
    '''
    def __init__(self, embed_size, n_classes, hidden_size=256, num_layers=1):
        '''
        Initializes class variables.

        input : embed_size <int> : embedding size
        input : hidden_size <int> : hidden size for the LSTM
        input : n_classes <int> : number of classes
        input : num_layers <int> : number of layers of LSTM
        '''
        super(Classifier, self).__init__()
        
        self.num_layers = num_layers
        self.embed_size = embed_size

        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=False)
        self.out = nn.Linear(hidden_size, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inp, batch_size):
        '''
        input : inp <torch tensor> : Input matrix of dimension (batch size, embedding size)
        input : batch_size <int> : Batch size

        return : output <torch tensor> : Ouput matrix of dimension (batch size, num classes) 
        '''
        inp = inp.view(batch_size, 1, self.embed_size)
        output = inp.float()
        #[batch_size, 1, hidden_size]
        
        output, _ = self.lstm(output)
        output = output.squeeze(1)
        output = self.softmax(self.out(output))
        return output
        #[batch_size, n_classes]

if __name__ == "__main__":
    x = np.array([[1, 2, 2], [1,2,2]])
    device = torch.device("cuda")
    model = Classifier(embed_size=3, hidden_size=2, n_classes=3).to(device)
    
    x = torch.from_numpy(x).to(device)
    y = model(x, batch_size=2)
    print(y)
