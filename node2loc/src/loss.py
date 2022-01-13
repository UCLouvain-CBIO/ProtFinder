import torch
import torch.nn as nn

def classification_loss(pred, target):
    cross_entropy = nn.CrossEntropyLoss()

    #Pick the argmax of target
    _, class_val = target.max(dim=1)

    return cross_entropy(pred, class_val)

def get_correct(pred, target):
    n_points = target.shape[0]

    #Pick the argmax of target and pred
    _, target = target.max(dim=1)
    _, pred = pred.max(dim=1)

    n_correct = (pred == target).float().sum()

    return n_correct, n_points
