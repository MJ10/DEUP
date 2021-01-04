import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiplicativeLR

from collections import OrderedDict


def create_network(input_dim, output_dim, n_hidden, activation='relu', positive_output=False, dropout_prob=0.0):
    """
    This function instantiates and returns a 1 hidden layer NN with the corresponding parameters
    """
    if activation == 'relu':
        activation_fn = nn.ReLU
    elif activation == 'tanh':
        activation_fn = nn.Tanh
    else:
        raise NotImplementedError("Only 'relu' and 'tanh' activations are supported")
    model = nn.Sequential(OrderedDict([
        ('input_layer', nn.Linear(input_dim, n_hidden)),
        ('activation1', activation_fn()),
        ('dropout1', nn.Dropout(p=dropout_prob)),
        ('hidden_layer', nn.Linear(n_hidden, n_hidden)),
        ('activation2', activation_fn()),
        ('dropout2', nn.Dropout(p=dropout_prob)),
        ('output_layer', nn.Linear(n_hidden, output_dim))
    ]))

    if positive_output:
        model.add_module('softplus', nn.Softplus())
    return model


def create_optimizer(network, lr, weight_decay=0, output_weight_decay=None):
    """
    This function instantiates and returns optimizer objects of the input neural network
    """
    assert 'output_layer' in dir(network), "The network doesn't have a child module called output_layer"
    non_output_parameters = [val for key, val in network.named_parameters() if 'output' not in key]
    sub_groups = [{"params": non_output_parameters},
                  {"params": network.output_layer.parameters(),
                   "weight_decay": output_weight_decay if output_weight_decay is not None else weight_decay}]
    optimizer = Adam(sub_groups, lr=lr, weight_decay=weight_decay)
    return optimizer


def create_multiplicative_scheduler(optimizer, lr_schedule):
    if lr_schedule is None:
        lr_schedule = 1
    return MultiplicativeLR(optimizer, lr_lambda=lambda epoch: lr_schedule)


def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
