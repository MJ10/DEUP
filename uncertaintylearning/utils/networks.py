import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiplicativeLR

from collections import OrderedDict

class DenseNormalGamma(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DenseNormalGamma, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.total_output_dim = 4 * output_dim
        self.linear = nn.Linear(self.input_dim, self.total_output_dim)

    def forward(self, x):
        x = self.linear(x)
        if len(x.shape) == 1:
            gamma, lognu, logalpha, logbeta = torch.split(x, self.output_dim, dim=0)
        else:
            gamma, lognu, logalpha, logbeta = torch.split(x, self.output_dim, dim=1)
        nu = F.softplus(lognu)
        alpha = F.softplus(logalpha) + 1.
        beta = F.softplus(logbeta)

        return torch.stack([gamma, nu, alpha, beta]).to(x.device)


def create_network(input_dim, output_dim, n_hidden, activation='relu', positive_output=False, hidden_layers=2, dropout_prob=0.0, evidential_reg=False):
    """
    This function instantiates and returns a NN with the corresponding parameters
    """
    if activation == 'relu':
        activation_fn = nn.ReLU
    elif activation == 'tanh':
        activation_fn = nn.Tanh
    elif activation == 'identity':
        activation_fn = nn.Identity
    else:
        raise NotImplementedError("Only 'relu' and 'tanh' activations are supported")
    model = nn.Sequential(OrderedDict([
        ('input_layer', nn.Linear(input_dim, n_hidden)),
        ('activation1', activation_fn()),
        ('dropout1', nn.Dropout(p=dropout_prob)),
    ]))
    for i in range(hidden_layers):
        model.add_module('hidden_layer{}'.format(i + 1), nn.Linear(n_hidden, n_hidden))
        model.add_module('activation{}'.format(i + 2), activation_fn())
        model.add_module('dropout{}'.format(i + 2), nn.Dropout(p=dropout_prob))
    if evidential_reg:
        model.add_module('output_layer', DenseNormalGamma(n_hidden, output_dim))
    else:
        model.add_module('output_layer', nn.Linear(n_hidden, output_dim))
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
