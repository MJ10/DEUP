from .uncertainty_estimation_utils import get_dropout_uncertainty_estimate, get_ensemble_uncertainty_estimate
from .networks import create_network, create_optimizer, reset_weights, create_wrapped_network, create_epistemic_pred_network
from .maf import MAFMOG

import torch


def softplus(x, beta=1):
    return 1. / beta * (torch.log((beta * x).exp() + 1))


def invsoftplus(x, beta=1):
    return 1. / beta * (torch.log((beta * x).exp() - 1))

