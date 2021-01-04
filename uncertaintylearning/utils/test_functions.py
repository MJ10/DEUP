# File containing test functions for optimization

import numpy as np
import torch


def sinusoid(X, noise):
    # X is a n x 1 tensor
    # outputs a n x 1 tensor
    # Reasonable bounds is (-1, 2)
    return (-torch.sin(5 * X ** 2) - X ** 4 + 0.3 * X ** 3 + 2 * X ** 2 + 4.1 * X +
            noise * torch.randn(X.size(0), 1))


def multi_optima(X, noise):
    # X is a n x 1 tensor
    # outputs a n x 1 tensor
    # Reasonable bounds is (-1, 2)
    return torch.sin(X) * torch.cos(5 * X) * torch.cos(22 * X) + noise * torch.randn(X.size(0), 1)


def levi_n13(X, noise):
    # X is a n x 2 tensor
    # outputs a n x 1 tensor
    # Reasonable bounds is (-4, 4) ^ 2
    x = X[:, 0]
    y = X[:, 1]
    return noise * torch.randn(X.size(0), 1) - (torch.sin(3 * np.pi * x) ** 2 +
                                                (x - 1) ** 2 * (1 + torch.sin(3 * np.pi * y) ** 2) +
                                                (y - 1) ** 2 * (1 + torch.sin(2 * np.pi * y) ** 2)).unsqueeze(1)


def booth(X, noise):
    # X is a n x 2 tensor
    # outputs a n x 1 tensor
    # Reasonable bounds is (-4, 4) ^ 2
    x = X[:, 0]
    y = X[:, 1]
    return noise * torch.randn(X.size(0), 1) - ((x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2).unsqueeze(1)
