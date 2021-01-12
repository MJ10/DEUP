import torch
import torch.nn as nn
import numpy as np

def get_uncertainty_estimate(model, x, num_samples):
    outputs = np.array([model(x).data.cpu().numpy() for _ in range(num_samples)]).squeeze()
    y_mean = outputs.mean(axis=0)
    y_var = outputs.var(axis=0)
    return y_mean, y_var
