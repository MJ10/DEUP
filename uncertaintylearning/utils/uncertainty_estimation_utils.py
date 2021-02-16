import torch


def get_dropout_uncertainty_estimate(model, x, num_samples):
    outputs = torch.cat([model(x).unsqueeze(0) for _ in range(num_samples)])
    y_mean = outputs.mean(axis=0)
    y_std = outputs.std(axis=0)
    return y_mean, y_std


def get_ensemble_uncertainty_estimate(models, x):
    outputs = torch.cat([model(x).unsqueeze(0) for model in models], axis=0)
    y_mean = outputs.mean(axis=0)
    y_var = outputs.var(axis=0)
    return y_mean, y_var
