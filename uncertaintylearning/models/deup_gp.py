import torch
import logging
import torch.nn as nn
from uncertaintylearning.utils import softplus
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
import numpy as np
from .base import BaseModel
from tqdm import tqdm
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood


class DEUP_GP(BaseModel):
    def __init__(self, data, feature_generator=None,
                 exp_pred_uncert=False, no_deup=False):
        super().__init__()
        self.feature_generator = feature_generator
        self.x = data['x']
        self.y = data['y']

        self.input_dim = self.x.size(1)
        self.output_dim = self.y.size(1)

        self.f_predictor = None
        self.e_predictor = None

        self.no_deup = no_deup
        self.exp_pred_uncert = exp_pred_uncert

    def fit(self):
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        self.f_predictor = SingleTaskGP(self.x, self.y, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(self.f_predictor.likelihood, self.f_predictor)
        fit_gpytorch_model(mll)

    def fit_uncertainty_estimator(self, features=None, targets=None):
        if self.no_deup:
            return None
        self.e_predictor = SingleTaskGP(features, targets)
        mll = ExactMarginalLogLikelihood(self.e_predictor.likelihood, self.e_predictor)
        fit_gpytorch_model(mll)

    def _uncertainty(self, features=None, x=None,
                     use_raw=False, unseen=True):
        if self.no_deup:
            assert x is not None
            predicted_uncertainties = self.f_predictor(x).variance.unsqueeze(-1)
        else:
            assert (features is None) ^ (x is None), "Exactly one of `features` and `x` shouldn't be None"
            if features is None:
                features = self.feature_generator(x, unseen=unseen)
            predicted_uncertainties = self.e_predictor(features).mean
        if not self.exp_pred_uncert and not use_raw:
            predicted_uncertainties = nn.ReLU()(predicted_uncertainties)
        else:
            predicted_uncertainties.clamp_(-20, 3)
        if self.exp_pred_uncert and not use_raw:
            predicted_uncertainties = predicted_uncertainties.exp()
        return predicted_uncertainties

    def get_prediction_with_uncertainty(self, x, features=None, **kwargs):
        out = super().get_prediction_with_uncertainty(x)
        if out is None:
            pred = self.f_predictor(x).mean.unsqueeze(-1)
            return pred, self._uncertainty(features, x if features is None else None)
        return out

    def predict(self, x, return_std=False, features=None):
        mean, var = self.get_prediction_with_uncertainty(x, features)
        if return_std:
            return mean.cpu(), var.cpu().sqrt()
        else:
            return mean.cpu()
