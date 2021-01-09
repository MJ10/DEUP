import torch
import torch.nn as nn
import numpy as np
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal

from torch.utils.data import DataLoader, TensorDataset
from uncertaintylearning.utils import get_uncertainty_estimate


class MCDropout(Model):
    def __init__(self, train_X, train_Y,
                 network,
                 optimizer,
                 scheduler=None,
                 batch_size=16,
                 device=torch.device("cpu")):  # For now, the code runs on CPU only, `.to(self.device)` should be added!
        super(MCDropout, self).__init__()
        self.train_X = train_X
        self.train_Y = train_Y

        self.is_fitted = False

        self.input_dim = train_X.size(1)
        self.output_dim = train_Y.size(1)

        self.epoch = 0
        self.loss_fn = nn.MSELoss()

        self.f_predictor = network

        self.f_optimizer = optimizer

        self.actual_batch_size = min(batch_size, len(self.train_X) // 2)

        # For now, the only implemented version uses self.scheduler = None.
        # The code should be revisited if we want a scheduler
        self.scheduler = scheduler
        if scheduler is None:
            self.scheduler = {}

    @property
    def num_outputs(self):
        return self.output_dim

    def fit(self):
        """
        Update a,f,e predictors with acquired batch
        """

        self.train()
        data = TensorDataset(self.train_X, self.train_Y)

        loader = DataLoader(data, shuffle=True, batch_size=self.actual_batch_size)
        for batch_id, (xi, yi) in enumerate(loader):
            self.f_optimizer.zero_grad()
            y_hat = self.f_predictor(xi)
            f_loss = self.loss_fn(y_hat, yi)
            f_loss.backward()
            self.f_optimizer.step()

        self.epoch += 1
        for scheduler in self.scheduler.values():
            scheduler.step()

        self.is_fitted = True
        return {
            'f': f_loss.detach().item(),
        }

    def _epistemic_uncertainty(self, x):
        """
        Computes uncertainty for input sample and
        returns epistemic uncertainty estimate.
        """
        _, std = get_uncertainty_estimate(self.f_predictor, x, num_samples=100)
        std = torch.FloatTensor(std).unsqueeze(-1)
        return std

    def get_prediction_with_uncertainty(self, x):
        if not self.is_fitted:
            raise Exception('Model not fitted')
        self.eval()
        mean = self.f_predictor(x)
        self.train()
        std =  self._epistemic_uncertainty(x)
        return mean, std

    # def predict(self, x, return_std=False):
    #     # x should be a n x d tensor
    #     if not self.is_fitted:
    #         raise Exception('Model not fitted')
    #     # self.eval()
    #     if not return_std:
    #         self.eval()
    #         mean = self.f_predictor(x).detach()
    #         self.train()
    #         return mean
    #     else:
    #         mean, std = self.get_prediction_with_uncertainty(x)
    #         return mean.detach(), std.detach()

    def posterior(self, x):
        # this works with 1d output only
        # x should be a n x d tensor
        mvn = self.forward(x)
        return GPyTorchPosterior(mvn)

    def forward(self, x):
        if x.ndim == 3:
            assert x.size(1) == 1
            return self.forward(x.squeeze(1))
        means, std = self.get_prediction_with_uncertainty(x)
        variances = std ** 2
        mvn = MultivariateNormal(means, variances.unsqueeze(-1))
        return mvn
