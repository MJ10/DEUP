import torch
import torch.nn as nn
import numpy as np
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch.distributions import Normal

from torch.utils.data import DataLoader, TensorDataset

class EvidentialRegression(Model):
    def __init__(self, train_X, train_Y,
            network,  # dict with keys 'a_predictor', 'e_predictor' and 'f_predictor'
            optimizer,  # dict with keys 'a_optimizer', 'e_optimizer' and 'f_optimizer'
            scheduler=None,
            batch_size=16,
            reg_coefficient=1,
            kl_reg=False,
            omega=0.01,
            device=torch.device("cpu")):
        super(EvidentialRegression, self).__init__()
        self.train_X = train_X
        self.train_Y = train_Y

        self.is_fitted = False

        self.input_dim = train_X.size(1)
        self.output_dim = train_Y.size(1)

        self.epoch = 0

        self.f_predictor = network

        self.f_optimizer = optimizer
        
        self.actual_batch_size = min(batch_size, len(self.train_X) // 2)
        self.omega = 0.01
        self.kl_reg = kl_reg
        self.reg_coefficient = reg_coefficient

        self.evidence = torch.nn.Softplus()

        self.scheduler = scheduler
        if scheduler is None:
            self.scheduler = {}
    
    @property
    def num_outputs(self):
        return self.output_dim

    def loss(self, ops, y):
        alpha, beta, gamma, v = ops
        twoBlambda = 2 * beta * (1 + v)
        error = torch.abs(y - gamma)
        nll = 0.5 * torch.log(np.pi / v) \
            - alpha * torch.log(twoBlambda) \
            + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda) \
            + torch.lgamma(alpha) \
            - torch.lgamma(alpha + 0.5)
        if self.kl_reg:
            kl = self.get_reg_kl(mu1, v1, a1, b1, mu2, v2, a2, b2)
            reg = error * kl
        else:
            reg = error * (2 * v + alpha)
        return (nll + self.reg_coefficient * reg).mean()

    def fit(self):
        self.train()
        data = TensorDataset(self.train_X, self.train_Y)

        loader = DataLoader(data, shuffle=True, batch_size=self.actual_batch_size)
        for batch_id, (xi, yi) in enumerate(loader):
            self.f_optimizer.zero_grad()
            ops = self.get_outputs(xi)
            # print(ops)
            f_loss = self.loss(ops, yi)
            # print(f_loss)
            f_loss.backward()
            self.f_optimizer.step()

        self.epoch += 1
        for scheduler in self.scheduler.values():
            scheduler.step()

        self.is_fitted = True
        return {
            'f': f_loss.detach().item(),
        }
    
    def get_outputs(self, x):
        ops = self.f_predictor(x)
        logalpha, logbeta, gamma, logv = ops[:, 0], ops[:, 1], ops[:, 2], ops[:, 3]
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return alpha, beta, gamma, v

    def get_prediction_with_uncertainty(self, x):
        alpha, beta, gamma, v = self.get_outputs(x)
        mean = gamma
        var = beta / (v  * (alpha - 1))
        return mean, var

    def posterior(self, x):
        # this works with 1d output only
        # x should be a n x d tensor
        mvn = self.forward(x)
        return GPyTorchPosterior(mvn)

    def forward(self, x):
        if x.ndim == 3:
            assert x.size(1) == 1
            return self.forward(x.squeeze(1))
        means, var = self.get_prediction_with_uncertainty(x)
        # variances = std ** 2
        mvn = Normal(means, var)
        return mvn
