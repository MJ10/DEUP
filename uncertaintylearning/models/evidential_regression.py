import torch
import numpy as np
from .base import BaseModel
from torch.utils.data import DataLoader, TensorDataset


class EvidentialRegression(BaseModel):
    def __init__(self, train_X, train_Y,
                 network,
                 optimizer,
                 batch_size=16,
                 reg_coefficient=1,
                 kl_reg=False,
                 omega=0.01,
                 device=torch.device("cpu")):
        super(EvidentialRegression, self).__init__()
        self.train_X = train_X
        self.train_Y = train_Y

        self.input_dim = train_X.size(1)
        self.output_dim = train_Y.size(1)

        self.epoch = 0

        self.f_predictor = network
        self.device = device

        self.f_predictor.to(device)

        self.f_optimizer = optimizer

        self.actual_batch_size = min(batch_size, len(self.train_X) // 2)
        self.omega = 0.01
        self.kl_reg = kl_reg
        self.reg_coefficient = reg_coefficient

        self.evidence = torch.nn.Softplus()

    def get_reg_kl(self, **kwargs):
        # kwargs should be a dict with values mu1, v1, a1, b1, mu2, v2, a2, b2
        # TODO: Finish this implementation
        return 0

    def loss(self, ops, y, **kwargs):
        alpha, beta, gamma, v = ops
        twoBlambda = 2 * beta * (1 + v)
        error = torch.abs(y - gamma)

        nll = 0.5 * torch.log(np.pi / v) \
              - alpha * torch.log(twoBlambda) \
              + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda) \
              + torch.lgamma(alpha) \
              - torch.lgamma(alpha + 0.5)

        if self.kl_reg:
            kl = self.get_reg_kl(**kwargs)  # TODO: add support for this
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
            yi = yi.to(self.device)
            ops = self.get_outputs(xi)
            # print(ops)
            f_loss = self.loss(ops, yi)
            # print(f_loss)
            f_loss.backward()
            self.f_optimizer.step()

        self.epoch += 1

        return {
            'f': f_loss.detach().item(),
        }

    def get_outputs(self, x):
        x = x.to(self.device)
        gamma, v, alpha, beta = self.f_predictor(x)
        return alpha, beta, gamma, v

    def get_prediction_with_uncertainty(self, x):
        out = super().get_prediction_with_uncertainty(x)
        if out is None:
            alpha, beta, gamma, v = self.get_outputs(x)
            mean = gamma
            var = beta / (v * (alpha - 1))
            return mean, var
        return out

