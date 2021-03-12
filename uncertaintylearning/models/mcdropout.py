import torch
import torch.nn as nn
from .base import BaseModel
from torch.utils.data import DataLoader, TensorDataset
from uncertaintylearning.utils import get_dropout_uncertainty_estimate


class MCDropout(BaseModel):
    def __init__(self, train_X, train_Y,
                 network,
                 optimizer,
                 batch_size=16,
                 device=torch.device("cpu")):
        super().__init__()
        self.train_X = train_X
        self.train_Y = train_Y

        self.input_dim = train_X.size(1)
        self.output_dim = train_Y.size(1)

        self.epoch = 0
        self.loss_fn = nn.MSELoss()

        self.f_predictor = network
        self.device = device

        self.f_optimizer = optimizer

        self.actual_batch_size = min(batch_size, len(self.train_X) // 2)

    def fit(self, epochs=100):
        """
        Update a,f,e predictors with acquired batch
        """

        self.train()
        data = TensorDataset(self.train_X, self.train_Y)

        loader = DataLoader(data, shuffle=True, batch_size=self.actual_batch_size)
        for epoch in range(epochs):
            for batch_id, (xi, yi) in enumerate(loader):
                xi = xi.to(self.device)
                yi = yi.to(self.device)
                self.f_optimizer.zero_grad()
                y_hat = self.f_predictor(xi)
                f_loss = self.loss_fn(y_hat, yi)
                f_loss.backward()
                self.f_optimizer.step()

            self.epoch += 1

        return {
            'f': f_loss.detach().item(),
        }

    def _uncertainty(self, x, num_samples=100):
        """
        Computes uncertainty for input sample and
        returns total uncertainty estimate.
        """
        _, var = get_dropout_uncertainty_estimate(self.f_predictor, x, num_samples=num_samples)
        var = var ** 2
        return var

    def get_prediction_with_uncertainty(self, x):
        out = super().get_prediction_with_uncertainty(x)
        if out is None:
            self.eval()
            mean = self.f_predictor(x)
            self.train()
            var = self._uncertainty(x)
            return mean, var
        return out

