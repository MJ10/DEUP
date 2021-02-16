import torch
import torch.nn as nn
from .base import BaseModel
from torch.utils.data import DataLoader, TensorDataset
from uncertaintylearning.utils import get_ensemble_uncertainty_estimate


class Ensemble(BaseModel):
    def __init__(self, train_X, train_Y,
                 networks,
                 optimizers,
                 batch_size=16,
                 device=torch.device("cpu")):  # For now, the code runs on CPU only, `.to(self.device)` should be added!
        super().__init__()
        self.train_X = train_X
        self.train_Y = train_Y

        self.input_dim = train_X.size(1)
        self.output_dim = train_Y.size(1)

        self.epoch = 0
        self.loss_fn = nn.MSELoss()

        self.f_predictors = networks
        self.device = device
        self.f_optimizers = optimizers

        self.actual_batch_size = min(batch_size, len(self.train_X) // 2)

    def fit(self, epochs=100):
        """
        Update a,f,e predictors with acquired batch
        """

        self.train()
        data = TensorDataset(self.train_X, self.train_Y)

        loader = DataLoader(data, shuffle=True, batch_size=self.actual_batch_size)
        for (predictor, optimizer) in zip(self.f_predictors, self.f_optimizers):
            for epoch in range(epochs):
                for batch_id, (xi, yi) in enumerate(loader):
                    xi, yi = xi.to(self.device), yi.to(self.device)
                    optimizer.zero_grad()
                    y_hat = predictor(xi)
                    f_loss = self.loss_fn(y_hat, yi)
                    f_loss.backward()
                    optimizer.step()

        self.epoch += 1

        return {
            'f': f_loss.detach().item(),
        }

    def get_prediction_with_uncertainty(self, x):
        out = super().get_prediction_with_uncertainty(x)
        if out is None:
            return get_ensemble_uncertainty_estimate(self.f_predictors, x)
        return out

