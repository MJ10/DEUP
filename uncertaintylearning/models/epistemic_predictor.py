import torch
import torch.nn as nn
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal

from torch.utils.data import DataLoader, TensorDataset


class EpistemicPredictor(Model):
    def __init__(self,
                 x,  # n x d tensor
                 y,  # n x 1 tensor
                 feature_generator,  # Instance of FeatureGenerator (..utils.feature_generator)
                 networks,  # dict with keys 'a_predictor', 'e_predictor' and 'f_predictor'
                 optimizers,  # dict with keys 'a_optimizer', 'e_optimizer' and 'f_optimizer'
                 batch_size=16,
                 dataloader_seed=1,
                 device=torch.device("cpu"),
                 ):
        super().__init__()

        self.device = device
        self.feature_generator = feature_generator

        self.x = x
        self.y = y

        self.is_fitted = False

        self.input_dim = x.size(1)
        self.output_dim = y.size(1)

        self.actual_batch_size = min(batch_size, x.size(0) // 2)
        assert self.actual_batch_size >= 1, "Need more input points initially !"

        self.epoch = 0
        self.loss_fn = nn.MSELoss()

        self.e_predictor = networks['e_predictor']
        self.f_predictor = networks['f_predictor']

        self.e_optimizer = optimizers['e_optimizer']
        self.f_optimizer = optimizers['f_optimizer']

        self.dataloader_seed = dataloader_seed
        torch.manual_seed(self.dataloader_seed)

    @property
    def num_outputs(self):
        return self.output_dim

    def fit(self):
        """
        Update a,f,e predictors with acquired batch
        """

        self.train()
        train_losses = []

        data = TensorDataset(self.x, self.y)

        loader = DataLoader(data, shuffle=True, batch_size=self.actual_batch_size)

        for batch_id, (xi, yi) in enumerate(loader):
            f_loss = self.train_with_batch(xi, yi)

            train_losses.append(f_loss.item())

        self.epoch += 1

        self.is_fitted = True
        return train_losses

    def train_with_batch(self, xi, yi):
        self.f_optimizer.zero_grad()
        y_hat = self.f_predictor(xi)
        f_loss = self.loss_fn(y_hat, yi)
        f_loss.backward()
        self.f_optimizer.step()

        return f_loss

    def fit_uncertainty_estimator(self, features, targets):
        """
        features should be a list of n x 1 tensors, and maybe one n x d tensor if the input is used as feature
        targets should be a list of n x 1 tensors (should be L2 errors made by the main predictor
        """
        # TODO: remove our usage of list of features, and directly make features a n x something tensor
        self.train()
        batch_size = min(self.actual_batch_size, targets.size(0) // 2)
        train_losses = []
        loader = DataLoader(TensorDataset(features, targets), shuffle=True, batch_size=batch_size)

        for batch_id, (features_i, target_i) in enumerate(loader):
            e_loss = self.train_uncertainty_estimator_batch(features_i, target_i)
            train_losses.append(e_loss.item())

        return train_losses

    def train_uncertainty_estimator_batch(self, features_i, target_i):
        self.e_optimizer.zero_grad()
        loss_hat = self._uncertainty(features=features_i)
        e_loss = self.loss_fn(loss_hat, target_i)
        e_loss.backward()
        self.e_optimizer.step()
        return e_loss

    def _uncertainty(self, features=None, x=None):
        # Either compute uncertainty using features, or x, not both
        assert sum(([_ is None for _ in (features, x)])) == 1, "Exactly one of `features` and `x` shouldn't be None"
        if features is None:
            features = self.feature_generator(x)
        return self.e_predictor(features)

    def get_prediction_with_uncertainty(self, x, features=None):
        if x.ndim == 3:
            assert features is None, "x cannot be of 3 dimensions, when features is explicitly given"
            preds = self.get_prediction_with_uncertainty(x.view(x.size(0) * x.size(1), x.size(2)))
            return preds[0].view(x.size(0), x.size(1), 1), preds[1].view(x.size(0), x.size(1), 1)

        if not self.is_fitted:
            raise Exception('Model not fitted')

        return self.f_predictor(x), self._uncertainty(features, x if features is None else None)

    def predict(self, x, return_std=False, features=None):
        # x should be a n x d tensor
        if not self.is_fitted:
            raise Exception('Model not fitted')
        self.eval()
        if not return_std:
            return self.f_predictor(x).detach()
        else:
            mean, var = self.get_prediction_with_uncertainty(x, features)
            return mean.detach(), var.detach().sqrt()

    def posterior(self, x, output_indices=None, observation_noise=False, **kwargs):
        # this works with 1d output only
        # x should be a n x d tensor
        features = None
        if 'features' in kwargs:
            feautres = kwargs['features']
        mvn = self.forward(x, features)
        return GPyTorchPosterior(mvn)

    def forward(self, x, features=None):
        # ONLY WORKS WITH 1d output !!!!!
        # When x is of shape n x d, the posterior should have mean of shape n, and covar of shape n x n (diagonal)
        # When x is of shape n x q x d, the posterior should have mean of shape n x 1, and covar of shape n x q x q ( n diagonals)

        means, variances = self.get_prediction_with_uncertainty(x, features)
        # Sometimes the predicted variances are too low, and MultivariateNormal doesn't accept their range

        # TODO: maybe the two cases can be merged into one with torch.diag_embed
        if means.ndim == 2:
            mvn = MultivariateNormal(means.squeeze(), torch.diag(variances.squeeze() + 1e-6))
        elif means.ndim == 3:
            assert means.size(-1) == variances.size(-1) == 1
            try:
                mvn = MultivariateNormal(means.squeeze(-1), torch.diag_embed(variances.squeeze(-1) + 1e-6))
            except RuntimeError:
                print('RuntimeError')
                print(torch.diag_embed(variances.squeeze(-1)) + 1e-6)
        else:
            raise NotImplementedError("Something is wrong, just cmd+f this error message and you can start debugging.")
        return mvn
