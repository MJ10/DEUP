import torch
import logging
import torch.nn as nn
from uncertaintylearning.utils import softplus
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from .base import BaseModel
from tqdm import tqdm


class DEUP(BaseModel):
    def __init__(self,
                 data,  # dict with keys 'dataloader' or 'x' and 'y'
                 feature_generator,  # Instance of FeatureGenerator (..utils.feature_generator)
                 networks,  # dict with keys 'e_predictor' (optional) and 'f_predictor' (required)
                 optimizers,  # dict with keys 'e_optimizer' and 'f_optimizer'
                 batch_size=16,
                 dataloader_seed=1,
                 loss_fn=nn.MSELoss(),
                 device=torch.device("cpu"),
                 exp_pred_uncert=False,  # If true, exp(uncertainties) is returned instead of uncertainties
                 beta=None,  # If beta is not none, softplus(uncertainties, beta) is returned instead of uncertainties
                 estimator='nn',
                 one_hot_labels=False,
                 num_classes=10,
                 reduce_loss=False,
                 e_loss_fn=None,
                 **kwargs  # kwargs for estimator LinearRegression and GaussianProcessRegressor
                 ):
        super().__init__()

        self.device = device
        self.feature_generator = feature_generator
        self.one_hot_labels = one_hot_labels
        self.num_classes = num_classes
        self.reduce_loss = reduce_loss
        self.e_loss_fn = e_loss_fn

        if 'dataloader' in data.keys():
            self.use_dataloader = True
            self.dataloader = data['dataloader']
            self.actual_batch_size = batch_size
        else:
            self.x = data['x'] #.to(device)
            self.y = data['y'] #.to(device)
            self.use_dataloader = False

            self.input_dim = self.x.size(1)
            self.output_dim = self.y.size(1)

            self.actual_batch_size = min(batch_size, self.x.size(0) // 2)
            assert self.actual_batch_size >= 1, "Need more input points initially !"
            self.dataloader_seed = dataloader_seed
            torch.manual_seed(self.dataloader_seed)

        self.loss_fn = loss_fn

        self.f_predictor = networks['f_predictor']
        self.f_optimizer = optimizers['f_optimizer']


        self.exp_pred_uncert = exp_pred_uncert

        assert estimator in ('nn', 'linreg', 'gp')
        self.estimator = estimator
        if estimator == 'nn':
            self.e_predictor = networks['e_predictor']
            self.e_optimizer = optimizers['e_optimizer']
        elif estimator == 'linreg':
            self.e_predictor = LinearRegression(**kwargs)
        elif estimator == 'gp':
            self.e_predictor = GaussianProcessRegressor(**kwargs)
        else:
            raise NotImplementedError('`estimator` has to be one of `nn`, `linreg`, `gp')

        self.beta = beta
        if self.beta is not None:
            assert not self.exp_pred_uncert, "either log or invsoftplus"

    def fit(self, epochs=None, progress=False):
        """
        Update a,f,e predictors with acquired batch
        """
        if epochs is None:
            epochs = 1
        self.f_predictor.train()
        if not self.use_dataloader:
            data = TensorDataset(self.x, self.y)
            loader = DataLoader(data, shuffle=True, batch_size=self.actual_batch_size)
        else:
            loader = self.dataloader
        logging.info('Training main predictor')
        train_losses = []
        iterable = range(epochs)
        if progress:
            iterable = tqdm(iterable)
        for epoch in iterable:
            epoch_losses = []
            iterable2 = enumerate(loader)
            if progress:
                iterable2 = tqdm(iterable2)
            for batch_id, (xi, yi) in iterable2:
                f_loss = self.train_with_batch(xi, yi)
                epoch_losses.append(f_loss.item())
                break
            train_losses.append(np.mean(epoch_losses))

        return train_losses

    def train_with_batch(self, xi, yi):
        if self.one_hot_labels:
            yi = torch.nn.functional.one_hot(yi, self.num_classes).to(torch.float32)
        xi, yi = xi.to(self.device), yi.to(self.device)
        self.f_optimizer.zero_grad()
        y_hat = self.f_predictor(xi)
        f_loss = self.loss_fn(y_hat, yi)
        if self.reduce_loss:
            f_loss = f_loss.sum(1).mean(0)
        f_loss.backward()
        self.f_optimizer.step()

        return f_loss

    def fit_uncertainty_estimator(self, features, targets, epochs=None):
        """
        features should be a list of n x 1 tensors, and maybe one n x d tensor if the input is used as feature
        targets should be a list of n x 1 tensors (should be errors made by the main predictor)
        """
        features = features.to(self.device)
        targets = targets.to(self.device)
        if self.estimator != 'nn':
            self.e_predictor.fit(features, targets)
            train_losses = [np.mean((self.e_predictor.predict(features) - targets.numpy()) ** 2)]
            return train_losses
        if epochs is None:
            epochs = 1

        self.e_predictor.train()
        batch_size = min(self.actual_batch_size, targets.size(0) // 2)
        train_losses = []
        loader = DataLoader(TensorDataset(features, targets), shuffle=True, batch_size=batch_size)
        for epoch in range(epochs):
            epoch_losses = []
            for batch_id, (features_i, target_i) in enumerate(loader):
                e_loss = self.train_uncertainty_estimator_batch(features_i, target_i)
                epoch_losses.append(e_loss.item())
            train_losses.append(np.mean(epoch_losses))

        return train_losses

    def train_uncertainty_estimator_batch(self, features_i, target_i):
        self.e_optimizer.zero_grad()
        loss_hat = self._uncertainty(features=features_i, use_raw=True)
        if self.e_loss_fn is not None:
            e_loss = self.e_loss_fn(loss_hat, target_i)
        else:
            e_loss = self.loss_fn(loss_hat, target_i)
        e_loss.backward()
        self.e_optimizer.step()
        return e_loss

    def _uncertainty(self, features=None, x=None, use_raw=False, unseen=True):
        # Either compute uncertainty using features, or x, not both
        assert (features is None) ^ (x is None), "Exactly one of `features` and `x` shouldn't be None"
        if features is None:
            features = self.feature_generator(x, unseen=unseen, device=self.device)
        features = features.to(self.device)
        if self.estimator == 'nn':
            predicted_uncertainties = self.e_predictor(features)
        else:
            predicted_uncertainties = self.e_predictor.predict(features.cpu().detach())
            predicted_uncertainties = torch.FloatTensor(predicted_uncertainties).to(self.device)
            if not self.exp_pred_uncert and self.beta is None and not use_raw:
                predicted_uncertainties = nn.ReLU()(predicted_uncertainties)
            else:
                predicted_uncertainties = predicted_uncertainties.clamp(-20, 3)
        if self.exp_pred_uncert and not use_raw:
            predicted_uncertainties = predicted_uncertainties.exp()
        elif self.beta is not None and not use_raw:
            predicted_uncertainties = softplus(predicted_uncertainties, beta=self.beta)
        return predicted_uncertainties

    def get_prediction_with_uncertainty(self, x, features=None, **kwargs):
        out = super().get_prediction_with_uncertainty(x)
        if out is None:
            return self.f_predictor(x), self._uncertainty(features, x if features is None else None)
        return out

    def predict(self, x, return_std=False, features=None):
        self.eval()
        if not return_std:
            return self.f_predictor(x).cpu().detach()
        else:
            mean, var = self.get_prediction_with_uncertainty(x, features)
            return mean.cpu().detach(), var.cpu().detach().sqrt()
