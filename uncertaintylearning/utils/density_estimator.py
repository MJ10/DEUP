from abc import ABC, abstractmethod
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from .maf import MAFMOG, MADEMOG
from .uncertainty_estimation_utils import get_dropout_uncertainty_estimate

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import logging

class DensityEstimator(ABC):
    """
    Base class for all density estimators that the epistemic predictor can use
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, training_points):
        pass

    @abstractmethod
    def score_samples(self, test_points):
        pass


class DistanceEstimator(DensityEstimator):
    def __init__(self):
        self.postprocessor = IdentityPostprocessor()

    def fit(self, training_points):
        self.dim = training_points.shape[1]
        self.training_points = training_points

    def score_samples(self, test_points):
        values = self.postprocessor.transform(torch.cdist(self.training_points, test_points).min(axis=0)[0])
        values = torch.FloatTensor(values).unsqueeze(-1)
        assert values.ndim == 2 and values.size(0) == len(test_points)
        return values


class VarianceSource:
    """
    Variance estimator
    """
    def __init__(self, mcdropout_model, num_samples, epochs=10):
        super().__init__()
        self.var_model = mcdropout_model
        self.num_samples = num_samples
        self.epochs = epochs

    def fit(self, training_points=None):
        for epoch in range(self.epochs):
            self.var_model.fit()

    def score_samples(self, test_points):
        return self.var_model._epistemic_uncertainty(test_points, num_samples=self.num_samples)


class IdentityPostprocessor:
    """
    Simple class that mimics the way MinMaxScaler and other scalers work, with either identity or exp. transformations
    """
    def __init__(self, exponentiate=False):
        self.exponentiate = exponentiate

    def fit(self, values):
        pass

    def transform(self, values):
        if isinstance(values, np.ndarray):
            return self.transform(torch.FloatTensor(values))
        if self.exponentiate:
            return torch.exp(values)
        return values


class NNDensityEstimator(DensityEstimator):
    def __init__(self, batch_size=10, hidden_size=64, n_hidden=2, epochs=10, lr=1e-4,
                 use_log_density=True, use_density_scaling=False):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.lr = lr
        self.use_log_density = use_log_density
        if use_density_scaling:
            if not use_log_density:
                raise NotImplementedError('Cannot use kernel estimation with density scaling without logs')
            self.postprocessor = MinMaxScaler()
        else:
            self.postprocessor = IdentityPostprocessor(exponentiate=not use_log_density)

    def fit(self, training_points, device, path):
        assert self.model is not None
        try:
            self.dataset = TensorDataset(training_points)
            dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        except:
            dataloader = DataLoader(training_points, batch_size=self.batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.model = self.model.to(device)
        logging.info("Training Density Estimator...")
        for epoch in tqdm(range(self.epochs)):
            epoch_loss = 0
            for i, data in enumerate(dataloader):
                # print("Iter {}".format(i))
                self.model.train()

                # check if labeled dataset
                x = data[0]
                x = x.view(x.shape[0], -1)
                x = x.to(device)
                loss = - self.model.log_prob(x, None).mean(0)
                epoch_loss += loss.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i % 25 == 0:
                    print("Iteration: {}, Loss: {}, saving model ...".format(i, epoch_loss / (i+1)))
        
        torch.save(self.model.state_dict(), path)
        # self.postprocessor.fit(self.score_samples(training_points, no_preprocess=True))

    def score_samples(self, test_points, device, no_preprocess=False):
        try: 
            ds = TensorDataset(test_points)
            dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        except:
            dataloader = DataLoader(test_points, batch_size=self.batch_size, shuffle=False)

        logprobs = []
        for data in dataloader:
            x = data[0].view(data[0].shape[0], -1).to(device)
            logprobs.append(self.model.log_prob(x).detach().cpu())
        logprobs = torch.cat(logprobs, dim=0).clamp_min(-5).detach()
        if no_preprocess:
            values = logprobs.numpy().ravel()
        else:
            values = self.postprocessor.transform(logprobs.unsqueeze(-1)).squeeze()
        values = torch.FloatTensor(values).unsqueeze(-1)
        assert values.ndim == 2 and values.size(0) == len(test_points)
        return values


class MAFMOGDensityEstimator(NNDensityEstimator):
    """
    Masked Auto Regressive Flow to estimate log-probabilities, works on 2d or more
    """
    def __init__(self, batch_size=10, n_blocks=5, n_components=1, hidden_size=64,
                 n_hidden=2, batch_norm=False, epochs=10, lr=1e-4, use_log_density=True, use_density_scaling=False):
        super().__init__(batch_size, hidden_size, n_hidden, epochs, lr, use_log_density, use_density_scaling)
        self.n_blocks = n_blocks
        self.n_components = n_components
        self.batch_norm = batch_norm
        # self.model = MAFMOG(self.n_blocks, self.n_components, self.dim, self.hidden_size, self.n_hidden,
        #             batch_norm=self.batch_norm)

    def fit(self, training_points, device, save_path, init_only=False):
        try:
            self.dim = training_points.size(-1)
        except:
            self.dim = training_points[0][0].view(1, -1).size(-1)
        print(self.dim)
        self.model = MAFMOG(self.n_blocks, self.n_components, self.dim, self.hidden_size, self.n_hidden,
                            batch_norm=self.batch_norm)
        if init_only:
            return
        super().fit(training_points, device, save_path)


class MADEMOGDensityEstimator(NNDensityEstimator):
    """
    MADE to estimate log-probabilities, works on 2d or more
    """
    def __init__(self, batch_size=10, n_components=1, hidden_size=64,
                 n_hidden=2, lr=1e-4, epochs=10, use_log_density=True, use_density_scaling=False):
        super().__init__(batch_size, hidden_size, n_hidden, epochs, lr, use_log_density, use_density_scaling)
        self.n_components = n_components


    def fit(self, training_points):
        self.dim = training_points.size(-1)
        self.model = MADEMOG(self.n_components, self.dim, self.hidden_size, self.n_hidden)

        super().fit(training_points)


class KernelDensityEstimator(DensityEstimator):
    """
    Base class for sklearn density estimators
    """
    def __init__(self, use_log_density=True, use_density_scaling=False):
        self.kde = None
        if use_density_scaling:
            if not use_log_density:
                raise NotImplementedError('Cannot use kernel estimation with density scaling without logs')
            self.postprocessor = MinMaxScaler()
        else:
            self.postprocessor = IdentityPostprocessor(exponentiate=not use_log_density)

    def score_samples(self, test_points):
        if isinstance(test_points, torch.Tensor) and test_points.requires_grad:
            return self.score_samples(test_points.detach())

        values = self.postprocessor.transform(self.kde.score_samples(test_points))
        values = torch.FloatTensor(values).unsqueeze(-1)
        assert values.ndim == 2 and values.size(0) == len(test_points)
        return values


class FixedKernelDensityEstimator(KernelDensityEstimator):
    """
    Kernel Density Estimator with fixed kernel
    """
    def __init__(self, kernel, bandwith, use_log_density=True, use_density_scaling=False):
        super().__init__(use_log_density, use_density_scaling)
        self.kde = KernelDensity(kernel=kernel, bandwidth=bandwith)

    def fit(self, training_points):
        self.kde.fit(training_points)
        self.postprocessor.fit(self.kde.score(training_points))


class CVKernelDensityEstimator(KernelDensityEstimator):
    """
    Kernel Density Estimator with grid search and cross validation
    """
    def __init__(self, use_log_density=True, use_density_scaling=False):
        super().__init__(use_log_density, use_density_scaling)
        self.kde = None
        params = {'bandwidth': np.logspace(0, 1, 20), 'kernel': ['exponential', 'tophat', 'gaussian', 'linear']}
        self.grid = GridSearchCV(KernelDensity(), params)

    def fit(self, training_points):
        if isinstance(training_points, torch.Tensor) and training_points.requires_grad:
            return self.fit(training_points.detach())
        self.grid.fit(training_points)
        self.kde = self.grid.best_estimator_
        self.postprocessor.fit(self.kde.score(training_points))
