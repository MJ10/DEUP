from abc import ABC, abstractmethod
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

import torch
import numpy as np


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
