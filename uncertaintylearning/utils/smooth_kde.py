from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity

import numpy as np


class SmoothKDE(BaseEstimator):
    """
    Smoothing a Kernel Dernsity estimator with a Gaussian fit on data
    """
    def __init__(self, kernel='gaussian', bandwidth=.2, alpha=.5):
        super().__init__()
        self.kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        self.gmm = GaussianMixture()
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, x):
        self.kde.fit(x)
        self.gmm.fit(x)
        return self

    def score_samples(self, x):
        return np.logaddexp(np.log(self.alpha) + self.kde.score_samples(x),
                            np.log(1 - self.alpha) + self.gmm.score_samples(x))

    def score(self, x):
        return np.sum(self.score_samples(x))
